"""Shared visualization helpers (PNG panels, best-frame selection, semantic gate grid)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def upsample_to_fine(arr: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    if arr.ndim == 3:
        return np.stack([upsample_to_fine(arr[c], target_hw) for c in range(arr.shape[0])], 0)
    t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, size=target_hw, mode='nearest').squeeze().numpy()


def normalize_gate_volume(arr) -> Optional[np.ndarray]:
    if arr is None:
        return None
    if hasattr(arr, 'detach'):
        arr = arr.detach().cpu().numpy()
    a = np.asarray(arr, dtype=np.float32)
    while a.ndim > 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 4:
        a = a[0]
    return a


def gate_maps_to_numpy(gm: Dict) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in gm.items():
        a = normalize_gate_volume(v)
        if a is None:
            continue
        if k in ('den_p2', 'dg_logit', 'occ_target'):
            if a.ndim == 3:
                a = a[0] if a.shape[0] == 1 else a.mean(axis=0)
            out[k] = a.astype(np.float32)
        else:
            out[k] = a.astype(np.float32)
    return out


def gate_mean_map(arr: Optional[np.ndarray]) -> np.ndarray:
    vol = normalize_gate_volume(arr)
    if vol is None:
        return None
    if vol.ndim == 3:
        return vol.mean(axis=0)
    return vol


def gate_std_map(arr: Optional[np.ndarray], fine_hw: Tuple[int, int]) -> np.ndarray:
    vol = normalize_gate_volume(arr)
    if vol is None or vol.ndim < 3:
        return np.zeros(fine_hw, dtype=np.float32)
    s = vol.std(axis=0)
    if s.shape != fine_hw:
        s = upsample_to_fine(s, fine_hw)
    return s


def prep_2d_map(arr, fine_hw: Tuple[int, int]) -> np.ndarray:
    vol = normalize_gate_volume(arr)
    if vol is None:
        return np.zeros(fine_hw, dtype=np.float32)
    m = vol.mean(axis=0) if vol.ndim == 3 else vol
    if m.shape != fine_hw:
        m = upsample_to_fine(m, fine_hw)
    return m


def make_height_bev_from_pts(pts: np.ndarray,
                             pc_range: Tuple[float, ...],
                             fine_vs: float,
                             f_nx: int,
                             f_ny: int) -> np.ndarray:
    """(2, H, W): per-cell z_max / z_min normalised to [0, 1] in pc_range z span."""
    z0, z1 = float(pc_range[2]), float(pc_range[5])
    z_range = max(z1 - z0, 1e-3)
    x0, y0 = float(pc_range[0]), float(pc_range[1])
    z_max_map = np.full((f_ny, f_nx), float('-inf'), np.float32)
    z_min_map = np.full((f_ny, f_nx), float('inf'), np.float32)

    mask = ((pts[:, 0] >= pc_range[0]) & (pts[:, 0] < pc_range[3]) &
            (pts[:, 1] >= pc_range[1]) & (pts[:, 1] < pc_range[4]) &
            (pts[:, 2] >= pc_range[2]) & (pts[:, 2] < pc_range[5]))
    p = pts[mask]
    if len(p) == 0:
        return np.zeros((2, f_ny, f_nx), np.float32)

    cxi = np.clip(np.floor((p[:, 0] - x0) / fine_vs).astype(np.int32), 0, f_nx - 1)
    cyi = np.clip(np.floor((p[:, 1] - y0) / fine_vs).astype(np.int32), 0, f_ny - 1)
    np.maximum.at(z_max_map, (cyi, cxi), p[:, 2])
    np.minimum.at(z_min_map, (cyi, cxi), p[:, 2])

    occ = np.isfinite(z_max_map)
    z_max_n = np.where(occ, np.clip((z_max_map - z0) / z_range, 0.0, 1.0), 0.0)
    z_min_n = np.where(occ, np.clip((z_min_map - z0) / z_range, 0.0, 1.0), 0.0)
    return np.stack([z_max_n, z_min_n], axis=0).astype(np.float32)


def _pool_fine_density_p2(den: np.ndarray, fine_hw: Tuple[int, int]) -> np.ndarray:
    """Adaptive pool fine density to ~P2 resolution when den_p2 not in gate_maps."""
    t = torch.from_numpy(np.maximum(den, 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)
    p2_hw = (max(1, fine_hw[0] // 2), max(1, fine_hw[1] // 2))
    pooled = F.adaptive_avg_pool2d(t, p2_hw).squeeze().numpy()
    return upsample_to_fine(pooled, fine_hw)


def _infer_gate_layout(gm: Dict[str, np.ndarray]) -> str:
    has_gs = 'G_s' in gm
    has_gd = 'G_d' in gm
    if has_gs and has_gd:
        return 'dual'
    if has_gd:
        return 'bcaf'
    if has_gs:
        return 'semantic'
    return 'semantic'


def resolve_viz_subdir(args=None, default: str = 'viz_semantic') -> str:
    """Subfolder under eval out_dir for gate visualizations (default: viz_semantic)."""
    import os
    if args is not None and getattr(args, 'viz_subdir', None):
        return str(args.viz_subdir)
    return os.environ.get('VIZ_SUBDIR', default)


def _pick_gate_volume(gm: Dict[str, np.ndarray]):
    """Return (volume, symbol, gate_type) for the primary gate map."""
    if 'G_s' in gm:
        return gm['G_s'], 'G_s', 'semantic'
    if 'G' in gm and 'G_d' not in gm:
        return gm['G'], 'G_s', 'semantic'
    if 'G_d' in gm:
        return gm['G_d'], 'G_d', 'density'
    if 'G' in gm:
        return gm['G'], 'G', 'fusion'
    return None, 'G', 'semantic'


def viz_semantic_gate_grid(density: np.ndarray,
                           gate_maps: Dict[str, np.ndarray],
                           save_path: str,
                           title: str = '',
                           pc_range: Optional[Tuple[float, ...]] = None,
                           model_tag: str = 'SGF') -> None:
    """
    Canonical 2×3 semantic gate panel (reference: noDensity KITTI 003345).

      (a) Fine pillar density [context only]
      (b) G gate mean over neck channels
      (c) G std across neck channels
      (d) G > 0.5 active regions
      (e) density × G overlap
      (f) Fine density log-scale (normalized)
    """
    import matplotlib.pyplot as plt

    if pc_range is None:
        pc_range = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0)
    gm = gate_maps or {}
    fine_hw = density.shape[-2:]
    extent = (pc_range[0], pc_range[3], pc_range[1], pc_range[4])
    den = np.maximum(density, 0)
    den_log = np.log1p(den)
    den_norm = np.clip(den_log / (den_log.max() + 1e-6), 0, 1)

    g_vol, g_sym, g_kind = _pick_gate_volume(gm)
    G_mean = prep_2d_map(g_vol, fine_hw)
    G_std = gate_std_map(g_vol, fine_hw)

    if g_kind == 'semantic':
        b_lbl = (f'(b) {g_sym} = G  semantic self-attention gate\n'
                 f'σ(W_s·f_mid), mean over neck channels')
        c_lbl = (f'(c) {g_sym} std across neck channels\n'
                 'Where attention is most decisive/heterogeneous')
        d_lbl = (f'(d) {g_sym} > 0.5  active gate regions\n'
                 'Binary mask of strong semantic attention')
        e_lbl = (f'(e) density × {g_sym}  overlap\n'
                 'Where scene occupancy meets semantic attention')
    else:
        b_lbl = (f'(b) {g_sym} density gate\n'
                 f'σ(W·D), mean over neck channels')
        c_lbl = (f'(c) {g_sym} std across neck channels\n'
                 'Spatial heterogeneity of gate activation')
        d_lbl = (f'(d) {g_sym} > 0.5  active gate regions\n'
                 'Binary mask of strong density gating')
        e_lbl = (f'(e) density × {g_sym}  overlap\n'
                 'Where occupancy meets density gate')

    panels = [
        ('(a) Fine pillar density [context only]\n'
         'NOT used in gate — purely spatial reference',
         den_log, 'viridis', None, None),
        (b_lbl, G_mean, 'magma', 0, 1),
        (c_lbl, G_std, 'plasma', 0, None),
        (d_lbl, (G_mean > 0.5).astype(np.float32), 'Reds', 0, 1),
        (e_lbl, np.clip(den_norm * G_mean, 0, 1), 'cividis', 0, None),
        ('(f) Fine density log-scale\n'
         'Sparse background / foreground separation',
         den_norm, 'viridis', 0, 1),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax, (sub, arr, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        kw = dict(origin='lower', cmap=cmap, extent=extent, aspect='equal',
                  interpolation='nearest')
        if vmin is not None:
            kw['vmin'] = vmin
        if vmax is not None:
            kw['vmax'] = vmax
        im = ax.imshow(arr.T if arr.ndim == 2 else arr, **kw)
        ax.set_title(sub, fontsize=9)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_linewidth(0.5)
    if title:
        fig.suptitle(title, y=1.01, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def viz_agdpe_semantic_gate_grid(density_ctx: np.ndarray,
                                 d_hat: np.ndarray,
                                 gate_w: np.ndarray,
                                 save_path: str,
                                 title: str = '',
                                 pc_range: Optional[Tuple[float, ...]] = None) -> None:
    """
    AG-DPE 2×3 panel aligned with SGF semantic gate grid.
    gate_w: (3, H, W) softmax weights [w_fine, w_mid, w_coarse].
    density_ctx: GT or pillar density for spatial reference.
    """
    import matplotlib.pyplot as plt

    if pc_range is None:
        pc_range = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0)
    fine_hw = density_ctx.shape[-2:]
    extent = (pc_range[0], pc_range[3], pc_range[1], pc_range[4])
    den = np.maximum(density_ctx, 0)
    den_log = np.log1p(den)
    den_norm = np.clip(den_log / (den_log.max() + 1e-6), 0, 1)
    w = np.clip(gate_w, 0, 1).astype(np.float32)
    if w.ndim == 2:
        w = w[None]
    G_mean = prep_2d_map(w[0], fine_hw)
    G_std = w.std(axis=0) if w.shape[0] > 1 else np.zeros(fine_hw, np.float32)
    if G_std.shape != fine_hw:
        G_std = upsample_to_fine(G_std, fine_hw)

    panels = [
        ('(a) Fine pillar density [context only]\n'
         'GT density supervision / spatial reference',
         den_log, 'viridis', None, None),
        ('(b) w_fine = G  fine-scale softmax gate\n'
         'τ-softmax fusion, fine branch weight',
         G_mean, 'magma', 0, 1),
        ('(c) w std across scales\n'
         'fine / mid / coarse heterogeneity',
         G_std, 'plasma', 0, None),
        ('(d) w_fine > 0.5  active gate regions\n'
         'Binary mask of dominant fine-scale weight',
         (G_mean > 0.5).astype(np.float32), 'Reds', 0, 1),
        ('(e) density × w_fine  overlap\n'
         'Where occupancy meets fine-scale gating',
         np.clip(den_norm * G_mean, 0, 1), 'cividis', 0, None),
        ('(f) Predicted density d̂ log-scale\n'
         'DensityEstimator output (normalized)',
         np.clip(np.log1p(np.maximum(d_hat, 0)) /
                 (np.log1p(np.maximum(d_hat, 0)).max() + 1e-6), 0, 1),
         'viridis', 0, 1),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax, (sub, arr, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        kw = dict(origin='lower', cmap=cmap, extent=extent, aspect='equal',
                  interpolation='nearest')
        if vmin is not None:
            kw['vmin'] = vmin
        if vmax is not None:
            kw['vmax'] = vmax
        im = ax.imshow(arr.T, **kw)
        ax.set_title(sub, fontsize=9)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_linewidth(0.5)
    if title:
        fig.suptitle(title, y=1.01, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def viz_nine_gate_panels(density: np.ndarray,
                         gate_maps: Dict[str, np.ndarray],
                         save_path: str,
                         title: str = '',
                         height_bev: Optional[np.ndarray] = None,
                         layout: str = 'auto',
                         pc_range: Optional[Tuple[float, ...]] = None,
                         fine_vs: float = 0.16) -> None:
    """
    3×3 DC-SGF gate panel (matches EvalV4PlusCarfix / v4plus meaningful layout).

    dual     — G_s + G_d: density, z_max, z_min, P2 D, G_s, G_d, G, occ, |G_s−G_d|
    semantic — G_s only:  density, z_max, z_min, P2 D, G_s, G_s std, G_s mask, occ, density×G_s
    bcaf     — MidFusion: density, z_max, z_min, P2 D, aux foreground, G_d, density×G_d, occ, G_d std
    """
    import matplotlib.pyplot as plt

    if pc_range is None:
        pc_range = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0)
    gm = gate_maps or {}
    if layout == 'auto':
        layout = _infer_gate_layout(gm)

    fine_hw = density.shape[-2:]
    extent = (pc_range[0], pc_range[3], pc_range[1], pc_range[4])
    den = np.maximum(density, 0)
    den_log = np.log1p(den)

    z_max_f = height_bev[0] if height_bev is not None and height_bev.ndim == 3 else None
    z_min_f = height_bev[1] if height_bev is not None and height_bev.ndim == 3 else None

    dp2 = prep_2d_map(gm.get('den_p2'), fine_hw)
    if float(dp2.max()) <= 0:
        dp2 = _pool_fine_density_p2(den, fine_hw)
    occ = prep_2d_map(gm.get('occ_target'), fine_hw)
    if float(occ.max()) <= 0:
        occ = (dp2 > 1e-6).astype(np.float32)

    Gs = prep_2d_map(gm.get('G_s'), fine_hw)
    Gd = prep_2d_map(gm.get('G_d'), fine_hw)
    Gc = prep_2d_map(gm.get('G'), fine_hw)
    if float(Gc.max()) <= 0 and float(Gs.max()) > 0 and float(Gd.max()) > 0:
        Gc = Gs * Gd
    elif float(Gc.max()) <= 0 and float(Gs.max()) > 0:
        Gc = Gs

    if layout == 'dual':
        panels = [
            ('(a) Fine pillar density\n0.16 m grid, log-scaled count',
             den_log, 'viridis', None, None),
            ('(b) Height BEV z_max (norm)\nper-cell max z, gate input ch1',
             z_max_f if z_max_f is not None else np.zeros(fine_hw), 'plasma', 0, 1),
            ('(c) Height BEV z_min (norm)\nper-cell min z, gate input ch2',
             z_min_f if z_min_f is not None else np.zeros(fine_hw), 'plasma', 0, 1),
            ('(d) P2 downsampled density\nadaptive pool → gate input ch0',
             np.log1p(np.maximum(dp2, 0)), 'viridis', None, None),
            ('(e) Semantic gate G_s = σ(W_s·f_mid)\nmean over neck channels',
             Gs, 'magma', 0, 1),
            ('(f) Density gate G_d = σ(W_d·[D,z_max,z_min])\nmean over neck channels',
             Gd, 'magma', 0, 1),
            ('(g) Combined gate G = G_s ⊙ G_d\nmultiplicative fusion at P2',
             Gc, 'magma', 0, 1),
            ('(h) Occupancy target (aux loss)\n1 where den_p2 > 0',
             occ, 'gray', 0, 1),
            ('(i) Gate contrast |G_s − G_d|\nwhere semantic vs density disagree',
             np.abs(Gs - Gd), 'coolwarm', 0, 1),
        ]
    elif layout == 'bcaf':
        dg = prep_2d_map(gm.get('dg_logit'), fine_hw)
        fg = 1.0 / (1.0 + np.exp(-np.clip(dg, -12, 12)))
        Gd_std = gate_std_map(gm.get('G_d'), fine_hw)
        panels = [
            ('(a) Fine pillar density\n0.16 m grid, log-scaled count',
             den_log, 'viridis', None, None),
            ('(b) Height BEV z_max (norm)\nper-cell max z, structural context',
             z_max_f if z_max_f is not None else np.zeros(fine_hw), 'plasma', 0, 1),
            ('(c) Height BEV z_min (norm)\nper-cell min z, ground contact',
             z_min_f if z_min_f is not None else np.zeros(fine_hw), 'plasma', 0, 1),
            ('(d) P2 pooled density D\n3×3 conv gate input (log)',
             np.log1p(np.maximum(dp2, 0)), 'viridis', None, None),
            ('(e) Aux foreground σ(dg_logit)\nobject-likelihood supervision',
             fg, 'magma', 0, 1),
            ('(f) Density gate G_d = σ(W·D)\nmean over neck channels',
             Gd, 'magma', 0, 1),
            ('(g) density × G_d overlap\noccupancy meets gate activation',
             np.clip(den_log / (den_log.max() + 1e-6) * Gd, 0, 1), 'cividis', 0, 1),
            ('(h) Occupancy target\n1 where den_p2 > 0',
             occ, 'gray', 0, 1),
            ('(i) G_d std across channels\nspatial gate heterogeneity',
             Gd_std, 'plasma', 0, None),
        ]
    else:
        Gs_std = gate_std_map(gm.get('G_s') or gm.get('G'), fine_hw)
        panels = [
            ('(a) Fine pillar density\n0.16 m grid, log-scaled count',
             den_log, 'viridis', None, None),
            ('(b) Height BEV z_max (norm)\nper-cell max z, structural context',
             z_max_f if z_max_f is not None else np.zeros(fine_hw), 'plasma', 0, 1),
            ('(c) Height BEV z_min (norm)\nper-cell min z, ground contact',
             z_min_f if z_min_f is not None else np.zeros(fine_hw), 'plasma', 0, 1),
            ('(d) P2 downsampled density\nadaptive pool → contextual reference',
             np.log1p(np.maximum(dp2, 0)), 'viridis', None, None),
            ('(e) Semantic gate G_s = σ(W_s·f_mid)\nmean over neck channels',
             Gs, 'magma', 0, 1),
            ('(f) G_s std across channels\nwhere attention is decisive',
             Gs_std, 'plasma', 0, None),
            ('(g) density × G_s overlap\noccupancy meets semantic attention',
             np.clip(den_log / (den_log.max() + 1e-6) * Gs, 0, 1), 'cividis', 0, 1),
            ('(h) Occupancy mask\n1 where pillar density > 0',
             occ, 'gray', 0, 1),
            ('(i) G_s > 0.5 active regions\nstrong semantic attention',
             (Gs > 0.5).astype(np.float32), 'Reds', 0, 1),
        ]

    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for ax, (sub, arr, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        if arr is None:
            arr = np.zeros(fine_hw, dtype=np.float32)
        kw = dict(origin='lower', cmap=cmap, extent=extent, aspect='equal',
                  interpolation='nearest')
        if vmin is not None:
            kw['vmin'] = vmin
        if vmax is not None:
            kw['vmax'] = vmax
        im = ax.imshow(arr.T if arr.ndim == 2 else arr, **kw)
        ax.set_title(sub, fontsize=9)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_linewidth(0.5)
    if title:
        fig.suptitle(title, y=1.01, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def viz_six_gate_heatmaps(density, gate_maps, save_path, title='',
                          layout='auto', pc_range=None, model_tag='SGF', **kwargs):
    """Backward-compatible alias → canonical 2×3 semantic gate grid."""
    viz_semantic_gate_grid(density, gate_maps, save_path, title=title,
                           pc_range=pc_range, model_tag=model_tag)


def frame_detection_score(gt_boxes: np.ndarray,
                          pred_boxes: np.ndarray,
                          iou_fn) -> float:
    if gt_boxes is None or len(gt_boxes) == 0:
        return 0.0
    gt7 = gt_boxes[:, :7].astype(np.float64)
    if pred_boxes is None or len(pred_boxes) == 0:
        return 0.0
    pb7 = pred_boxes[:, :7].astype(np.float64)
    matched_ious: List[float] = []
    used = set()
    order = np.argsort(-pb7[:, 0])
    for j in order:
        best_iou, best_i = 0.0, -1
        for i in range(len(gt7)):
            if i in used:
                continue
            iou = float(iou_fn(pb7[j], gt7[i]))
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_i >= 0 and best_iou >= 0.25:
            matched_ious.append(best_iou)
            used.add(best_i)
    if not matched_ious:
        return 0.0
    recall = len(matched_ious) / max(len(gt7), 1)
    precision = len(matched_ious) / max(len(pb7), 1)
    mean_iou = float(np.mean(matched_ious))
    return mean_iou * recall * precision


def select_viz_indices(scores: List[Tuple[float, int]], num_viz: int,
                       viz_best: bool) -> List[int]:
    if viz_best:
        ranked = sorted(scores, key=lambda x: (-x[0], x[1]))
        return [i for _, i in ranked[:num_viz]]
    return [i for _, i in scores[:num_viz]]


def _tensor_to_2d(arr, fine_hw: Tuple[int, int]) -> np.ndarray:
    vol = normalize_gate_volume(arr)
    if vol is None:
        return np.zeros(fine_hw, dtype=np.float32)
    if vol.ndim == 3:
        m = vol[0] if vol.shape[0] == 1 else vol.mean(axis=0)
    else:
        m = vol
    if m.shape != fine_hw:
        m = upsample_to_fine(m, fine_hw)
    return m.astype(np.float32)


def viz_agdpe_architecture(d_gt: np.ndarray,
                           d_hat: np.ndarray,
                           g_f: np.ndarray,
                           g_m: np.ndarray,
                           g_c: np.ndarray,
                           save_path: str,
                           title: str = '',
                           pc_range: Optional[Tuple[float, ...]] = None) -> None:
    """Six-panel AG-DPE density + scale-gate figure (paper L6-L18)."""
    import matplotlib.pyplot as plt

    if pc_range is None:
        pc_range = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0)
    fine_hw = d_gt.shape[-2:]
    extent = (pc_range[0], pc_range[3], pc_range[1], pc_range[4])
    d_gt = np.maximum(d_gt, 0)
    d_hat = np.clip(d_hat, 0, 1)
    err = np.abs(d_hat - d_gt)

    panels = [
        ('(a) GT density  tanh(ρ/100)\nsupervision target', d_gt, 'viridis', 0, 1),
        ('(b) Predicted density  d̂\nDensityEstimator L6-L9', d_hat, 'viridis', 0, 1),
        ('(c) |d̂ − d_gt| error\nL_density supervision', err, 'inferno', 0, None),
        ('(d) Fine gate  g_f\nσ(α·d̂ + β_f)  dense regions', g_f, 'magma', 0, 1),
        ('(e) Mid gate  g_m\nσ(α·(0.5−|d̂−0.5|)+β_m)', g_m, 'magma', 0, 1),
        ('(f) Coarse gate  g_c\nσ(α·(1−d̂)+β_c)  sparse regions', g_c, 'magma', 0, 1),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax, (sub, arr, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        kw = dict(origin='lower', cmap=cmap, extent=extent, aspect='equal',
                  interpolation='nearest')
        if vmin is not None:
            kw['vmin'] = vmin
        if vmax is not None:
            kw['vmax'] = vmax
        im = ax.imshow(arr.T, **kw)
        ax.set_title(sub, fontsize=9)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
    if title:
        fig.suptitle(title, y=1.01, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def viz_agdpe_softmax_gates(w_f: np.ndarray,
                            w_m: np.ndarray,
                            w_c: np.ndarray,
                            save_path: str,
                            title: str = '',
                            pc_range: Optional[Tuple[float, ...]] = None) -> None:
    """Three-panel temperature-softmax fusion weights (L14-L18)."""
    import matplotlib.pyplot as plt

    if pc_range is None:
        pc_range = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0)
    extent = (pc_range[0], pc_range[3], pc_range[1], pc_range[4])
    panels = [
        ('(a) w_fine  softmax gate\nfine-scale branch weight', w_f, 'Blues', 0, 1),
        ('(b) w_mid  softmax gate\nmid-scale branch weight', w_m, 'Greens', 0, 1),
        ('(c) w_coarse  softmax gate\ncoarse-scale branch weight', w_c, 'Oranges', 0, 1),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (sub, arr, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(arr.T, origin='lower', cmap=cmap, extent=extent,
                       aspect='equal', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(sub, fontsize=9)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title, y=1.02, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    rgb = np.stack([w_f, w_m, w_c], axis=-1)
    rgb = rgb / (rgb.max() + 1e-6)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower', extent=extent, aspect='equal')
    ax.set_title(f'{title}\nDominant scale (R=fine G=mid B=coarse)' if title else
                 'Dominant scale (R=fine G=mid B=coarse)', fontsize=10)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    base, ext = save_path.rsplit('.', 1)
    plt.savefig(f'{base}_dominant.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)


def viz_heatmap_triplet(hm_gt: np.ndarray,
                        hm_pred: np.ndarray,
                        save_path: str,
                        title: str = '',
                        class_names: Optional[List[str]] = None,
                        pc_range: Optional[Tuple[float, ...]] = None) -> None:
    """GT target heatmap, predicted peak heatmap, and overlay."""
    import matplotlib.pyplot as plt

    if pc_range is None:
        pc_range = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0)
    if class_names is None:
        class_names = ['Car', 'Ped', 'Cyc']
    extent = (pc_range[0], pc_range[3], pc_range[1], pc_range[4])
    gt_max = hm_gt.max(axis=0) if hm_gt.ndim == 3 else hm_gt
    pr_max = hm_pred.max(axis=0) if hm_pred.ndim == 3 else hm_pred

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    panels = [
        ('(a) GT Gaussian targets\nmax over classes', gt_max, 'hot', 0, 1),
        ('(b) Predicted heatmap peaks\nmax over classes', pr_max, 'hot', 0, 1),
        ('(c) GT (green) + Pred (magenta) overlay', None, None, None, None),
    ]
    for ax, (sub, arr, cmap, vmin, vmax) in zip(axes, panels):
        if arr is not None:
            im = ax.imshow(arr.T, origin='lower', cmap=cmap, extent=extent,
                           aspect='equal', interpolation='nearest', vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.imshow(np.zeros_like(gt_max.T), origin='lower', cmap='gray',
                      extent=extent, aspect='equal', alpha=0.2)
            ax.imshow(gt_max.T, origin='lower', cmap='Greens', extent=extent,
                      aspect='equal', interpolation='nearest', alpha=0.55, vmin=0, vmax=1)
            ax.imshow(pr_max.T, origin='lower', cmap='magma', extent=extent,
                      aspect='equal', interpolation='nearest', alpha=0.55, vmin=0, vmax=1)
        ax.set_title(sub, fontsize=9)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')

    cls_txt = '  '.join(f'{n}: GT={hm_gt[i].max():.2f} Pred={hm_pred[i].max():.2f}'
                        for i, n in enumerate(class_names[:hm_gt.shape[0]]))
    if title:
        fig.suptitle(f'{title}\n{cls_txt}', y=1.05, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
