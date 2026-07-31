"""
AG-DPE (Adaptive Geometry Density-guided Pillar Encoder) — Waymo Open Dataset.

Single-file standalone PyTorch port. Architecture identical to the KITTI AG-DPE
best variant:  PillarVFE(9-dim) → AGDPEBackbone (density estimator + 3-scale
gating + SE/CBAM + ResBlock FPN) → AGDPEHead (hm/reg/dir + density + gate losses).
NO dual-grid BCAF neck.

Data layer reuses the Waymo NPZ-cache pipeline from modelv4_midFusion_waymo.py
(parquet decode → per-frame .npz, lazy LRU segment loading, frame_stride sampling).
Points are intensity-normalised; density_gt/density_occ are computed from pts.

Official eval: Waymo LEVEL_1 mAP via sgf_eval_official.evaluate_waymo_official.

────────────────────────────────────────────────────────────────────────────────
  §1   Config constants (Waymo 768×768 @ 0.20 m, 3 classes)
  §2   Waymo parquet parsing + NPZ cache (from midFusion_waymo)
  §3   Data augmentation
  §4   Pillar voxelization
  §5   Density GT map  (make_density_gt_from_pts, FINE_VS pillar area)
  §6   PillarVFE (9-dim, no intensity) + PillarScatter
  §7   AG-DPE backbone modules + AGDPEBackbone
  §8   Gaussian heatmap targets (Waymo class anchors)
  §9   AGDPEHead (hm / reg / dir + density + gate losses)
  §10  AGDPEModel (tau anneal, fp32 VFE guard)
  §11  Rotated BEV IoU + rotated NMS
  §12  AP evaluator (diagnostic + official Waymo LEVEL_1/LEVEL_2)
  §13  Visualization (BEV + density / gate PNGs)
  §14  Dataset + collate  (WaymoAGDPEDataset)
  §15  Trainer
  §16  Evaluator
  §17  Visualizer
  §18  Smoke test
  §19  CLI
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_PORT_CODE = Path('/home/frahman8/scratch/SGF_PortV2/code')
if str(_PORT_CODE) not in sys.path:
    sys.path.insert(0, str(_PORT_CODE))
from sgf_port_data import normalize_point_intensity, resolve_torch_device, HM_BIAS_INIT  # noqa: E402
from dataset_configs import waymo as ds_cfg  # noqa: E402
from dataset_configs._common import expand_per_class  # noqa: E402
from sgf_opcd_protocol import opcd_augment, WAYMO_ROT_RANGE  # noqa: E402
from sgf_gtdb import make_gt_sampler  # noqa: E402

# AMP helpers (PyTorch 1.10 <-> 2.x)
_HAS_NEW_AMP = hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast')

def amp_autocast(use_amp: bool):
    if _HAS_NEW_AMP:
        return torch.amp.autocast('cuda', enabled=use_amp)
    return torch.cuda.amp.autocast(enabled=use_amp)

def amp_grad_scaler(use_amp: bool):
    if _HAS_NEW_AMP:
        return torch.amp.GradScaler('cuda', enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


# ═══════════════════════════════════════════════════════════════════════════════
# §1  CONFIG  (from dataset_configs.waymo)
# ═══════════════════════════════════════════════════════════════════════════════

PC_RANGE   = ds_cfg.PC_RANGE
FINE_VS    = ds_cfg.FINE_VS
NUM_CLASSES = ds_cfg.NUM_CLASSES
CLASS_NAMES = ds_cfg.CLASS_NAMES
CLASS_TO_ID = ds_cfg.CLASS_TO_ID
F_NX = ds_cfg.F_NX
F_NY = ds_cfg.F_NY
MAX_PTS          = ds_cfg.MAX_PTS
MAX_VOXELS_TRAIN = ds_cfg.MAX_VOXELS_TRAIN
MAX_VOXELS_EVAL  = ds_cfg.MAX_VOXELS_EVAL
WAYMO_USE_RETURN2 = ds_cfg.WAYMO_USE_RETURN2
WAYMO_TYPE_MAP = ds_cfg.WAYMO_TYPE_MAP
IOU_THRESH_3D  = ds_cfg.IOU_THRESH_3D
IOU_THRESH_BEV = ds_cfg.IOU_THRESH_BEV
FOCAL_ALPHA    = ds_cfg.FOCAL_ALPHA
_LEGACY_CLS_ALIAS = ds_cfg._LEGACY_CLS_ALIAS
WAYMO_TOP_LASER = ds_cfg.WAYMO_TOP_LASER
CLASS_ANCHORS = ds_cfg.CLASS_ANCHORS
_SMALL_CLS_IDX = ds_cfg._SMALL_CLS_IDX
IOU_ALPHA_DEFAULT = ds_cfg.IOU_ALPHA_DEFAULT
EVAL_SCORE_THRESH_DEFAULT = ds_cfg.EVAL_SCORE_THRESH_DEFAULT


def _normalize_label_cls(cls_name: str) -> str:
    return _LEGACY_CLS_ALIAS.get(cls_name, cls_name)

def _per_class_thresh(val, n: int = NUM_CLASSES, default: float = 0.01):
    if isinstance(val, (int, float)):
        return tuple([float(val)] * n)
    t = tuple(val)
    return (t + tuple([t[-1]] * (n - len(t))))[:n]


# ═══════════════════════════════════════════════════════════════════════════════
# §2  WAYMO PARQUET PARSING + NPZ CACHE
# ═══════════════════════════════════════════════════════════════════════════════

def _require_pyarrow():
    try:
        import pyarrow.parquet as pq; return pq
    except ImportError:
        raise ImportError("pyarrow not found.  Run: module load arrow/17.0.0")


def _load_segment_cache(waymo_root: str, split: str) -> Dict[str, Any]:
    pq = _require_pyarrow()
    root = Path(waymo_root) / split
    cache: Dict[str, Any] = {}
    for seg_file in sorted((root / 'lidar').glob('*.parquet')):
        seg = seg_file.stem
        tc = pq.ParquetFile(root / 'lidar_calibration' / f'{seg}.parquet').read(columns=[
            'key.laser_name',
            '[LiDARCalibrationComponent].extrinsic.transform',
            '[LiDARCalibrationComponent].beam_inclination.values',
        ])
        laser_names = tc['key.laser_name'].to_pylist()
        top_idx = next((i for i, ln in enumerate(laser_names) if int(ln) == WAYMO_TOP_LASER), None)
        if top_idx is None:
            continue
        ext = np.array(
            tc['[LiDARCalibrationComponent].extrinsic.transform'][top_idx].as_py(),
            dtype=np.float64).reshape(4, 4)
        incl = np.array(
            tc['[LiDARCalibrationComponent].beam_inclination.values'][top_idx].as_py(),
            dtype=np.float64)
        cache[seg] = {
            'extrinsic': ext, 'beam_incl': incl,
            'lidar_file': seg_file,
            'box_file': root / 'lidar_box' / f'{seg}.parquet',
            'lidar_tbl': None, 'lidar_row_by_ts': None,
            'box_tbl': None, 'box_rows_by_ts': None,
        }
    return cache


def _ensure_segment_tables(info: Dict[str, Any], pq) -> None:
    if info['lidar_tbl'] is None:
        lidar_tbl = pq.ParquetFile(info['lidar_file']).read(columns=[
            'key.frame_timestamp_micros', 'key.laser_name',
            '[LiDARComponent].range_image_return1.values',
            '[LiDARComponent].range_image_return1.shape',
            '[LiDARComponent].range_image_return2.values',
            '[LiDARComponent].range_image_return2.shape',
        ])
        ts = np.asarray(lidar_tbl['key.frame_timestamp_micros'].to_pylist(), dtype=np.int64)
        laser = np.asarray(lidar_tbl['key.laser_name'].to_pylist(), dtype=np.int32)
        top_rows = np.nonzero(laser == WAYMO_TOP_LASER)[0]
        row_by_ts: Dict[int, int] = {}
        for ridx in top_rows.tolist():
            t = int(ts[ridx])
            if t not in row_by_ts:
                row_by_ts[t] = ridx
        info['lidar_tbl'] = lidar_tbl
        info['lidar_row_by_ts'] = row_by_ts
    if info['box_tbl'] is None:
        box_tbl = pq.ParquetFile(info['box_file']).read(columns=[
            'key.frame_timestamp_micros',
            '[LiDARBoxComponent].type',
            '[LiDARBoxComponent].box.center.x',
            '[LiDARBoxComponent].box.center.y',
            '[LiDARBoxComponent].box.center.z',
            '[LiDARBoxComponent].box.size.x',
            '[LiDARBoxComponent].box.size.y',
            '[LiDARBoxComponent].box.size.z',
            '[LiDARBoxComponent].box.heading',
            '[LiDARBoxComponent].num_top_lidar_points_in_box',
        ])
        ts = np.asarray(box_tbl['key.frame_timestamp_micros'].to_pylist(), dtype=np.int64)
        rows_by_ts: Dict[int, List[int]] = {}
        for ridx, t in enumerate(ts.tolist()):
            rows_by_ts.setdefault(int(t), []).append(ridx)
        info['box_tbl'] = box_tbl
        info['box_rows_by_ts'] = rows_by_ts


def range_image_to_points(values, shape, beam_incl: np.ndarray,
                           extrinsic: np.ndarray) -> np.ndarray:
    H, W, C = shape
    ri = np.asarray(values, dtype=np.float32).reshape(H, W, C)
    ranges = ri[:, :, 0]; intensity = ri[:, :, 1]; in_nlz = ri[:, :, 3]
    azimuths = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float64)
    cos_inc = np.cos(beam_incl)[:, None]; sin_inc = np.sin(beam_incl)[:, None]
    cos_az = np.cos(azimuths)[None, :]; sin_az = np.sin(azimuths)[None, :]
    xs = ranges * cos_inc * cos_az; ys = ranges * cos_inc * sin_az; zs = ranges * sin_inc
    valid = (ranges > 0) & (in_nlz < 1)
    xs_v = xs[valid]; ys_v = ys[valid]; zs_v = zs[valid]; int_v = intensity[valid]
    if len(xs_v) == 0:
        return np.zeros((0, 4), np.float32)
    ones = np.ones(len(xs_v), dtype=np.float64)
    pts_hom = np.column_stack([xs_v, ys_v, zs_v, ones])
    pts_veh = (extrinsic @ pts_hom.T).T[:, :3]
    return np.column_stack([pts_veh.astype(np.float32), int_v[:, None]]).astype(np.float32)


def load_waymo_frame(seg_name: str, timestamp: int,
                     seg_cache: Dict[str, Any]) -> Dict[str, Any]:
    pq = _require_pyarrow()
    info = seg_cache[seg_name]
    _ensure_segment_tables(info, pq)
    tl = info['lidar_tbl']
    row0 = info['lidar_row_by_ts'].get(int(timestamp))
    if row0 is None:
        pts = np.zeros((0, 4), np.float32)
    else:
        pts = range_image_to_points(
            tl['[LiDARComponent].range_image_return1.values'][row0].as_py(),
            list(tl['[LiDARComponent].range_image_return1.shape'][row0].as_py()),
            info['beam_incl'], info['extrinsic'])
        if WAYMO_USE_RETURN2:
            vals2 = tl['[LiDARComponent].range_image_return2.values'][row0].as_py()
            if vals2 is not None and len(vals2) > 0:
                pts2 = range_image_to_points(
                    vals2,
                    list(tl['[LiDARComponent].range_image_return2.shape'][row0].as_py()),
                    info['beam_incl'], info['extrinsic'])
                if len(pts2):
                    pts = np.concatenate([pts, pts2], 0)
    tb = info['box_tbl']
    box_rows = info['box_rows_by_ts'].get(int(timestamp), [])
    boxes, labels = [], []
    for i in box_rows:
        cls = WAYMO_TYPE_MAP.get(int(tb['[LiDARBoxComponent].type'][i].as_py()))
        if cls is None:
            continue
        cx = float(tb['[LiDARBoxComponent].box.center.x'][i].as_py())
        cy = float(tb['[LiDARBoxComponent].box.center.y'][i].as_py())
        cz = float(tb['[LiDARBoxComponent].box.center.z'][i].as_py())
        lx = float(tb['[LiDARBoxComponent].box.size.x'][i].as_py())
        ly = float(tb['[LiDARBoxComponent].box.size.y'][i].as_py())
        lz = float(tb['[LiDARBoxComponent].box.size.z'][i].as_py())
        hd = float(tb['[LiDARBoxComponent].box.heading'][i].as_py())
        n_pts = int(tb['[LiDARBoxComponent].num_top_lidar_points_in_box'][i].as_py())
        from sgf_eval_official import waymo_detection_level
        diff = waymo_detection_level(n_pts)
        if diff == 'IGNORE':
            continue
        boxes.append(np.array([cx, cy, cz, lx, ly, lz, hd, CLASS_TO_ID[cls]], np.float32))
        labels.append({'cls_name': cls, 'difficulty': diff,
                       'box': [cx, cy, cz, lx, ly, lz, hd],
                       'num_lidar_pts': n_pts,
                       'bbox': [0, 0, 1, 1], 'truncated': 0., 'occluded': 0, 'alpha': 0.})
    gt = np.stack(boxes, 0).astype(np.float32) if boxes else np.zeros((0, 8), np.float32)
    sid = f"{seg_name[:16]}_{timestamp}"[:30]
    return {'pts': pts, 'gt_boxes': gt, 'labels': labels, 'sample_id': sid}


def _default_npz_cache() -> str:
    return os.environ.get('WAYMO_NPZ_CACHE',
                          '/home/frahman8/scratch/SGF_PortV2/waymo/cache/npz_v1')

def _frame_cache_path(cache_root: str, seg: str, ts: int) -> Path:
    return Path(cache_root) / seg / f'{ts}.npz'

def _save_frame_npz(path: Path, pts: np.ndarray, gt_boxes: np.ndarray,
                    labels: List[Dict], sample_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pts=pts, gt_boxes=gt_boxes,
                        labels=np.array(json.dumps(labels)), sample_id=sample_id)

def _remap_legacy_gt_boxes(gt_boxes: np.ndarray) -> np.ndarray:
    return gt_boxes

def _load_frame_npz(path: Path) -> Dict[str, Any]:
    d = np.load(path, allow_pickle=False)
    labels_raw = d['labels']
    labels_txt = labels_raw.item() if hasattr(labels_raw, 'item') else str(labels_raw)
    sid_raw = d['sample_id']
    sid = sid_raw.item() if hasattr(sid_raw, 'item') else str(sid_raw)
    labels = json.loads(labels_txt)
    for lb in labels:
        lb['cls_name'] = _normalize_label_cls(lb.get('cls_name', ''))
    from sgf_port_data import normalize_waymo_eval_labels
    labels = normalize_waymo_eval_labels(labels)
    gt_boxes = _remap_legacy_gt_boxes(d['gt_boxes'])
    return {'pts': d['pts'], 'gt_boxes': gt_boxes, 'labels': labels, 'sample_id': sid}

def _unload_segment_tables(info: Dict[str, Any]) -> None:
    info['lidar_tbl'] = None; info['lidar_row_by_ts'] = None
    info['box_tbl'] = None; info['box_rows_by_ts'] = None


class _SegmentTableLRU:
    def __init__(self, seg_cache: Dict[str, Any], max_segments: int = 1):
        self.seg_cache = seg_cache
        self.max_segments = max(1, max_segments)
        self._loaded: List[str] = []

    def ensure_loaded(self, seg: str) -> bool:
        if seg not in self.seg_cache:
            return False
        if seg in self._loaded:
            return True
        while len(self._loaded) >= self.max_segments:
            old = self._loaded.pop(0)
            if old != seg:
                _unload_segment_tables(self.seg_cache[old])
        pq = _require_pyarrow()
        _ensure_segment_tables(self.seg_cache[seg], pq)
        self._loaded.append(seg)
        return True


def _read_split_frames(split_file: str) -> List[Tuple[str, int]]:
    with open(split_file) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return [(parts[0], int(parts[1])) for ln in lines
            for parts in [ln.rsplit(',', 1)] if len(parts) == 2]

def apply_frame_stride(frames: List[Tuple[str, int]],
                       stride: int) -> List[Tuple[str, int]]:
    if stride <= 1:
        return frames
    by_seg: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for seg, ts in frames:
        by_seg[seg].append((seg, ts))
    out: List[Tuple[str, int]] = []
    for seg in sorted(by_seg):
        seg_frames = sorted(by_seg[seg], key=lambda x: x[1])
        out.extend(seg_frames[::stride])
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# §3  AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def augment(pts, boxes, flip_prob=0.5, rot_range=WAYMO_ROT_RANGE, scale_range=(0.95, 1.05)):
    return opcd_augment(pts, boxes, flip_along=('x', 'y'), flip_prob=flip_prob,
                        rot_range=rot_range, scale_range=scale_range)

def _build_gt_sampler(args):
    """Build the KITTI-style GT copy-paste sampler from args (or None)."""
    db_path = getattr(args, 'gt_db_path', None)
    samples = getattr(args, 'gt_samples', None) or {}
    if not db_path or not samples:
        return None
    if not os.path.exists(db_path):
        print(f"[train] WARNING: gt_db_path={db_path} not found; GT-sampling disabled", flush=True)
        return None
    sampler, counts = make_gt_sampler(db_path, {str(k): int(v) for k, v in samples.items()},
                                      class_to_id=CLASS_TO_ID, pc_range=PC_RANGE,
                                      min_points=getattr(args, 'gt_min_points', 5))
    print(f"[train] GT-sampling enabled: targets={samples}  db_counts={counts}", flush=True)
    return sampler

def _waymo_cls_balanced_sampler(train_ds, cache_dir):
    """Scan the npz cache for per-frame classes and build an inverse-frequency sampler."""
    from sgf_opcd_protocol import make_frame_class_balanced_sampler
    print(f"[train] scanning {len(train_ds.frames)} frames for class balance…", flush=True)
    psc = []
    for seg, ts in train_ds.frames:
        cp = _frame_cache_path(cache_dir, seg, ts)
        classes = set()
        if cp.is_file():
            try:
                gb = _load_frame_npz(cp)['gt_boxes']
                for b in gb:
                    ci = int(b[7]) - 1
                    if 0 <= ci < NUM_CLASSES:
                        classes.add(ci)
            except Exception:
                pass
        psc.append(classes)
    print("[train] Waymo class-balanced sampler enabled", flush=True)
    return make_frame_class_balanced_sampler(psc)

def gt_array_for_training(boxes: np.ndarray) -> np.ndarray:
    pc = PC_RANGE
    if boxes.shape[0] == 0:
        return np.zeros((0, 8), np.float32)
    ok = ((boxes[:, 0] >= pc[0]) & (boxes[:, 0] <= pc[3]) &
          (boxes[:, 1] >= pc[1]) & (boxes[:, 1] <= pc[4]))
    return boxes[ok]


# ═══════════════════════════════════════════════════════════════════════════════
# §4  PILLAR VOXELIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def voxelize_pillars(pts, max_voxels=MAX_VOXELS_TRAIN, max_pts=MAX_PTS):
    pc = PC_RANGE
    ok = ((pts[:, 0] >= pc[0]) & (pts[:, 0] < pc[3]) &
          (pts[:, 1] >= pc[1]) & (pts[:, 1] < pc[4]) &
          (pts[:, 2] >= pc[2]) & (pts[:, 2] < pc[5]))
    pts = pts[ok]
    if len(pts) == 0:
        return (np.zeros((0, max_pts, 4), np.float32),
                np.zeros((0, 3), np.int32),
                np.zeros((0,), np.int32))
    xi = np.floor((pts[:, 0] - pc[0]) / FINE_VS).astype(np.int32).clip(0, F_NX - 1)
    yi = np.floor((pts[:, 1] - pc[1]) / FINE_VS).astype(np.int32).clip(0, F_NY - 1)
    flat = yi * F_NX + xi
    uniq, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
    V = len(uniq)
    if V > max_voxels:
        keep = np.argsort(-cnt)[:max_voxels]
        sel = np.zeros(V, dtype=bool); sel[keep] = True
        valid_pt = sel[inv]; pts = pts[valid_pt]; inv = inv[valid_pt]
        o2n = -np.ones(V, dtype=np.int32); o2n[keep] = np.arange(len(keep), dtype=np.int32)
        inv = o2n[inv]; uniq = uniq[keep]; V = len(uniq)
    voxels = np.zeros((V, max_pts, 4), np.float32)
    num_per_v = np.zeros((V,), np.int32)
    order = np.argsort(inv, kind='stable')
    inv_s = inv[order]; pts_s = pts[order]
    is_new = np.concatenate(([True], inv_s[1:] != inv_s[:-1]))
    csum = np.arange(1, len(inv_s) + 1, dtype=np.int32)
    reset_val = np.where(is_new, csum - 1, 0)
    reset_cum = np.maximum.accumulate(reset_val)
    offsets = csum - 1 - reset_cum
    keep_pt = offsets < max_pts
    inv_s = inv_s[keep_pt]; pts_s = pts_s[keep_pt]; offsets = offsets[keep_pt]
    voxels[inv_s, offsets] = pts_s
    np.add.at(num_per_v, inv_s, 1)
    coords = np.zeros((V, 3), np.int32)
    coords[:, 1] = uniq // F_NX; coords[:, 2] = uniq % F_NX
    return voxels.astype(np.float32), coords, num_per_v


# ═══════════════════════════════════════════════════════════════════════════════
# §5  DENSITY GT MAP
# ═══════════════════════════════════════════════════════════════════════════════

def make_density_gt_from_pts(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """AG-DPE density ground truth: d_gt = tanh(rho/100), rho = #pts / FINE_VS^2."""
    pc = PC_RANGE
    x, y = pts[:, 0], pts[:, 1]
    ok = (x >= pc[0]) & (x < pc[3]) & (y >= pc[1]) & (y < pc[4])
    x, y = x[ok], y[ok]
    pillar_area = FINE_VS * FINE_VS
    if len(x) == 0:
        z = np.zeros((1, F_NY, F_NX), np.float32)
        occ = np.zeros((1, F_NY, F_NX), np.float32)
        return z, occ
    xi = np.floor((x - pc[0]) / FINE_VS).astype(np.int32).clip(0, F_NX - 1)
    yi = np.floor((y - pc[1]) / FINE_VS).astype(np.int32).clip(0, F_NY - 1)
    cnt = np.bincount(yi * F_NX + xi, minlength=F_NY * F_NX).astype(np.float32)
    rho = cnt / pillar_area
    d_gt = np.tanh(rho / 100.0).reshape(1, F_NY, F_NX).astype(np.float32)
    occ = (cnt > 0).reshape(1, F_NY, F_NX).astype(np.float32)
    return d_gt, occ


# ═══════════════════════════════════════════════════════════════════════════════
# §6  PILLAR VFE (9-dim, no intensity) + PILLAR SCATTER
# ═══════════════════════════════════════════════════════════════════════════════

class PillarVFE(nn.Module):
    """9-dim: xyz + Δxyz_centroid + Δxy_pillar_center + range (no intensity)."""
    def __init__(self, out_ch: int = 64):
        super().__init__()
        self.linear = nn.Linear(9, out_ch, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.out_ch = out_ch

    def forward(self, voxels: torch.Tensor, num_per_v: torch.Tensor) -> torch.Tensor:
        if voxels.numel() == 0:
            return voxels.new_zeros((0, self.out_ch))
        V, P, _ = voxels.shape
        mask = torch.arange(P, device=voxels.device).unsqueeze(0) < num_per_v.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()
        denom = num_per_v.clamp(min=1).unsqueeze(-1).float()
        centroid = (voxels[..., :3] * mask_f).sum(dim=1) / denom
        f_center = voxels[..., :3] - centroid.unsqueeze(1)
        x_p = (torch.floor((voxels[..., 0] - PC_RANGE[0]) / FINE_VS) * FINE_VS
               + FINE_VS * .5 + PC_RANGE[0])
        y_p = (torch.floor((voxels[..., 1] - PC_RANGE[1]) / FINE_VS) * FINE_VS
               + FINE_VS * .5 + PC_RANGE[1])
        f_pillar = torch.stack([voxels[..., 0] - x_p, voxels[..., 1] - y_p], dim=-1)
        rng = (voxels[..., 0].pow(2) + voxels[..., 1].pow(2)).sqrt().unsqueeze(-1)
        # 9-dim: xyz(3) + f_center(3) + f_pillar(2) + rng(1)
        feat = torch.cat([voxels[..., :3], f_center, f_pillar, rng], dim=-1) * mask_f
        out_flat = self.bn((self.linear(feat) * mask_f).reshape(-1, self.out_ch))
        feat = F.relu(out_flat).reshape(V, P, self.out_ch) * mask_f
        out, _ = feat.max(dim=1)
        return out


class PillarScatter(nn.Module):
    def __init__(self, ch: int = 64, ny: int = F_NY, nx: int = F_NX):
        super().__init__()
        self.ch, self.ny, self.nx = ch, ny, nx

    def forward(self, pillar_feats: torch.Tensor, coords: torch.Tensor,
                batch_size: int) -> torch.Tensor:
        canvas = pillar_feats.new_zeros((batch_size, self.ch, self.ny * self.nx))
        if pillar_feats.numel() == 0:
            return canvas.view(batch_size, self.ch, self.ny, self.nx)
        b_idx = coords[:, 0].long()
        flat = coords[:, 1].long() * self.nx + coords[:, 2].long()
        canvas[b_idx, :, flat] = pillar_feats
        return canvas.view(batch_size, self.ch, self.ny, self.nx)


# ═══════════════════════════════════════════════════════════════════════════════
# §7  AG-DPE BACKBONE
# ═══════════════════════════════════════════════════════════════════════════════

def cbr(i, o, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(i, o, k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class ResBlock2D(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out))
        self.shortcut = (
            nn.Sequential(nn.Conv2d(ch_in, ch_out, 1, stride=stride, bias=False),
                          nn.BatchNorm2d(ch_out))
            if ch_in != ch_out or stride != 1 else nn.Identity())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.conv1(x)) + self.shortcut(x))


class DensityEstimator(nn.Module):
    """Lightweight density branch: 64→32→16→1 with sigmoid."""
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Dropout2d(0.1),
            nn.Conv2d(16, 1, 3, padding=1),
        )
        self.out_bn = nn.BatchNorm2d(1)

    def forward(self, f_init: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.out_bn(self.trunk(f_init)))


class DensityGuidedGating(nn.Module):
    """3-scale density gates with learnable alpha/beta and temperature softmax."""
    def __init__(self, alpha: float = 5.0, beta: float = -2.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta_fine = nn.Parameter(torch.tensor(beta))
        self.beta_mid = nn.Parameter(torch.tensor(beta))
        self.beta_coarse = nn.Parameter(torch.tensor(beta))
        self.tau = 1.0

    def forward(self, d_hat, f_fine, f_mid, f_coarse):
        g_f = torch.sigmoid(self.alpha * d_hat + self.beta_fine)
        g_m = torch.sigmoid(self.alpha * (0.5 - (d_hat - 0.5).abs()) + self.beta_mid)
        g_c = torch.sigmoid(self.alpha * (1.0 - d_hat) + self.beta_coarse)
        gates = torch.cat([g_f, g_m, g_c], dim=1)
        w = F.softmax(gates / max(self.tau, 1e-3), dim=1)
        f_gated = w[:, 0:1] * f_fine + w[:, 1:2] * f_mid + w[:, 2:3] * f_coarse
        self.last_g_f, self.last_g_m, self.last_g_c, self.last_w = g_f, g_m, g_c, w
        return f_gated, w


class DensitySE(nn.Module):
    """SE channel attention conditioned on global mean density."""
    def __init__(self, ch: int = 64, r: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch + 1, r), nn.ReLU(True),
            nn.Linear(r, ch), nn.Sigmoid())

    def forward(self, x, d_hat):
        gap = x.mean(dim=(2, 3))
        d_mean = d_hat.mean(dim=(2, 3))
        s = self.fc(torch.cat([gap, d_mean], dim=1)).unsqueeze(-1).unsqueeze(-1)
        return x * s


class DensityCBAM(nn.Module):
    """Spatial attention with density as extra channel."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 7, padding=3, bias=False)

    def forward(self, x, d_hat):
        m_avg = x.mean(dim=1, keepdim=True)
        m_max, _ = x.max(dim=1, keepdim=True)
        a = torch.sigmoid(self.conv(torch.cat([m_avg, m_max, d_hat], dim=1)))
        return x * a


class AGDPEBackbone(nn.Module):
    """Full AG-DPE feature extractor: scatter BEV → 448-ch multi-scale output."""
    def __init__(self, in_ch: int = 64, out_ch: int = 448):
        super().__init__()
        self.init_conv = cbr(in_ch, 64, 3, 1, 1)
        self.density_est = DensityEstimator()
        self.gating = DensityGuidedGating()
        self.refine1 = nn.Sequential(cbr(64, 64), cbr(64, 64))
        self.se = DensitySE(64)
        self.cbam = DensityCBAM()
        self.rb1 = ResBlock2D(64, 64, stride=1)
        self.rb2 = ResBlock2D(64, 128, stride=2)
        self.rb3 = ResBlock2D(128, 256, stride=2)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 2, stride=2, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True))
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 2, stride=2, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True))
        self.out_ch = out_ch
        self.gate_log: Dict[str, float] = {}
        self.gate_weights: Optional[torch.Tensor] = None
        self.d_hat: Optional[torch.Tensor] = None

    def _ms_pool(self, x):
        f1 = F.avg_pool2d(x, 1, stride=1, padding=0)
        f3 = F.avg_pool2d(x, 3, stride=1, padding=1)
        f5 = F.avg_pool2d(x, 5, stride=1, padding=2)
        return f1, f3, f5

    def forward(self, bev: torch.Tensor):
        f_init = self.init_conv(bev)
        d_hat = self.density_est(f_init)
        self.d_hat = d_hat
        f_fine, f_mid, f_coarse = self._ms_pool(f_init)
        f_gated, w = self.gating(d_hat, f_fine, f_mid, f_coarse)
        self.gate_weights = w
        x = self.refine1(f_gated)
        x = self.se(x, d_hat)
        x = self.cbam(x, d_hat)
        p1 = self.rb1(x)
        p2 = self.rb2(p1)
        p3 = self.rb3(p2)
        p2_up = F.interpolate(p2, size=p1.shape[-2:], mode='bilinear', align_corners=False)
        p3_up = self.up1(self.up2(p3))
        f_ms = torch.cat([p1, p2_up, p3_up], dim=1)
        with torch.no_grad():
            self.gate_log = {
                'D_mean': float(d_hat.mean()),
                'w_fine': float(w[:, 0].mean()),
                'w_mid': float(w[:, 1].mean()),
                'w_coarse': float(w[:, 2].mean()),
                'tau': float(self.gating.tau),
            }
        return f_ms, d_hat, w


# ═══════════════════════════════════════════════════════════════════════════════
# §8  GAUSSIAN HEATMAP TARGETS  (Waymo class anchors)
# ═══════════════════════════════════════════════════════════════════════════════

def _gaussian_radius(dh, dw, min_overlap=0.1):
    a1 = 1;    b1 = dh + dw;        c1 = dh * dw * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(b1 * b1 - 4 * a1 * c1, 0)); r1 = (b1 - sq1) / 2
    a2 = 4;    b2 = 2 * (dh + dw);  c2 = (1 - min_overlap) * dh * dw
    sq2 = math.sqrt(max(b2 * b2 - 4 * a2 * c2, 0)); r2 = (b2 - sq2) / 2
    a3 = 4 * min_overlap; b3 = -2 * min_overlap * (dh + dw); c3 = (min_overlap - 1) * dh * dw
    sq3 = math.sqrt(max(b3 * b3 - 4 * a3 * c3, 0)); r3 = (b3 + sq3) / 2
    return max(1, int(min(r1, r2, r3)))


def _draw_gaussian(hm, cx, cy, r):
    d = 2 * r + 1; s = d / 6; m = r
    y_, x_ = np.ogrid[-m:m + 1, -m:m + 1]
    g = np.exp(-(x_ * x_ + y_ * y_) / (2 * s * s)).astype(np.float32)
    g[g < np.finfo(g.dtype).eps * g.max()] = 0
    H, W = hm.shape
    l = min(cx, r); rb = min(W - cx, r + 1); t = min(cy, r); b = min(H - cy, r + 1)
    if min(rb - (-l), b - (-t)) > 0:
        np.maximum(hm[cy - t:cy + b, cx - l:cx + rb],
                   g[r - t:r + b, r - l:r + rb],
                   out=hm[cy - t:cy + b, cx - l:cx + rb])


def build_targets(gt_boxes: np.ndarray, num_class: int = NUM_CLASSES,
                  H: int = F_NY, W: int = F_NX) -> Tuple[np.ndarray, ...]:
    vs = FINE_VS; x0, y0 = PC_RANGE[0], PC_RANGE[1]
    hm = np.zeros((num_class, H, W), np.float32)
    reg = np.zeros((8, H, W), np.float32)
    iou = np.zeros((1, H, W), np.float32)
    pos = np.zeros((1, H, W), np.float32)
    for box in gt_boxes:
        if len(box) < 8:
            continue
        x, y, z, l, w, h, ry, cls_id = box[:8]
        ci = int(cls_id) - 1
        if ci < 0 or ci >= num_class:
            continue
        cxi, cyi = int((x - x0) / vs), int((y - y0) / vs)
        if not (0 <= cxi < W and 0 <= cyi < H):
            continue
        a = CLASS_ANCHORS[ci]
        cell_cx = (cxi + .5) * vs + x0; cell_cy = (cyi + .5) * vs + y0
        min_ov = 0.1 if ci in _SMALL_CLS_IDX else 0.25
        r = _gaussian_radius(max(1., l / vs), max(1., w / vs), min_overlap=min_ov)
        _draw_gaussian(hm[ci], cxi, cyi, r)
        reg[:, cyi, cxi] = [
            x - cell_cx, y - cell_cy, z - a['z'],
            math.log(max(h, 1e-3)) - math.log(a['h']),
            math.log(max(w, 1e-3)) - math.log(a['w']),
            math.log(max(l, 1e-3)) - math.log(a['l']),
            math.sin(ry), math.cos(ry)]
        iou[0, cyi, cxi] = 1.0
        pos[0, cyi, cxi] = 1.0
    return hm, reg, iou, pos


def decode_class_boxes(reg_ci: torch.Tensor, xi: torch.Tensor,
                       yi: torch.Tensor, ci: int) -> torch.Tensor:
    vs = FINE_VS; x0, y0 = PC_RANGE[0], PC_RANGE[1]; a = CLASS_ANCHORS[ci]
    cell_cx = (xi.float() + .5) * vs + x0; cell_cy = (yi.float() + .5) * vs + y0
    x = cell_cx + reg_ci[0]; y = cell_cy + reg_ci[1]; z = reg_ci[2] + a['z']
    h = (reg_ci[3] + math.log(a['h'])).exp()
    w = (reg_ci[4] + math.log(a['w'])).exp()
    l = (reg_ci[5] + math.log(a['l'])).exp()
    ry = torch.atan2(reg_ci[6], reg_ci[7])
    return torch.stack([x, y, z, l, w, h, ry], dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# §9  AGDPE HEAD
# ═══════════════════════════════════════════════════════════════════════════════

class AGDPEHead(nn.Module):
    """448→64 reduction, hm / reg / dir branches; density + gate supervision."""
    def __init__(self, input_channels: int = 448, num_class: int = NUM_CLASSES,
                 z_h_weight: float = 2.0,
                 score_thresh: Optional[Sequence[float]] = None,
                 nms_iou_thresh: Optional[Sequence[float]] = None):
        super().__init__()
        self.num_class = num_class
        self.w_cls = 1.0; self.w_box = 2.0; self.w_dir = 0.2
        self.w_density = 0.5; self.w_gate = 0.01
        self.register_buffer('reg_ch_weights',
            torch.tensor([1., 1., z_h_weight, z_h_weight, 1., 1., 1., 1.]))
        if score_thresh is None:
            self.score_thresh = expand_per_class(ds_cfg.SCORE_THRESH, num_class)
        else:
            self.score_thresh = expand_per_class(score_thresh, num_class)
        if nms_iou_thresh is None:
            self.nms_iou = expand_per_class(ds_cfg.NMS_IOU, num_class)
        else:
            self.nms_iou = expand_per_class(nms_iou_thresh, num_class)
        self.reduce = nn.Sequential(
            nn.Conv2d(input_channels, 64, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True))
        self.cls = nn.Conv2d(64, num_class, 1)
        nn.init.constant_(self.cls.bias, HM_BIAS_INIT)
        self.reg = nn.ModuleList([nn.Conv2d(64, 8, 1) for _ in range(num_class)])
        self.dir = nn.ModuleList([nn.Conv2d(64, 2, 1) for _ in range(num_class)])
        for ci in range(num_class):
            nn.init.zeros_(self.reg[ci].bias)
            with torch.no_grad():
                self.reg[ci].bias.data[7] = 1.0  # cos→1 → ry≈0 at init

    def forward(self, feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = self.reduce(feats)
        reg_all = torch.stack([self.reg[ci](f) for ci in range(self.num_class)], dim=1)
        dir_all = torch.stack([self.dir[ci](f) for ci in range(self.num_class)], dim=1)
        iou_dummy = reg_all[:, :, :1] * 0
        return {'hm': self.cls(f), 'reg': reg_all, 'dir': dir_all, 'iou': iou_dummy}

    @staticmethod
    def focal_loss(pred, gt, gamma=2.0):
        ps = torch.clamp(torch.sigmoid(pred), 1e-6, 1 - 1e-6)
        total = pred.new_zeros(1)
        for ci, alpha in enumerate(FOCAL_ALPHA[:pred.shape[1]]):
            p = ps[:, ci]; g = gt[:, ci]
            n_pos = (g == 1).sum().float().clamp(1)
            pl = (1 - p).pow(gamma) * torch.log(p) * (g == 1).float()
            nl = (1 - g).pow(4) * p.pow(gamma) * torch.log(1 - p) * (g < 1).float()
            total = total - alpha * (pl.sum() + nl.sum()) / n_pos
        return total / pred.shape[1]

    @staticmethod
    def gate_entropy(w: torch.Tensor) -> torch.Tensor:
        ww = w.clamp(1e-6, 1 - 1e-6)
        return -(ww * ww.log()).sum(dim=1).mean()

    def compute_loss(self, pred, hm_t, reg_t, pos_t, d_hat, d_gt, d_occ,
                     gate_w, gate_log=None):
        l_cls = self.focal_loss(pred['hm'], hm_t)
        l_box = pred['reg'].new_zeros(1)
        l_dir = pred['dir'].new_zeros(1)
        ch_w = self.reg_ch_weights.view(1, 8, 1, 1)
        n_cls = 0
        for ci in range(self.num_class):
            cls_pos = (hm_t[:, ci] >= 1.0 - 1e-4)
            npos = cls_pos.sum()
            if npos == 0:
                continue
            n_cls += 1
            reg_ci = pred['reg'][:, ci]
            diff = F.smooth_l1_loss(reg_ci, reg_t, reduction='none')
            l_box = l_box + (diff * ch_w).sum(dim=1)[cls_pos].sum() / npos.clamp(1)
            dir_t = torch.stack([reg_t[:, 6], reg_t[:, 7]], dim=1)
            dir_p = pred['dir'][:, ci]
            l_dir = l_dir + F.cross_entropy(
                dir_p.permute(0, 2, 3, 1)[cls_pos],
                dir_t.permute(0, 2, 3, 1)[cls_pos].argmax(-1))
        if n_cls > 0:
            l_box = l_box / n_cls; l_dir = l_dir / n_cls
        l_den = F.smooth_l1_loss(d_hat, d_gt, reduction='none')
        l_den = (l_den * d_occ).sum() / d_occ.sum().clamp(1)
        l_gate = self.gate_entropy(gate_w)
        total = (self.w_cls * l_cls + self.w_box * l_box + self.w_dir * l_dir
                 + self.w_density * l_den + self.w_gate * l_gate)
        tb = {'loss_hm': float(l_cls.detach()), 'loss_reg': float(l_box.detach()),
              'loss_dir': float(l_dir.detach()), 'loss_density': float(l_den.detach()),
              'loss_gate': float(l_gate.detach()), 'loss': float(total.detach())}
        if gate_log:
            tb.update({f'gate_{k}': float(v) for k, v in gate_log.items()})
        return total, tb

    @torch.no_grad()
    def post_process(self, pred, use_soft_nms=False):
        hm = torch.sigmoid(pred['hm'])
        reg = pred['reg']
        hm_pool = F.max_pool2d(hm, 3, 1, 1)
        hm_peak = (hm == hm_pool).float() * hm
        B = hm.shape[0]; out = []
        for bi in range(B):
            all_b, all_s, all_l = [], [], []
            for ci, (thr, nt) in enumerate(zip(self.score_thresh, self.nms_iou)):
                sm = hm_peak[bi, ci]
                pos = (sm > thr).nonzero(as_tuple=False)
                if not len(pos):
                    continue
                sc = sm[pos[:, 0], pos[:, 1]]
                if len(sc) > 500:
                    idx = sc.topk(500).indices; pos = pos[idx]; sc = sc[idx]
                r = reg[bi, ci, :, pos[:, 0], pos[:, 1]]
                boxes = decode_class_boxes(r, pos[:, 1], pos[:, 0], ci)
                keep = rotated_nms(boxes, sc, nt, top_k=500)
                if keep.numel() == 0:
                    continue
                all_b.append(boxes[keep]); all_s.append(sc[keep])
                all_l.append(torch.full((len(sc[keep]),), ci + 1,
                                        dtype=torch.long, device=hm.device))
            if all_b:
                out.append({'pred_boxes': torch.cat(all_b, 0),
                            'pred_scores': torch.cat(all_s, 0),
                            'pred_labels': torch.cat(all_l, 0)})
            else:
                dev = hm.device
                out.append({'pred_boxes': torch.zeros(0, 7, device=dev),
                            'pred_scores': torch.zeros(0, device=dev),
                            'pred_labels': torch.zeros(0, dtype=torch.long, device=dev)})
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# §10  AGDPEModel
# ═══════════════════════════════════════════════════════════════════════════════

class AGDPEModel(nn.Module):
    def __init__(self, vfe_ch: int = 64, num_class: int = NUM_CLASSES,
                 score_thresh=None, total_epochs: int = 80):
        super().__init__()
        self.vfe = PillarVFE(out_ch=vfe_ch)
        self.scatter = PillarScatter(ch=vfe_ch, ny=F_NY, nx=F_NX)
        self.backbone = AGDPEBackbone(in_ch=vfe_ch, out_ch=448)
        st = _per_class_thresh(score_thresh) if score_thresh is not None else None
        self.head = AGDPEHead(input_channels=448, num_class=num_class, score_thresh=st)
        self.current_epoch = 0
        self.total_epochs = total_epochs

    def set_epoch(self, epoch: int, total_epochs: int):
        self.current_epoch = epoch
        self.total_epochs = max(total_epochs, 1)
        t = epoch / max(total_epochs - 1, 1)
        self.backbone.gating.tau = 2.0 + (0.5 - 2.0) * t

    def forward(self, batch: Dict) -> Dict:
        with amp_autocast(False):
            pillar_feats = self.vfe(batch['voxels'].float(), batch['num_per_v'])
        bev = self.scatter(pillar_feats, batch['coords'], batch['batch_size'])
        feats, d_hat, gate_w = self.backbone(bev)
        pred = self.head(feats)
        return {'bev': bev, 'feats': feats, 'd_hat': d_hat, 'gate_w': gate_w,
                'gate_log': self.backbone.gate_log, **pred}

    def count_params(self) -> Dict[str, int]:
        return {'vfe': sum(p.numel() for p in self.vfe.parameters()),
                'backbone': sum(p.numel() for p in self.backbone.parameters()),
                'head': sum(p.numel() for p in self.head.parameters()),
                'total': sum(p.numel() for p in self.parameters())}


# ═══════════════════════════════════════════════════════════════════════════════
# §11  ROTATED BEV IoU + ROTATED NMS
# ═══════════════════════════════════════════════════════════════════════════════

def _polygon_clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) > (b[1] - a[1]) * (p[0] - a[0])
    def isect(p1, p2, a, b):
        x1,y1=p1; x2,y2=p2; x3,y3=a; x4,y4=b
        denom=(x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
        if abs(denom)<1e-12: return p1
        t=((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))/denom
        return np.array([x1+t*(x2-x1), y1+t*(y2-y1)])
    out=list(subject); Nc=len(clip)
    for i in range(Nc):
        if not out: return np.zeros((0,2))
        a=clip[i]; b=clip[(i+1)%Nc]; inp=out; out=[]
        for j in range(len(inp)):
            curr=inp[j]; prev=inp[j-1]
            ci_=inside(curr,a,b); pi_=inside(prev,a,b)
            if ci_:
                if not pi_: out.append(isect(prev,curr,a,b))
                out.append(curr)
            elif pi_: out.append(isect(prev,curr,a,b))
    return np.asarray(out) if out else np.zeros((0,2))

def _poly_area(poly: np.ndarray) -> float:
    if len(poly)<3: return 0.0
    x,y=poly[:,0],poly[:,1]
    return float(0.5*abs(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1))))

def box_corners_2d(x,y,l,w,ry) -> np.ndarray:
    c,s=math.cos(ry),math.sin(ry)
    local=np.array([[l/2,w/2],[-l/2,w/2],[-l/2,-w/2],[l/2,-w/2]],dtype=np.float64)
    R=np.array([[c,-s],[s,c]],dtype=np.float64)
    return (R@local.T).T+np.array([x,y])

def rotated_bev_iou(b1: np.ndarray, b2: np.ndarray) -> float:
    p1=box_corners_2d(b1[0],b1[1],b1[3],b1[4],b1[6])
    p2=box_corners_2d(b2[0],b2[1],b2[3],b2[4],b2[6])
    a1=float(b1[3]*b1[4]); a2=float(b2[3]*b2[4])
    inter=_poly_area(_polygon_clip(p1,p2))
    return inter/(a1+a2-inter+1e-9)

def rotated_3d_iou(b1: np.ndarray, b2: np.ndarray) -> float:
    p1=box_corners_2d(b1[0],b1[1],b1[3],b1[4],b1[6])
    p2=box_corners_2d(b2[0],b2[1],b2[3],b2[4],b2[6])
    a_inter=_poly_area(_polygon_clip(p1,p2))
    z1l,z1h=b1[2]-b1[5]/2,b1[2]+b1[5]/2
    z2l,z2h=b2[2]-b2[5]/2,b2[2]+b2[5]/2
    h_inter=max(0.0,min(z1h,z2h)-max(z1l,z2l))
    inter_v=a_inter*h_inter
    v1=b1[3]*b1[4]*b1[5]; v2=b2[3]*b2[4]*b2[5]
    return float(inter_v/(v1+v2-inter_v+1e-9))

def rotated_nms(boxes: torch.Tensor, scores: torch.Tensor,
                iou_thresh: float=0.1, top_k: int=500) -> torch.Tensor:
    if boxes.numel()==0:
        return torch.zeros((0,),dtype=torch.long,device=boxes.device)
    dev=boxes.device; boxes_np=boxes.detach().cpu().numpy()
    scores_np=scores.detach().cpu().numpy()
    order=np.argsort(-scores_np)[:top_k]; suppressed=np.zeros(len(boxes_np),dtype=bool)
    keep: List[int]=[]
    for idx in order:
        if suppressed[idx]: continue
        keep.append(int(idx)); bi=boxes_np[idx]
        bi7=np.concatenate([bi[:7],np.zeros(max(0,7-len(bi)))])
        for jdx in order:
            if suppressed[jdx] or jdx==idx: continue
            bj=boxes_np[jdx]
            bj7=np.concatenate([bj[:7],np.zeros(max(0,7-len(bj)))])
            if rotated_bev_iou(bi7,bj7)>iou_thresh: suppressed[jdx]=True
    return torch.tensor(keep,dtype=torch.long,device=dev)


# ═══════════════════════════════════════════════════════════════════════════════
# §12  AP EVALUATOR  (diagnostic + official Waymo)
# ═══════════════════════════════════════════════════════════════════════════════

def _gt_subset(gts, cls, diff):
    dr={'Easy':0,'Moderate':1,'Hard':2,'Unknown':99,'LEVEL_1':0,'LEVEL_2':1}
    rn=dr.get(diff,99); v,ig=[],[]
    for g in gts:
        if _normalize_label_cls(g.get('cls_name',''))!=cls: continue
        (v if dr.get(g.get('difficulty','Unknown'),99)<=rn else ig).append(g)
    return v,ig

def _ap40(rec, prec):
    if not len(rec): return 0.
    ap=0.
    for r in np.linspace(1/40,1.,40):
        mask=rec>=r
        if mask.any(): ap+=float(prec[mask].max())
    return ap/40.*100.

def evaluate_waymo(all_preds, all_gts, iou_mode='3d'):
    iou_fn = rotated_3d_iou if iou_mode == '3d' else rotated_bev_iou
    thr_map = IOU_THRESH_3D if iou_mode == '3d' else IOU_THRESH_BEV
    results = {}
    for ci, cls in enumerate(CLASS_NAMES):
        thr = thr_map[cls]; results[cls] = {}
        for diff in ('Easy', 'Moderate', 'Hard'):
            tp, fp, sc, n_gt = [], [], [], 0
            for pred, gts in zip(all_preds, all_gts):
                vg, ig = _gt_subset(gts, cls, diff); n_gt += len(vg)
                pb_t = pred.get('pred_boxes', pred.get('boxes'))
                ps_t = pred.get('pred_scores', pred.get('scores'))
                pl_t = pred.get('pred_labels', pred.get('labels'))
                if pb_t is None or len(pb_t) == 0: continue
                if torch.is_tensor(pb_t): pb_t=pb_t.numpy()
                if torch.is_tensor(ps_t): ps_t=ps_t.numpy()
                if torch.is_tensor(pl_t): pl_t=pl_t.numpy()
                mask = pl_t == (ci + 1)
                if not mask.any(): continue
                pb = pb_t[mask]; ps = ps_t[mask]
                matched = np.zeros(len(vg), dtype=bool)
                for bi in np.argsort(-ps):
                    best_iou = 0; best_j = -1
                    for j, g in enumerate(vg):
                        if matched[j]: continue
                        io = iou_fn(pb[bi], np.array(g['box']))
                        if io > best_iou: best_iou = io; best_j = j
                    in_ign = any(iou_fn(pb[bi], np.array(g['box'])) >= thr for g in ig)
                    if best_iou >= thr and best_j >= 0:
                        tp.append(1); fp.append(0); matched[best_j] = True
                    elif in_ign: continue
                    else: tp.append(0); fp.append(1)
                    sc.append(float(ps[bi]))
            if n_gt == 0 or not sc: results[cls][diff] = 0.; continue
            order = np.argsort(-np.array(sc))
            tp_c = np.cumsum(np.array(tp)[order]); fp_c = np.cumsum(np.array(fp)[order])
            rec = tp_c / max(n_gt, 1); prec = tp_c / np.maximum(tp_c + fp_c, 1)
            results[cls][diff] = _ap40(rec, prec)
    return results

def print_ap_table(results):
    hdr = f"{'Class':<14} {'Easy':>8} {'Moderate':>10} {'Hard':>8}"
    print(hdr); print('-' * len(hdr)); mAPs = []
    for cls in CLASS_NAMES:
        r = results.get(cls, {})
        e = r.get('Easy', 0); m = r.get('Moderate', 0); h = r.get('Hard', 0)
        print(f"{cls:<14} {e:8.2f} {m:10.2f} {h:8.2f}"); mAPs.append(m)
    print('-' * len(hdr)); print(f"{'mAP@Mod':<14} {''*8} {np.mean(mAPs):10.2f}")
    return float(np.mean(mAPs))

def print_waymo_official_table(l1, l2, iou_mode='3d'):
    print(f"\n[eval] Official Waymo AP ({iou_mode.upper()}):")
    hdr = f"  {'Class':<14} {'L1':>8} {'L2':>8}"
    print(hdr); print('  ' + '-' * (len(hdr) - 2))
    l1_vals, l2_vals = [], []
    for cls in CLASS_NAMES:
        v1 = l1.get(cls, 0.); v2 = l2.get(cls, 0.)
        print(f"  {cls:<14} {v1:8.2f} {v2:8.2f}")
        l1_vals.append(v1); l2_vals.append(v2)
    mAP_l1 = float(np.mean(l1_vals)); mAP_l2 = float(np.mean(l2_vals))
    print('  ' + '-' * (len(hdr) - 2))
    print(f"  {'mAP':<14} {mAP_l1:8.2f} {mAP_l2:8.2f}")
    return mAP_l1, mAP_l2


# ═══════════════════════════════════════════════════════════════════════════════
# §13  VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

VIZ_IMAGE_EXT = '.png'

def _setup_pub_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'font.family': 'serif', 'font.size': 9,
        'axes.labelsize': 10, 'axes.titlesize': 11,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8, 'grid.alpha': 0.25,
    })

def _box_to_polygon(box): return box_corners_2d(box[0],box[1],box[3],box[4],box[6])

def box_corners_3d(x,y,z,l,w,h,ry) -> np.ndarray:
    c,s=math.cos(ry),math.sin(ry)
    x_c=np.array([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2],dtype=np.float64)
    y_c=np.array([w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2],dtype=np.float64)
    z_c=np.array([h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2],dtype=np.float64)
    R=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]],dtype=np.float64)
    corners=R@np.vstack([x_c,y_c,z_c])
    corners[0]+=x; corners[1]+=y; corners[2]+=z
    return corners.T

def viz_bev_detection(pts, gt_boxes, pred_boxes, pred_scores, pred_labels,
                      save_path: str, title: str = '') -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from matplotlib.lines import Line2D
    _setup_pub_style()
    pc = PC_RANGE
    ok = ((pts[:,0]>=pc[0])&(pts[:,0]<=pc[3])&(pts[:,1]>=pc[1])&(pts[:,1]<=pc[4]))
    p = pts[ok]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(p[:,0], p[:,1], s=0.05, c='#888888', alpha=0.5, linewidths=0, rasterized=True)
    gt_patches=[Polygon(_box_to_polygon(b),closed=True) for b in gt_boxes]
    if gt_patches:
        ax.add_collection(PatchCollection(gt_patches,facecolor='none',
            edgecolor='#1b7e3d',linewidth=1.4,linestyle='-'))
    pred_patches=[Polygon(_box_to_polygon(b),closed=True) for b in pred_boxes]
    if pred_patches:
        ax.add_collection(PatchCollection(pred_patches,facecolor='none',
            edgecolor='#c8302a',linewidth=1.0,linestyle='--'))
        for b,sc,lb in zip(pred_boxes,pred_scores,pred_labels):
            cname=CLASS_NAMES[int(lb)-1] if 1<=int(lb)<=len(CLASS_NAMES) else '?'
            ax.text(b[0],b[1],f'{cname[:3]}\n{sc:.2f}',fontsize=5.5,color='#c8302a',
                    ha='center',va='center',
                    bbox=dict(facecolor='white',edgecolor='none',alpha=0.7,pad=0.6))
    ax.set_xlim(pc[0],pc[3]); ax.set_ylim(pc[1],pc[4])
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    if title: ax.set_title(title)
    ax.grid(True, linewidth=0.4, color='#cccccc')
    ax.legend(handles=[
        Line2D([],[],color='#888888',marker='.',linestyle='None',markersize=3,label='LiDAR'),
        Line2D([],[],color='#1b7e3d',linestyle='-',linewidth=1.4,label='GT'),
        Line2D([],[],color='#c8302a',linestyle='--',linewidth=1.0,label='Pred'),
    ], loc='upper right')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def viz_density_gates(d_hat_np: np.ndarray, gate_w_np: np.ndarray,
                      save_path: str, title: str = '') -> None:
    """Density estimate + 3-scale gate weight maps."""
    import matplotlib.pyplot as plt
    _setup_pub_style()
    panels = [('d_hat', d_hat_np[0, 0]),
              ('w_fine', gate_w_np[0, 0]),
              ('w_mid', gate_w_np[0, 1]),
              ('w_coarse', gate_w_np[0, 2])]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, (name, data) in zip(axes, panels):
        im = ax.imshow(data, origin='lower', cmap='viridis',
                       vmin=0, vmax=1, aspect='auto')
        ax.set_title(f'{name}  μ={data.mean():.3f}')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# §14  DATASET + COLLATE
# ═══════════════════════════════════════════════════════════════════════════════

class WaymoAGDPEDataset(Dataset):
    def __init__(self, waymo_root, split_file, split, training, augment=True,
                 keep_labels_for_eval=False, max_voxels=None,
                 cache_dir: Optional[str] = None, frame_stride: int = 1,
                 require_cache: bool = False, return_pts: bool = False, gt_sampler=None):
        self.waymo_root = waymo_root; self.split = split
        self.training = training; self.augment_flag = augment and training
        self.keep_labels = keep_labels_for_eval
        self.max_voxels = max_voxels or (MAX_VOXELS_TRAIN if training else MAX_VOXELS_EVAL)
        self.cache_dir = cache_dir or _default_npz_cache()
        self.require_cache = require_cache
        self.return_pts = return_pts or (not training)
        self.gt_sampler = gt_sampler if training else None
        self.frames = _read_split_frames(split_file)
        stride = frame_stride if training else 1
        if stride > 1:
            n0 = len(self.frames)
            self.frames = apply_frame_stride(self.frames, stride)
            print(f"[dataset] frame_stride={stride}: {n0} -> {len(self.frames)} frames")
        self.frames.sort(key=lambda x: (x[0], x[1]))
        needed = set(s for s, _ in self.frames)
        if self.require_cache:
            print(f"[dataset] npz-only mode (require_cache); skip parquet calibration "
                  f"({len(needed)} segments)", flush=True)
            self.seg_cache = {}
            self._table_lru = None
        else:
            print(f"[dataset] loading Waymo calibration for {len(needed)} segments ({split})…")
            all_cache = _load_segment_cache(waymo_root, split)
            self.seg_cache = {k: v for k, v in all_cache.items() if k in needed}
            miss = needed - set(self.seg_cache)
            if miss:
                print(f"[dataset] WARNING: {len(miss)} segments not found")
            self._table_lru = _SegmentTableLRU(self.seg_cache, max_segments=1)
        if self.cache_dir:
            n_cached = sum(1 for seg, ts in self.frames
                           if _frame_cache_path(self.cache_dir, seg, ts).is_file())
            print(f"[dataset] npz cache {n_cached}/{len(self.frames)} at {self.cache_dir}")
        print(f"[dataset] {len(self.frames)} frames ready (lazy load).")

    def _load_frame(self, seg: str, ts: int) -> Dict[str, Any]:
        cache_path = _frame_cache_path(self.cache_dir, seg, ts) if self.cache_dir else None
        if cache_path and cache_path.is_file():
            return _load_frame_npz(cache_path)
        if self.require_cache:
            raise FileNotFoundError(f"Missing npz cache: {cache_path}")
        if self._table_lru is None or not self._table_lru.ensure_loaded(seg):
            return {'pts': np.zeros((0, 4), np.float32),
                    'gt_boxes': np.zeros((0, 8), np.float32),
                    'labels': [], 'sample_id': 'missing'}
        data = load_waymo_frame(seg, ts, self.seg_cache)
        if cache_path:
            _save_frame_npz(cache_path, data['pts'], data['gt_boxes'],
                            data['labels'], data['sample_id'])
        return data

    def __len__(self): return len(self.frames)

    def __getitem__(self, idx):
        seg, ts = self.frames[idx]
        data = self._load_frame(seg, ts)
        pts = normalize_point_intensity(data['pts'])
        gt = gt_array_for_training(data['gt_boxes'])
        if self.gt_sampler is not None:
            pts, gt = self.gt_sampler(pts, gt)
        if self.augment_flag:
            pts, gt = augment(pts, gt)
        voxels, coords, num_per_v = voxelize_pillars(pts, self.max_voxels)
        density_gt, density_occ = make_density_gt_from_pts(pts)
        out = {'voxels': voxels, 'coords': coords, 'num_per_v': num_per_v,
               'density_gt': density_gt, 'density_occ': density_occ,
               'gt_boxes': gt, 'sample_id': data['sample_id']}
        if self.training:
            hm_t, reg_t, iou_t, pos_t = build_targets(gt, NUM_CLASSES, F_NY, F_NX)
            out.update({'hm_t': hm_t, 'reg_t': reg_t, 'iou_t': iou_t, 'pos_t': pos_t})
        if self.keep_labels or (not self.training):
            out['labels'] = data['labels']
        if self.return_pts:
            out['pts'] = pts
        return out


def collate_waymo(batch_list):
    B = len(batch_list); vl, cl, nl = [], [], []
    for bi, s in enumerate(batch_list):
        c = s['coords'].copy(); c[:, 0] = bi
        vl.append(s['voxels']); cl.append(c); nl.append(s['num_per_v'])
    voxels = np.concatenate(vl, 0) if vl else np.zeros((0, MAX_PTS, 4), np.float32)
    coords = np.concatenate(cl, 0) if cl else np.zeros((0, 3), np.int32)
    num_per_v = np.concatenate(nl, 0) if nl else np.zeros((0,), np.int32)
    out = {
        'voxels': torch.from_numpy(voxels),
        'coords': torch.from_numpy(coords).long(),
        'num_per_v': torch.from_numpy(num_per_v).long(),
        'density_gt': torch.from_numpy(np.stack([s['density_gt'] for s in batch_list], 0)),
        'density_occ': torch.from_numpy(np.stack([s['density_occ'] for s in batch_list], 0)),
        'gt_boxes': [s['gt_boxes'] for s in batch_list],
        'sample_ids': [s['sample_id'] for s in batch_list],
        'batch_size': B,
    }
    if 'hm_t' in batch_list[0] and batch_list[0]['hm_t'] is not None:
        out['hm_t'] = torch.from_numpy(np.stack([s['hm_t'] for s in batch_list], 0))
        out['reg_t'] = torch.from_numpy(np.stack([s['reg_t'] for s in batch_list], 0))
        out['iou_t'] = torch.from_numpy(np.stack([s['iou_t'] for s in batch_list], 0))
        out['pos_t'] = torch.from_numpy(np.stack([s['pos_t'] for s in batch_list], 0))
    if 'labels' in batch_list[0]:
        out['labels'] = [s['labels'] for s in batch_list]
    if 'pts' in batch_list[0]:
        out['pts_batch'] = [s['pts'] for s in batch_list]
    return out


def move_batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# §15  TRAINER
# ═══════════════════════════════════════════════════════════════════════════════

class JsonLogger:
    def __init__(self, path): self._f = open(path, 'a')
    def log(self, row, pp=''):
        if pp:
            print(pp + '  '.join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in row.items()), flush=True)
        self._f.write(json.dumps(row) + '\n'); self._f.flush()
    def close(self): self._f.close()


class WeightEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.float()
    def swap_in(self, model):
        backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.to(next(model.parameters()).device)
                               for k, v in self.shadow.items()}, strict=False)
        return backup
    def restore(self, model, backup): model.load_state_dict(backup)


def train_one_epoch(model, loader, opt, sched, scaler, dev, epoch, logger, use_amp=False):
    model.train()
    if hasattr(model, 'set_epoch'):
        model.set_epoch(epoch, getattr(model, 'total_epochs', 80))
    t0 = time.time(); tl = 0; n = 0
    for i, batch in enumerate(loader):
        batch = move_batch_to_device(batch, dev)
        with amp_autocast(use_amp):
            out = model(batch)
            loss, tb = model.head.compute_loss(
                pred={'hm': out['hm'], 'reg': out['reg'], 'dir': out['dir']},
                hm_t=batch['hm_t'], reg_t=batch['reg_t'], pos_t=batch['pos_t'],
                d_hat=out['d_hat'], d_gt=batch['density_gt'],
                d_occ=batch['density_occ'], gate_w=out['gate_w'],
                gate_log=out.get('gate_log'))
        if not torch.isfinite(loss):
            bad = {k: (float(v) if torch.is_tensor(v) and v.numel() == 1 else None)
                   for k, v in tb.items()}
            print(f'[train ep{epoch:03d} step{i+1:04d}] non-finite loss {bad} — skip step')
            continue
        opt.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        sched.step()
        tl += float(loss.detach()); n += 1
        if (i + 1) % 20 == 0:
            logger.log({'epoch': epoch, 'step': i + 1, **tb},
                       pp=f'[train ep{epoch:03d} step{i+1:04d}] ')
    print(f'[train ep{epoch:03d}] avg_loss={tl/max(n,1):.4f}  t={time.time()-t0:.0f}s')
    return tl / max(n, 1)


@torch.no_grad()
def _run_eval(model, loader, dev, soft_nms_flag=False,
              eval_score_thresh=0.10, progress_every=25, tag='val'):
    from sgf_eval_official import eval_score_thresh_context, summarize_det_lists
    model.eval(); all_preds, all_gts = [], []
    t0 = time.time()
    with eval_score_thresh_context(model, eval_score_thresh, NUM_CLASSES):
        for bi, batch in enumerate(loader):
            if progress_every and bi > 0 and bi % progress_every == 0:
                print(f'[{tag}] batch {bi}/{len(loader)} …', flush=True)
            batch = move_batch_to_device(batch, dev)
            out = model(batch)
            preds = model.head.post_process({'hm': out['hm'], 'reg': out['reg']},
                                            use_soft_nms=soft_nms_flag)
            for pd in preds:
                all_preds.append({k: v.cpu() for k, v in pd.items()})
            for lbs in batch.get('labels', [[] for _ in range(batch['batch_size'])]):
                all_gts.append(lbs)
    stats = summarize_det_lists(all_preds, all_gts)
    print(f"[{tag}] inference {time.time()-t0:.0f}s  frames={stats['frames']}  "
          f"gt={stats['n_gt']}  pred={stats['n_pred']}  "
          f"eval_score_thresh={eval_score_thresh}", flush=True)
    return all_preds, all_gts


def train(args):
    dev = resolve_torch_device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = getattr(args, 'cache_dir', None) or _default_npz_cache()
    require_cache = getattr(args, 'require_cache', False)
    frame_stride = getattr(args, 'frame_stride', 5)
    epochs = args.epochs
    do_inline_val = 0 < args.val_interval <= epochs

    gt_sampler = _build_gt_sampler(args)
    train_ds = WaymoAGDPEDataset(args.waymo_root, args.train_split, args.split,
                                  True, True, max_voxels=MAX_VOXELS_TRAIN,
                                  cache_dir=cache_dir, frame_stride=frame_stride,
                                  require_cache=require_cache, return_pts=False,
                                  gt_sampler=gt_sampler)
    val_ds = val_loader = None
    if do_inline_val:
        val_ds = WaymoAGDPEDataset(args.waymo_root, args.val_split, args.val_split_name,
                                    False, keep_labels_for_eval=True,
                                    max_voxels=MAX_VOXELS_EVAL, cache_dir=cache_dir,
                                    frame_stride=1, require_cache=require_cache,
                                    return_pts=False)
    nw = args.num_workers
    val_nw = getattr(args, 'val_num_workers', None)
    if val_nw is None:
        val_nw = max(2, nw // 2) if nw > 0 else 4
    val_bs = getattr(args, 'val_batch_size', None) or args.batch_size
    dl_kw = dict(collate_fn=collate_waymo, pin_memory=(nw > 0))
    if nw > 0:
        dl_kw.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, drop_last=True, **dl_kw)
    if getattr(args, 'cls_balanced', False):
        sampler = _waymo_cls_balanced_sampler(train_ds, cache_dir)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=nw, drop_last=True, **dl_kw)
    if do_inline_val:
        val_dl_kw = dict(collate_fn=collate_waymo, pin_memory=True)
        if val_nw > 0:
            val_dl_kw.update(persistent_workers=True, prefetch_factor=2)
        val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False,
                                num_workers=val_nw, drop_last=False, **val_dl_kw)
    else:
        print('[train] inline val disabled — official eval via submit_eval.sh', flush=True)

    st = getattr(args, 'score_thresh', 0.01)
    model = AGDPEModel(score_thresh=_per_class_thresh(st),
                       total_epochs=epochs).to(dev)
    p = model.count_params()
    print(f"[train] AG-DPE Waymo  params total={p['total']:,}  score_thresh={st}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=getattr(args, 'weight_decay', 1e-4))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=len(train_loader) * epochs,
        pct_start=getattr(args, 'pct_start', 0.2), div_factor=getattr(args, 'div_factor', 10.),
        final_div_factor=getattr(args, 'final_div_factor', 100.))
    scaler = amp_grad_scaler(args.amp)
    ema = WeightEMA(model, args.ema_decay) if args.ema else None

    tlog = JsonLogger(str(out_dir / 'train_log.jsonl'))
    vlog = JsonLogger(str(out_dir / 'val_log.jsonl'))
    best = -1.; start = 1

    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt['model'])
        start = ckpt.get('epoch', 0) + 1; best = ckpt.get('best_metric', -1.)
        print(f"[train] resumed ep={start-1}  best={best:.4f}")

    print(f"[train] {len(train_ds)} train"
          f"{f', {len(val_ds)} val' if val_ds is not None else ''} — "
          f"{epochs} ep, batch {args.batch_size}, lr {args.lr}")

    for ep in range(start, epochs + 1):
        train_one_epoch(model, train_loader, opt, sched, scaler, dev, ep, tlog,
                        use_amp=args.amp)
        if ema: ema.update(model)
        torch.save({'model': model.state_dict(), 'epoch': ep, 'best_metric': best,
                    'arch': 'agdpe_waymo'},
                   out_dir / 'last.pt')
        if do_inline_val and ep % args.val_interval == 0:
            eval_st = getattr(args, 'eval_score_thresh', EVAL_SCORE_THRESH_DEFAULT)
            print(f'[val ep{ep:03d}] official Waymo eval (eval_score_thresh={eval_st})…', flush=True)
            if ema: bk = ema.swap_in(model)
            ap, ag = _run_eval(model, val_loader, dev, eval_score_thresh=eval_st,
                               tag=f'val ep{ep:03d}')
            if ema: ema.restore(model, bk)
            from sgf_eval_official import run_waymo_official_metrics
            wm = run_waymo_official_metrics(ap, ag, rotated_3d_iou, '3d', print_table=True)
            mAP = wm['mAP_LEVEL_1']
            l1 = wm['AP3D_LEVEL_1']
            print(f'[val ep{ep:03d}] official Waymo mAP@LEVEL_1 (3D) = {mAP:.2f}')
            row = {'epoch': ep, 'mAP_LEVEL_1': mAP, 'mAP_LEVEL_2': wm['mAP_LEVEL_2'],
                   **{f'AP3D_{c}_L1': l1.get(c, 0.) for c in CLASS_NAMES}}
            vlog.log(row, pp=f'[val ep{ep:03d}] ')
            if mAP > best:
                best = mAP
                if ema: bk2 = ema.swap_in(model)
                torch.save({'model': model.state_dict(), 'epoch': ep,
                            'best_metric': best, 'ema_used': ema is not None,
                            'arch': 'agdpe_waymo'},
                           out_dir / 'best.pt')
                if ema: ema.restore(model, bk2)
                (out_dir / 'best.meta.json').write_text(json.dumps(
                    {'arch': 'agdpe_waymo', 'epoch': ep,
                     'best_metric': float(best)}))
                print(f"[val ep{ep:03d}] NEW BEST mAP@LEVEL_1={best:.4f}")

    if not (out_dir / 'best.pt').is_file():
        if ema:
            bk = ema.swap_in(model)
        torch.save({'model': model.state_dict(), 'epoch': epochs,
                    'best_metric': best, 'ema_used': ema is not None,
                    'arch': 'agdpe_waymo'}, out_dir / 'best.pt')
        if ema:
            ema.restore(model, bk)
        (out_dir / 'best.meta.json').write_text(json.dumps(
            {'arch': 'agdpe_waymo', 'epoch': epochs, 'best_metric': float(best)}))
        print('[train] saved best.pt from final epoch (train-only / OpenPCDet protocol)', flush=True)

    tlog.close(); vlog.close()
    print(f"[train] DONE  best={best:.4f}  out={out_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# §16  EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(args):
    dev = resolve_torch_device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = getattr(args, 'cache_dir', None) or _default_npz_cache()
    require_cache = getattr(args, 'require_cache', False)
    ds = WaymoAGDPEDataset(args.waymo_root, args.val_split, args.val_split_name,
                            False, keep_labels_for_eval=True,
                            max_voxels=MAX_VOXELS_EVAL, cache_dir=cache_dir,
                            require_cache=require_cache, return_pts=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_waymo,
                        pin_memory=True)
    st = getattr(args, 'score_thresh', 0.01)
    eval_st = getattr(args, 'eval_score_thresh', EVAL_SCORE_THRESH_DEFAULT)
    model = AGDPEModel(score_thresh=_per_class_thresh(st)).to(dev)
    ckpt = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt['model']); model.eval()
    print(f"[eval] {args.ckpt}  arch={ckpt.get('arch','?')}  "
          f"score_thresh={st}  eval_score_thresh={eval_st}")
    ap, ag = _run_eval(model, loader, dev, eval_score_thresh=eval_st, tag='eval')
    from sgf_eval_official import (run_waymo_official_metrics, evaluate_waymo_official,
                                   print_waymo_official_table, format_waymo_eval_table)
    wm = run_waymo_official_metrics(ap, ag, rotated_3d_iou, '3d', print_table=True)
    l1_3d, l2_3d = wm['AP3D_LEVEL_1'], wm['AP3D_LEVEL_2']
    l1_aph, l2_aph = wm['APH3D_LEVEL_1'], wm['APH3D_LEVEL_2']
    mAP_l1, mAP_l2 = wm['mAP_LEVEL_1'], wm['mAP_LEVEL_2']
    mAPH_l1, mAPH_l2 = wm['mAPH_LEVEL_1'], wm['mAPH_LEVEL_2']
    l1_bev = evaluate_waymo_official(ap, ag, 'LEVEL_1', rotated_bev_iou, 'bev')
    l2_bev = evaluate_waymo_official(ap, ag, 'LEVEL_2', rotated_bev_iou, 'bev')
    print_waymo_official_table(l1_bev, l2_bev, 'bev')
    diag_3d = evaluate_waymo(ap, ag, '3d')
    print("\n[eval] diagnostic AP3D (KITTI-style E/M/H):")
    print_ap_table(diag_3d)
    (out_dir / 'eval_results.json').write_text(json.dumps({
        'official_AP3D_LEVEL_1': l1_3d, 'official_AP3D_LEVEL_2': l2_3d,
        'official_APH3D_LEVEL_1': l1_aph, 'official_APH3D_LEVEL_2': l2_aph,
        'official_mAP_LEVEL_1': mAP_l1, 'official_mAP_LEVEL_2': mAP_l2,
        'official_mAPH_LEVEL_1': mAPH_l1, 'official_mAPH_LEVEL_2': mAPH_l2,
        'diagnostic_AP3D': diag_3d,
    }, indent=2))
    (out_dir / 'eval_table.txt').write_text(
        format_waymo_eval_table(l1_3d, l2_3d, l1_aph, l2_aph, '3d'))
    print(f"[eval] Waymo mAP L1={mAP_l1:.2f}  mAPH L1={mAPH_l1:.2f}  → {out_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# §17  VISUALIZER
# ═══════════════════════════════════════════════════════════════════════════════

def visualize(args):
    dev = resolve_torch_device(args.device)
    viz_sub = getattr(args, 'viz_subdir', 'viz_agdpe')
    out_dir = Path(args.out_dir) / viz_sub
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = getattr(args, 'cache_dir', None) or _default_npz_cache()
    require_cache = getattr(args, 'require_cache', False)
    ds = WaymoAGDPEDataset(args.waymo_root, args.val_split, args.val_split_name,
                            False, keep_labels_for_eval=True,
                            max_voxels=MAX_VOXELS_EVAL, cache_dir=cache_dir,
                            require_cache=require_cache, return_pts=True)
    st = getattr(args, 'score_thresh', 0.01)
    model = AGDPEModel(score_thresh=_per_class_thresh(st)).to(dev)
    ckpt = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt['model']); model.eval()
    n = min(args.num_viz, len(ds)); ext = VIZ_IMAGE_EXT
    print(f"[viz] {n} samples → {out_dir}")
    for i in range(n):
        sample = ds[i]
        batch = move_batch_to_device(collate_waymo([sample]), dev)
        out = model(batch)
        pd = model.head.post_process({'hm': out['hm'], 'reg': out['reg'],
                                       'iou': out['iou']})[0]
        pb = pd['pred_boxes'].cpu().numpy(); ps = pd['pred_scores'].cpu().numpy()
        pl = pd['pred_labels'].cpu().numpy()
        gt_boxes = sample['gt_boxes']; pts = sample['pts']; sid = sample['sample_id']
        viz_bev_detection(pts, gt_boxes, pb, ps, pl,
            save_path=str(out_dir / f'{sid}_bev{ext}'),
            title=f'Waymo AG-DPE {sid}')
        d_hat_np = out['d_hat'].cpu().numpy()
        gate_w_np = out['gate_w'].cpu().numpy()
        viz_density_gates(d_hat_np, gate_w_np,
            save_path=str(out_dir / f'{sid}_density_gates{ext}'),
            title=f'Waymo AG-DPE {sid} — density + gates')
        print(f"[viz] {sid}: GT={len(gt_boxes)}  pred={len(pb)}")
    print(f"[viz] DONE → {out_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# §18  SMOKE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def smoke_test(args):
    """GPU sanity check with random points — no dataset required."""
    dev = resolve_torch_device(args.device)
    print(f"[smoke] device={dev}")
    rng = np.random.default_rng(0); N = 20000
    pts = np.stack([rng.uniform(PC_RANGE[i], PC_RANGE[i + 3], N).astype(np.float32)
                    for i in range(3)] + [rng.uniform(0, 1, N).astype(np.float32)], 1)
    gt = np.array([[5., 2., -0.5, 4.73, 1.99, 1.77, .3, 1.]], np.float32)
    voxels, coords, num_per_v = voxelize_pillars(pts, max_voxels=2000)
    density_gt, density_occ = make_density_gt_from_pts(pts)
    batch = {
        'voxels': torch.from_numpy(voxels).to(dev),
        'coords': torch.from_numpy(coords).long().to(dev),
        'num_per_v': torch.from_numpy(num_per_v).long().to(dev),
        'gt_boxes': [gt], 'batch_size': 1,
    }
    hm_t, reg_t, iou_t, pos_t = build_targets(gt)
    batch.update({
        'hm_t': torch.from_numpy(hm_t).unsqueeze(0).to(dev),
        'reg_t': torch.from_numpy(reg_t).unsqueeze(0).to(dev),
        'iou_t': torch.from_numpy(iou_t).unsqueeze(0).to(dev),
        'pos_t': torch.from_numpy(pos_t).unsqueeze(0).to(dev),
        'density_gt': torch.from_numpy(density_gt).unsqueeze(0).to(dev),
        'density_occ': torch.from_numpy(density_occ).unsqueeze(0).to(dev),
    })
    model = AGDPEModel().to(dev); p = model.count_params()
    print(f"[smoke] params: total={p['total']:,} vfe={p['vfe']:,} bb={p['backbone']:,} head={p['head']:,}")
    out = model(batch)
    loss, tb = model.head.compute_loss(
        pred={'hm': out['hm'], 'reg': out['reg'], 'dir': out['dir']},
        hm_t=batch['hm_t'], reg_t=batch['reg_t'], pos_t=batch['pos_t'],
        d_hat=out['d_hat'], d_gt=batch['density_gt'],
        d_occ=batch['density_occ'], gate_w=out['gate_w'],
        gate_log=out.get('gate_log'))
    print(f"[smoke] loss={float(loss):.4f}  hm={tb['loss_hm']:.4f}  "
          f"reg={tb['loss_reg']:.4f}  den={tb['loss_density']:.4f}  "
          f"gate={tb['loss_gate']:.4f}")
    print(f"[smoke] hm={tuple(out['hm'].shape)}  reg={tuple(out['reg'].shape)}")
    print(f"[smoke] F_NX={F_NX} F_NY={F_NY}  FINE_VS={FINE_VS}")
    print("[smoke] PASS ✓")


def train_smoke(args):
    """Run 3 real training steps with data — preflight before long Slurm job."""
    dev = resolve_torch_device(args.device)
    use_amp = bool(getattr(args, 'amp', False))
    cache_dir = getattr(args, 'cache_dir', None) or _default_npz_cache()
    train_ds = WaymoAGDPEDataset(
        args.waymo_root, args.train_split, args.split, True, True,
        max_voxels=MAX_VOXELS_TRAIN, cache_dir=cache_dir,
        require_cache=getattr(args, 'require_cache', False),
        frame_stride=getattr(args, 'frame_stride', 5))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate_waymo, drop_last=True)
    print(f"[train_smoke] dataset ready  batches={len(loader)}", flush=True)
    model = AGDPEModel(total_epochs=80).to(dev)
    print(f"[train_smoke] model on {dev}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=getattr(args, 'weight_decay', 1e-4))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(3, len(loader)),
        pct_start=.04, div_factor=10, final_div_factor=100)
    scaler = amp_grad_scaler(use_amp)
    n_steps = int(getattr(args, 'train_smoke_steps', 3))
    print(f"[train_smoke] device={dev}  amp={use_amp}  steps={n_steps}", flush=True)
    ok = 0
    for step, batch in enumerate(loader):
        if step >= n_steps:
            break
        print(f"[train_smoke] step {step+1}/{n_steps}: forward…", flush=True)
        batch = move_batch_to_device(batch, dev)
        with amp_autocast(use_amp):
            out = model(batch)
            loss, tb = model.head.compute_loss(
                pred={'hm': out['hm'], 'reg': out['reg'], 'dir': out['dir']},
                hm_t=batch['hm_t'], reg_t=batch['reg_t'], pos_t=batch['pos_t'],
                d_hat=out['d_hat'], d_gt=batch['density_gt'],
                d_occ=batch['density_occ'], gate_w=out['gate_w'],
                gate_log=out.get('gate_log'))
        if not torch.isfinite(loss):
            print(f"[train_smoke] FAIL step {step+1}: non-finite loss {tb}")
            import sys; sys.exit(1)
        opt.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        sched.step(); ok += 1
        print(f"[train_smoke] step {step+1}: loss={float(loss):.4f}  "
              f"hm={tb['loss_hm']:.4f}  den={tb['loss_density']:.4f}", flush=True)
    print(f"[train_smoke] PASS ({ok} steps)", flush=True)
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# §19  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description='AG-DPE Waymo Open Dataset')
    p.add_argument('--mode', choices=['smoke', 'train_smoke', 'train', 'eval', 'viz'],
                   required=True)
    p.add_argument('--waymo_root', type=str, default=None)
    p.add_argument('--split', type=str, default='training')
    p.add_argument('--val_split_name', type=str, default='validation')
    p.add_argument('--train_split', type=str, default=None)
    p.add_argument('--val_split', type=str, default=None)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--val_batch_size', type=int, default=None,
                   help='Val/eval batch size (default: same as --batch_size)')
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--frame_stride', type=int, default=5)
    p.add_argument('--cache_dir', type=str, default=None)
    p.add_argument('--require_cache', action='store_true')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--val_interval', type=int, default=5)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--ema', action='store_true')
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--score_thresh', type=float, default=0.01)
    p.add_argument('--eval_score_thresh', type=float, default=EVAL_SCORE_THRESH_DEFAULT,
                   help='Low threshold for val/eval AP (full PR curve)')
    p.add_argument('--num_viz', type=int, default=12)
    p.add_argument('--viz_subdir', type=str, default='viz_agdpe')
    p.add_argument('--out_dir', type=str,
                   default='/home/frahman8/scratch/SGF_Port/waymo/Runs/waymo_agdpe')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if str(args.device).lower().startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed)

    if args.mode == 'smoke':
        smoke_test(args)
    elif args.mode == 'train_smoke':
        if not (args.waymo_root and args.train_split):
            print("ERROR: --waymo_root --train_split required", file=sys.stderr)
            sys.exit(2)
        train_smoke(args)
    elif args.mode == 'train':
        if not (args.waymo_root and args.train_split and args.val_split):
            print("ERROR: --waymo_root --train_split --val_split required", file=sys.stderr)
            sys.exit(2)
        train(args)
    elif args.mode == 'eval':
        if not (args.waymo_root and args.val_split and args.ckpt):
            print("ERROR: --waymo_root --val_split --ckpt required", file=sys.stderr)
            sys.exit(2)
        evaluate(args)
    elif args.mode == 'viz':
        if not (args.waymo_root and args.val_split and args.ckpt):
            print("ERROR: --waymo_root --val_split --ckpt required", file=sys.stderr)
            sys.exit(2)
        visualize(args)


if __name__ == '__main__':
    main()
