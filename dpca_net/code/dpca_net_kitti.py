"""
SGF-DualGrid (BCAF: Bidirectional Cross-Stream Attention + density fix + height fix) — KITTI 3D detection.

This file is the user's best-working SGF-DualGrid runner with ONE change:
§7 uses the original single-gate neck with the DENSITY branch removed (pure
semantic attention gate G=σ(W_s(f_mid))); §8/§9 carry the height fix
(metric anchor-relative encoding + D1/D2/D3 head corrections). Everything else (parsing, augmentation, voxelization, VFE/scatter,
head, NMS, evaluator, viz, dataset, trainer, evaluator, viz, smoke, CLI) is
unchanged so that any metric delta is attributable to the fusion alone.

No OpenPCDet dependency. Single file. Trainable / evaluable / visualizable.

────────────────────────────────────────────────────────────────────────────────
  §1   Config constants
  §2   KITTI calib / label / point parsing (keeps difficulty metadata)
  §3   Data augmentation (random flip, rotation, scaling) + GT-sampler
  §4   Pillar voxelization
  §5   Coarse BEV + density-map precomputation
  §6   PillarVFE + PillarScatter
  §7   Backbone modules: FineBEV, CoarseBEV, WindowAttn, SGFFPNNeck (semantic gate)
  §8   Gaussian heatmap targets
  §9   SGFCenterHead
  §10  SGFDualGridModel + DCGR two-stage
  §11  Rotated BEV IoU + rotated NMS
  §12  KITTI AP evaluator (40-point)
  §13  Visualization
  §14  Dataset + collate
  §15  Trainer
  §16  Evaluator
  §17  Visualizer + test export
  §18  Smoke test
  §19  CLI
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# Forward/backward-compatible AMP helpers (handles PyTorch 1.10 <-> 2.x)
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
# §1  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

FINE_VS    = 0.16
COARSE_VS  = 1.28
PC_RANGE   = [0.0, -40.0, -3.0, 70.4, 40.0, 1.0]
MAX_PTS    = 32
NUM_CLASSES = 3
CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']
CLASS_TO_ID = {n: i + 1 for i, n in enumerate(CLASS_NAMES)}

F_NX = int((PC_RANGE[3] - PC_RANGE[0]) / FINE_VS)       # 440
F_NY = int((PC_RANGE[4] - PC_RANGE[1]) / FINE_VS)       # 500
C_NX = int((PC_RANGE[3] - PC_RANGE[0]) / COARSE_VS)
C_NY = int((PC_RANGE[4] - PC_RANGE[1]) / COARSE_VS)

MAX_VOXELS_TRAIN = 16000
MAX_VOXELS_EVAL  = 40000

KITTI_DIFFICULTY_LIMITS = {
    'Easy':     (0.15, 0, 40),
    'Moderate': (0.30, 1, 25),
    'Hard':     (0.50, 2, 25),
}

IOU_THRESH_3D  = {'Car': 0.7,  'Pedestrian': 0.5,  'Cyclist': 0.5}
IOU_THRESH_BEV = {'Car': 0.7,  'Pedestrian': 0.5,  'Cyclist': 0.5}

FOCAL_ALPHA = [1.0, 3.0, 3.0]


# ═══════════════════════════════════════════════════════════════════════════════
# §2  KITTI PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_calib(path: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, vals = line.split(':', 1)
            out[key.strip()] = np.array([float(v) for v in vals.split()], dtype=np.float64)
    T = np.eye(4, dtype=np.float64); T[:3, :4] = out['Tr_velo_to_cam'].reshape(3, 4)
    R = np.eye(4, dtype=np.float64); R[:3, :3] = out['R0_rect'].reshape(3, 3)
    V2C = R @ T
    C2V = np.linalg.inv(V2C)
    P2 = np.zeros((3, 4), dtype=np.float64)
    if 'P2' in out:
        P2[:, :] = out['P2'].reshape(3, 4)
    return {
        'V2C': V2C.astype(np.float32),
        'C2V': C2V.astype(np.float32),
        'P2':  P2.astype(np.float32),
    }


def parse_label(path: str, calib: Dict[str, np.ndarray]) -> List[Dict]:
    out: List[Dict] = []
    if not os.path.exists(path):
        return out
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15: continue
            cls = parts[0]
            truncation = float(parts[1])
            occlusion  = int(float(parts[2]))
            bbox_2d = np.array([float(parts[4]), float(parts[5]),
                                float(parts[6]), float(parts[7])], dtype=np.float32)
            h, w, l = float(parts[8]),  float(parts[9]),  float(parts[10])
            x_cam, y_cam, z_cam = float(parts[11]), float(parts[12]), float(parts[13])
            ry_cam = float(parts[14])

            y_cam_c = y_cam - h / 2.0
            p_cam = np.array([x_cam, y_cam_c, z_cam, 1.0], dtype=np.float32)
            p_velo = calib['C2V'] @ p_cam
            x_v, y_v, z_v = p_velo[0], p_velo[1], p_velo[2]
            ry_velo = -(ry_cam + math.pi / 2.0)
            ry_velo = (ry_velo + math.pi) % (2 * math.pi) - math.pi

            cls_id = CLASS_TO_ID.get(cls, -1)
            bbox_h = bbox_2d[3] - bbox_2d[1]

            diff_label = 'Unknown'
            for dname, (t_max, o_max, h_min) in KITTI_DIFFICULTY_LIMITS.items():
                if truncation <= t_max and occlusion <= o_max and bbox_h >= h_min:
                    diff_label = dname
                    break

            out.append({
                'cls_name':    cls,
                'cls_id':      cls_id,
                'box':         np.array([x_v, y_v, z_v, l, w, h, ry_velo,
                                          max(cls_id, 0)], dtype=np.float32),
                'truncation':  truncation,
                'occlusion':   occlusion,
                'bbox_2d':     bbox_2d,
                'difficulty':  diff_label,
            })
    return out


def velo_box_corners_xyz(box: np.ndarray) -> np.ndarray:
    x, y, z = float(box[0]), float(box[1]), float(box[2])
    l_, w_, h_ = float(box[3]), float(box[4]), float(box[5])
    ry = float(box[6])
    c, s = math.cos(ry), math.sin(ry)
    corners = np.zeros((8, 3), dtype=np.float64)
    idx = 0
    for dl in (-l_ / 2, l_ / 2):
        for dw in (-w_ / 2, w_ / 2):
            for dh in (-h_ / 2, h_ / 2):
                ox, oy, oz = dl, dw, dh
                wx = c * ox - s * oy
                wy = s * ox + c * oy
                corners[idx] = [x + wx, y + wy, z + oz]
                idx += 1
    return corners


def project_velo_corners_to_uv(corners_v: np.ndarray, calib: Dict[str, np.ndarray]) -> np.ndarray:
    V2C = calib['V2C'].astype(np.float64)
    P2 = calib['P2'].astype(np.float64)
    n = corners_v.shape[0]
    hom = np.concatenate([corners_v, np.ones((n, 1), dtype=np.float64)], axis=1)
    xyz = (hom @ V2C.T)[:, :3]
    hom2 = np.concatenate([xyz, np.ones((n, 1), dtype=np.float64)], axis=1)
    proj = hom2 @ P2.T
    d = proj[:, 2:3]
    d = np.where(np.abs(d) < 1e-6, 1e-6, d)
    return (proj[:, :2] / d).astype(np.float32)


def velo_box_to_kitti_line(box: np.ndarray, cls_name: str, score: float,
                           calib: Dict[str, np.ndarray]) -> str:
    V2C = calib['V2C'].astype(np.float64)
    p_velo = np.array([float(box[0]), float(box[1]), float(box[2]), 1.0], dtype=np.float64)
    p_cam = V2C @ p_velo
    xc, yc, zc = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    l_, w_, h_ = float(box[3]), float(box[4]), float(box[5])
    ry_velo = float(box[6])

    ry_cam = -ry_velo - math.pi / 2.0
    ry_cam = (ry_cam + math.pi) % (2 * math.pi) - math.pi

    y_bottom = yc + h_ / 2.0
    alpha = ry_cam - math.atan2(xc, zc)

    corners = velo_box_corners_xyz(box)
    uv = project_velo_corners_to_uv(corners, calib)
    u1, v1 = float(uv[:, 0].min()), float(uv[:, 1].min())
    u2, v2 = float(uv[:, 0].max()), float(uv[:, 1].max())

    truncation = 0.0
    occlusion = 0
    return (
        f"{cls_name} {truncation:.2f} {occlusion} {alpha:.6f} "
        f"{u1:.2f} {v1:.2f} {u2:.2f} {v2:.2f} "
        f"{h_:.4f} {w_:.4f} {l_:.4f} "
        f"{xc:.4f} {y_bottom:.4f} {zc:.4f} {ry_cam:.4f} {score:.6f}"
    )


def load_points_bin(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def gt_array_for_training(labels: List[Dict]) -> np.ndarray:
    out = []
    for o in labels:
        if o['cls_id'] >= 1:
            out.append(o['box'])
    if not out:
        return np.zeros((0, 8), dtype=np.float32)
    return np.stack(out, axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# §3  AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def augment(pts: np.ndarray, gt: np.ndarray,
            flip_p: float = 0.5,
            rot_range: float = math.pi / 4,
            scale_range: Tuple[float, float] = (0.95, 1.05),
            z_jitter: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    pts = pts.copy()
    gt  = gt.copy() if len(gt) else gt

    if np.random.rand() < flip_p:
        pts[:, 1] = -pts[:, 1]
        if len(gt):
            gt[:, 1] = -gt[:, 1]
            gt[:, 6] = -gt[:, 6]

    angle = float(np.random.uniform(-rot_range, rot_range))
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    pts[:, :2] = pts[:, :2] @ R.T
    if len(gt):
        gt[:, :2] = gt[:, :2] @ R.T
        gt[:, 6] += angle

    scale = float(np.random.uniform(*scale_range))
    pts[:, :3] *= scale
    if len(gt):
        gt[:, :3] *= scale
        gt[:, 3:6] *= scale

    dz = float(np.random.uniform(-z_jitter, z_jitter))
    pts[:, 2] += dz
    if len(gt):
        gt[:, 2] += dz

    return pts, gt


def points_in_rotated_box(pts: np.ndarray, box: np.ndarray) -> np.ndarray:
    x, y, z, l, w, h, ry = box[:7]
    dx = pts[:, 0] - x; dy = pts[:, 1] - y; dz = pts[:, 2] - z
    c, s = math.cos(-ry), math.sin(-ry)
    lx = c * dx - s * dy
    ly = s * dx + c * dy
    return (np.abs(lx) <= l / 2) & (np.abs(ly) <= w / 2) & (np.abs(dz) <= h / 2)


def gt_sample(pts: np.ndarray, gt_boxes: np.ndarray, db: Dict[str, List[Dict]],
              samples_per_class: Dict[str, int],
              perturb_xy: float = 0.5, perturb_ang: float = math.pi / 12,
              max_attempts_mult: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    pts = pts.copy()
    existing = [b.copy() for b in gt_boxes] if len(gt_boxes) else []

    for cls_name, n_target in samples_per_class.items():
        if cls_name not in CLASS_TO_ID: continue
        cls_id = CLASS_TO_ID[cls_name]
        n_have = sum(1 for b in existing if int(b[7]) == cls_id)
        n_need = max(0, n_target - n_have)
        bank = db.get(cls_name, [])
        if n_need == 0 or not bank: continue

        n_cand = min(n_need * max_attempts_mult, len(bank))
        idxs = np.random.choice(len(bank), n_cand, replace=False)

        for idx in idxs:
            current = sum(1 for b in existing if int(b[7]) == cls_id)
            if current >= n_target: break
            obj = bank[int(idx)]
            box   = obj['box'].copy()
            o_pts = obj['pts'].copy()

            dr = float(np.random.uniform(-perturb_ang, perturb_ang))
            dx = float(np.random.uniform(-perturb_xy, perturb_xy))
            dy = float(np.random.uniform(-perturb_xy, perturb_xy))
            cx, cy = float(box[0]), float(box[1])
            c, s = math.cos(dr), math.sin(dr)
            rx = c * (o_pts[:, 0] - cx) - s * (o_pts[:, 1] - cy)
            ry_ = s * (o_pts[:, 0] - cx) + c * (o_pts[:, 1] - cy)
            o_pts[:, 0] = rx + cx + dx
            o_pts[:, 1] = ry_ + cy + dy
            box_new = box.copy()
            box_new[0] += dx; box_new[1] += dy
            box_new[6] += dr

            if not (PC_RANGE[0] <= box_new[0] <= PC_RANGE[3]
                    and PC_RANGE[1] <= box_new[1] <= PC_RANGE[4]):
                continue

            collide = False
            for eb in existing:
                if rotated_bev_iou(box_new[:7], np.asarray(eb[:7], dtype=np.float64)) > 0.0:
                    collide = True; break
            if collide: continue

            inside_new = points_in_rotated_box(pts, box_new)
            pts = pts[~inside_new]
            pts = np.concatenate([pts, o_pts], axis=0)
            existing.append(box_new)

    gt_out = np.stack(existing, 0) if existing else np.zeros((0, 8), np.float32)
    return pts, gt_out


def load_gt_database(db_path: str) -> Dict[str, List[Dict]]:
    with open(db_path, 'rb') as f:
        return pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# §4  PILLAR VOXELIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def voxelize_pillars(pts: np.ndarray, max_voxels: int = MAX_VOXELS_TRAIN,
                     max_pts: int = MAX_PTS
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pc = PC_RANGE
    x, y, z, i = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    ok = ((x >= pc[0]) & (x < pc[3]) & (y >= pc[1]) & (y < pc[4]) &
          (z >= pc[2]) & (z < pc[5]))
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
        sel  = np.zeros(V, dtype=bool); sel[keep] = True
        valid_pt = sel[inv]
        pts  = pts[valid_pt]; inv = inv[valid_pt]
        old2new = -np.ones(V, dtype=np.int32)
        old2new[keep] = np.arange(len(keep), dtype=np.int32)
        inv  = old2new[inv]
        uniq = uniq[keep]; cnt = cnt[keep]
        V    = len(uniq)

    voxels    = np.zeros((V, max_pts, 4), np.float32)
    num_per_v = np.zeros((V,), np.int32)

    order = np.argsort(inv, kind='stable')
    inv_s = inv[order]; pts_s = pts[order]
    is_new = np.concatenate(([True], inv_s[1:] != inv_s[:-1]))
    csum = np.arange(1, len(inv_s) + 1, dtype=np.int32)
    reset_val = np.where(is_new, csum - 1, 0)
    reset_cum = np.maximum.accumulate(reset_val)
    offsets = csum - 1 - reset_cum

    keep_pt = offsets < max_pts
    inv_s   = inv_s[keep_pt]
    pts_s   = pts_s[keep_pt]
    offsets = offsets[keep_pt]

    voxels[inv_s, offsets] = pts_s
    np.add.at(num_per_v, inv_s, 1)

    coords = np.zeros((V, 3), np.int32)
    coords[:, 1] = uniq // F_NX
    coords[:, 2] = uniq %  F_NX
    return voxels, coords, num_per_v


# ═══════════════════════════════════════════════════════════════════════════════
# §5  COARSE BEV + DENSITY MAP
# ═══════════════════════════════════════════════════════════════════════════════

def make_coarse_bev(pts: np.ndarray) -> np.ndarray:
    pc = PC_RANGE
    x, y, z, i = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    ok = ((x >= pc[0]) & (x < pc[3]) & (y >= pc[1]) & (y < pc[4]) &
          (z >= pc[2]) & (z < pc[5]))
    x, y, z, i = x[ok], y[ok], z[ok], i[ok]
    if len(x) == 0:
        return np.zeros((4, C_NY, C_NX), np.float32)
    xi = np.floor((x - pc[0]) / COARSE_VS).astype(np.int32).clip(0, C_NX - 1)
    yi = np.floor((y - pc[1]) / COARSE_VS).astype(np.int32).clip(0, C_NY - 1)
    flt = yi * C_NX + xi
    z_n = ((z - pc[2]) / (pc[5] - pc[2])).clip(0, 1)
    nc = C_NY * C_NX
    cnt = np.bincount(flt, minlength=nc).astype(np.float32)
    s_z = np.bincount(flt, weights=z_n.astype(np.float64), minlength=nc).astype(np.float32)
    s_i = np.bincount(flt, weights=i.astype(np.float64),   minlength=nc).astype(np.float32)
    mx = np.zeros(nc, np.float32); np.maximum.at(mx, flt, z_n.astype(np.float32))
    v = cnt > 0
    mz = np.where(v, s_z / cnt.clip(1), 0.0)
    mi = np.where(v, s_i / cnt.clip(1), 0.0).clip(0, 1)
    ld = np.clip(np.log1p(cnt) / math.log1p(MAX_PTS), 0.0, 1.0)
    return np.stack([mz.reshape(C_NY, C_NX), mx.reshape(C_NY, C_NX),
                     ld.reshape(C_NY, C_NX), mi.reshape(C_NY, C_NX)],
                    axis=0).astype(np.float32)


def make_density_map(pts: np.ndarray) -> np.ndarray:
    pc = PC_RANGE
    x, y = pts[:, 0], pts[:, 1]
    ok = (x >= pc[0]) & (x < pc[3]) & (y >= pc[1]) & (y < pc[4])
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return np.zeros((1, F_NY, F_NX), np.float32)
    xi = np.floor((x - pc[0]) / FINE_VS).astype(np.int32).clip(0, F_NX - 1)
    yi = np.floor((y - pc[1]) / FINE_VS).astype(np.int32).clip(0, F_NY - 1)
    cnt = np.bincount(yi * F_NX + xi, minlength=F_NY * F_NX).astype(np.float32)
    ld = np.clip(np.log1p(cnt) / math.log1p(MAX_PTS), 0.0, 1.0)
    return ld.reshape(1, F_NY, F_NX)


# ═══════════════════════════════════════════════════════════════════════════════
# §6  PILLAR FEATURE ENCODER (PFE) + PILLAR SCATTER
# ═══════════════════════════════════════════════════════════════════════════════

class PillarVFE(nn.Module):
    """Dynamic Pillar Feature Encoding with max+mean dual pooling. 10-dim input."""
    def __init__(self, out_ch: int = 64):
        super().__init__()
        in_dim = 10
        half   = out_ch // 2
        self.linear_max  = nn.Linear(in_dim, half, bias=False)
        self.linear_mean = nn.Linear(in_dim, half, bias=False)
        self.bn_max  = nn.BatchNorm1d(half)
        self.bn_mean = nn.BatchNorm1d(half)
        self.out_ch = out_ch

    def forward(self, voxels: torch.Tensor, num_per_v: torch.Tensor) -> torch.Tensor:
        if voxels.numel() == 0:
            return voxels.new_zeros((0, self.out_ch))
        V, P, _ = voxels.shape
        half   = self.out_ch // 2
        mask   = torch.arange(P, device=voxels.device).unsqueeze(0) < num_per_v.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()
        denom  = num_per_v.clamp(min=1).unsqueeze(-1).float()

        centroid  = (voxels[..., :3] * mask_f).sum(dim=1) / denom
        f_center  = voxels[..., :3] - centroid.unsqueeze(1)

        x_p = (torch.floor((voxels[..., 0] - PC_RANGE[0]) / FINE_VS) * FINE_VS
               + FINE_VS * .5 + PC_RANGE[0])
        y_p = (torch.floor((voxels[..., 1] - PC_RANGE[1]) / FINE_VS) * FINE_VS
               + FINE_VS * .5 + PC_RANGE[1])
        f_pillar = torch.stack([voxels[..., 0] - x_p, voxels[..., 1] - y_p], dim=-1)

        rng = (voxels[..., 0].pow(2) + voxels[..., 1].pow(2)).sqrt().unsqueeze(-1)

        feat = torch.cat([voxels, f_center, f_pillar, rng], dim=-1) * mask_f

        fm = F.relu(self.bn_max(
            (self.linear_max(feat) * mask_f).reshape(-1, half)
        ).reshape(V, P, half) * mask_f)
        fm_out, _ = fm.max(dim=1)

        fv = F.relu(self.bn_mean(
            (self.linear_mean(feat) * mask_f).reshape(-1, half)
        ).reshape(V, P, half) * mask_f)
        fv_out = (fv * mask_f).sum(dim=1) / denom.expand(-1, half)

        return torch.cat([fm_out, fv_out], dim=-1)


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
        flat  = coords[:, 1].long() * self.nx + coords[:, 2].long()
        canvas[b_idx, :, flat] = pillar_feats
        return canvas.view(batch_size, self.ch, self.ny, self.nx)


# ═══════════════════════════════════════════════════════════════════════════════
# §7  BACKBONE / NECK  —  FUSION POINT 2: Bidirectional Cross-Stream Attention
#      Fusion (BCAF) + REDESIGNED density gate (density fix)
# ═══════════════════════════════════════════════════════════════════════════════
#
# THEORY
#   The single-gate neck pushed context one way only (coarse → fine, via a gate).
#   BCAF instead lets the two streams attend to each other:
#       fine queries coarse  ("what context am I near?")
#       coarse queries fine  ("where is the detailed evidence?")
#   through windowed cross-attention, then a density gate modulates the fused
#   result. This recovers the fine→coarse information path a one-way gate
#   discards and can model non-local "this sparse region resembles that dense
#   one" relationships a local gate cannot.
#
# DENSITY FIX (addresses the G_d-collapse diagnosed from the gate heatmaps)
#   Root causes were: BN collapsing the sparse density signal; the G_s × G_d
#   product deadlocking the density gradient; and no direct supervision. The
#   redesigned DensityGate:
#     • consumes the ALREADY log-compressed fine density map (log1p occupancy);
#     • uses a 3×3 conv for spatial context with NO BatchNorm (BN on a mostly-
#       empty map washes the signal out);
#     • negative-bias init so the gate starts selective (not the flat ≈0.5);
#     • identity-preserving residual (alpha init 1.0) so the fused feature is
#       not suppressed before the gate has learned anything;
#     • is supervised by an AUXILIARY BCE loss against an object-foreground
#       target (max over class heatmaps), giving the density gate a DIRECT
#       gradient path — so it learns object-likelihood, NOT raw occupancy
#       (which is what made the old G_d tautological).
#   Crucially there is no G_s × G_d product here, so the gradient deadlock is
#   structurally gone: the density gate's only multiplicative partner is the
#   cross-attention–fused feature, and it also has the aux-BCE path.
#
# HEIGHT FIX  lives in §8/§9 (metric anchor-relative encoding + D1/D2/D3),
# identical to the height-fixed single-gate file.
#
# Extra cost vs the single-gate neck: two windowed cross-attentions instead of
# one self-attention, plus a merge conv. Measure FLOPs before any efficiency
# claim — this is the higher-capacity variant by design.

def cbr(i, o, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(i, o, k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class ResBlock2D(nn.Module):
    """Identity-init residual block (PillarNet/CenterPoint-equivalent)."""
    def __init__(self, ch_in: int, ch_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out))
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(ch_out))
            if ch_in != ch_out or stride != 1
            else nn.Identity())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.conv1(x)) + self.shortcut(x))


class FineBEVBackbone(nn.Module):
    """3-stage deep residual BEV backbone → P1(64), P2(128,/2), P3(256,/4)."""
    def __init__(self, in_ch: int = 64, base: int = 64):
        super().__init__()
        self.b1 = nn.Sequential(
            ResBlock2D(in_ch, base),
            ResBlock2D(base,  base),
            ResBlock2D(base,  base))
        self.b2 = nn.Sequential(
            ResBlock2D(base,    base*2, stride=2),
            ResBlock2D(base*2,  base*2),
            ResBlock2D(base*2,  base*2),
            ResBlock2D(base*2,  base*2))
        self.b3 = nn.Sequential(
            ResBlock2D(base*2,  base*4, stride=2),
            ResBlock2D(base*4,  base*4),
            ResBlock2D(base*4,  base*4),
            ResBlock2D(base*4,  base*4))

    def forward(self, x: torch.Tensor):
        p1 = self.b1(x)
        p2 = self.b2(p1)
        p3 = self.b3(p2)
        return p1, p2, p3


class CoarseBEVBackbone(nn.Module):
    def __init__(self, in_ch=4, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            cbr(in_ch, 32), cbr(32, out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True))
    def forward(self, x): return self.net(x)


class CrossWindowAttn(nn.Module):
    """
    Windowed cross-attention: the query stream attends to the key/value stream
    within local windows. q_map and kv_map must share spatial size.
    """
    def __init__(self, ch, heads=4, ws=8):
        super().__init__()
        self.ws = ws
        self.norm_q  = nn.LayerNorm(ch)
        self.norm_kv = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)
        self.proj = nn.Linear(ch, ch)

    def forward(self, q_map, kv_map):
        B, C, H, W = q_map.shape
        ws = min(self.ws, H, W)
        ph = (ws - H % ws) % ws; pw = (ws - W % ws) % ws
        if ph or pw:
            q_map  = F.pad(q_map,  (0, pw, 0, ph))
            kv_map = F.pad(kv_map, (0, pw, 0, ph))
        _, _, H2, W2 = q_map.shape
        nwh, nww = H2 // ws, W2 // ws

        def to_win(x):
            return (x.reshape(B, C, nwh, ws, nww, ws)
                     .permute(0, 2, 4, 3, 5, 1)
                     .reshape(B * nwh * nww, ws * ws, C))

        qw = to_win(q_map); kw = to_win(kv_map)
        ao, _ = self.attn(self.norm_q(qw), self.norm_kv(kw), self.norm_kv(kw))
        qw = qw + self.proj(ao)
        out = (qw.reshape(B, nwh, nww, ws, ws, C)
                 .permute(0, 5, 1, 3, 2, 4)
                 .reshape(B, C, H2, W2))
        return out[:, :, :H, :W]


class DensityGate(nn.Module):
    """
    Redesigned density gate (the 'density fix').

      D       : already-log-compressed fine density map, pooled to feat size
      z_d     = Conv3x3(D)         (NO BatchNorm; bias=True, negative init)
      G_d     = sigmoid(z_d)
      out     = (G_d + alpha) * feat      (alpha init 1.0 → identity at start)
      dg_logit= mean_c(z_d)        (1-channel logits for the auxiliary BCE loss)

    The aux BCE (assembled in the head) supervises dg_logit toward an object-
    foreground target, so the gate learns object-likelihood rather than raw
    occupancy — breaking the tautology that collapsed the previous G_d.
    """
    def __init__(self, ch: int, init_bias: float = -2.0):
        super().__init__()
        self.conv = nn.Conv2d(1, ch, 3, padding=1, bias=True)   # no BN
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='linear')
        nn.init.constant_(self.conv.bias, init_bias)            # selective start
        self.alpha = nn.Parameter(torch.ones(1))                # identity start
        self.ablate_density_gate = False                      # eval: G_d ≡ 1

    def forward(self, feat: torch.Tensor, density: torch.Tensor):
        D = F.adaptive_avg_pool2d(density, feat.shape[-2:])     # log-compressed in
        z_d = self.conv(D)                                      # (B, C, H, W)
        G_d = torch.sigmoid(z_d)
        if self.ablate_density_gate:
            G_d = torch.ones_like(G_d)
        out = (G_d + self.alpha) * feat
        dg_logit = z_d.mean(dim=1, keepdim=True)                # (B, 1, H, W)
        return out, G_d, dg_logit


class BCAFFPNNeck(nn.Module):
    """
    Bidirectional Cross-Stream Attention Fusion neck (FUSION POINT 2).

      project fine P1/P2/P3 and coarse → neck_ch
      f_fine   = merge_fine([P2, P3↗P2])            (fine context at P2 res)
      f_coarse = coarse↗P2
      f_fine'  = f_fine   + xattn_f2c(q=f_fine,   kv=f_coarse)
      f_coarse'= f_coarse + xattn_c2f(q=f_coarse, kv=f_fine)
      f_fused  = merge_x([f_fine', f_coarse'])
      f_gated, G_d, dg_logit = DensityGate(f_fused, density)     (density fix)
      F_ctx    = f_gated ↗ P1
      F_cat    = [LReLU(F_ctx⊙P1), LReLU(P1), LReLU(F_ctx)]       (SiGF injection)
      out      = out_norm(fuse(F_cat))

    dg_logit is stashed for the auxiliary BCE; gate_maps exposes G_d for viz.
    """
    def __init__(self, fine_chs=(64, 128, 256), coarse_ch=64,
                 neck_ch=128, out_ch=256, heads=4, ws=8,
                 dg_init_bias: float = -2.0, save_gates: bool = False):
        super().__init__()
        self.p1p = nn.Conv2d(fine_chs[0], neck_ch, 1, bias=False)
        self.p2p = nn.Conv2d(fine_chs[1], neck_ch, 1, bias=False)
        self.p3p = nn.Conv2d(fine_chs[2], neck_ch, 1, bias=False)
        self.c1p = nn.Conv2d(coarse_ch,   neck_ch, 1, bias=False)

        self.coarse_enrich = nn.Sequential(
            nn.Conv2d(neck_ch, neck_ch, 3, padding=1, groups=neck_ch, bias=False),
            nn.BatchNorm2d(neck_ch), nn.ReLU(True),
            nn.Conv2d(neck_ch, neck_ch, 1, bias=False),
            nn.BatchNorm2d(neck_ch), nn.ReLU(True))
        self.merge_fine = nn.Sequential(
            nn.Conv2d(neck_ch * 2, neck_ch, 1, bias=False),
            nn.BatchNorm2d(neck_ch), nn.ReLU(True))

        # bidirectional windowed cross-attention
        self.xattn_f2c = CrossWindowAttn(neck_ch, heads, ws)   # fine ← coarse
        self.xattn_c2f = CrossWindowAttn(neck_ch, heads, ws)   # coarse ← fine
        self.merge_x = nn.Sequential(
            nn.Conv2d(neck_ch * 2, neck_ch, 1, bias=False),
            nn.BatchNorm2d(neck_ch), nn.ReLU(True))

        # density fix
        self.dgate = DensityGate(neck_ch, init_bias=dg_init_bias)

        self.fuse = nn.Sequential(
            nn.Conv2d(neck_ch * 3, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.1, True))
        self.out_norm = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.1, True))

        self.save_gates = save_gates
        self.gate_log: Dict[str, float] = {}
        self.gate_maps: Dict[str, torch.Tensor] = {}
        self.dg_logit: Optional[torch.Tensor] = None   # surfaced for aux BCE

    def forward(self, p1, p2, p3, c1, density):
        p1r = self.p1p(p1); p2r = self.p2p(p2); p3r = self.p3p(p3)
        c1  = self.coarse_enrich(self.c1p(c1))

        # fine context at P2 resolution
        f_fine = self.merge_fine(torch.cat(
            [p2r, F.interpolate(p3r, size=p2r.shape[-2:], mode='nearest')], dim=1))
        # coarse context resampled to P2 resolution
        f_coarse = (F.interpolate(c1, size=p2r.shape[-2:], mode='nearest')
                    if c1.shape[-2:] != p2r.shape[-2:] else c1)

        # bidirectional cross-attention
        f_fine2   = f_fine   + self.xattn_f2c(f_fine,   f_coarse)
        f_coarse2 = f_coarse + self.xattn_c2f(f_coarse, f_fine)
        f_fused   = self.merge_x(torch.cat([f_fine2, f_coarse2], dim=1))

        # density-conditioned gating (fixed gate)
        f_gated, G_d, dg_logit = self.dgate(f_fused, density)
        self.dg_logit = dg_logit

        # SiGF-style injection into raw P1 geometry
        F_ctx = F.interpolate(f_gated, size=p1r.shape[-2:], mode='nearest')
        F_cat = torch.cat([
            F.leaky_relu(F_ctx * p1r, 0.1),
            F.leaky_relu(p1r,         0.1),
            F.leaky_relu(F_ctx,       0.1)], dim=1)
        out = self.out_norm(self.fuse(F_cat))

        with torch.no_grad():
            self.gate_log = {
                'G_d_mean': float(G_d.mean()), 'G_d_std': float(G_d.std()),
                'alpha':    float(self.dgate.alpha.detach()),
            }
            if self.save_gates:
                p1_hw = p1r.shape[-2:]
                G_p1 = F.interpolate(G_d, size=p1_hw, mode='nearest').detach().cpu().numpy()
                Gm = G_p1.mean(axis=1) if G_p1.ndim == 4 else G_p1.mean(axis=0)
                den_p2 = F.adaptive_avg_pool2d(density, f_fused.shape[-2:]).detach().cpu().numpy()
                den_p1 = F.interpolate(
                    torch.as_tensor(den_p2, device=G_d.device), size=p1_hw, mode='nearest'
                ).detach().cpu().numpy()
                dg_p1 = F.interpolate(dg_logit, size=p1_hw, mode='nearest').detach().cpu().numpy()
                occ_p1 = (den_p1 > 1e-6).astype(np.float32)
                self.gate_maps = {
                    'den_p2': den_p1,
                    'G_d': G_p1,
                    'G': Gm,
                    'dg_logit': dg_p1,
                    'occ_target': occ_p1,
                }
        return out


class SGFDualGridBackbone(nn.Module):
    def __init__(self, in_ch=64, neck_ch=128, out_ch=256, heads=4, ws=8,
                 save_gates: bool = False):
        super().__init__()
        self.fine_bb   = FineBEVBackbone(in_ch, base=64)
        self.coarse_bb = CoarseBEVBackbone(in_ch=4, out_ch=64)
        self.neck      = BCAFFPNNeck(
            fine_chs=(64, 128, 256), coarse_ch=64,
            neck_ch=neck_ch, out_ch=out_ch, heads=heads, ws=ws,
            save_gates=save_gates)
        self.num_bev_features = out_ch

    def forward(self, fine_bev, coarse_bev, density):
        p1, p2, p3 = self.fine_bb(fine_bev)
        c1         = self.coarse_bb(coarse_bev)
        out        = self.neck(p1, p2, p3, c1, density)
        return out, self.neck.gate_log

# ═══════════════════════════════════════════════════════════════════════════════
# §8  HEATMAP TARGETS (Gaussian) + METRIC ANCHOR-RELATIVE BOX ENCODING
# ═══════════════════════════════════════════════════════════════════════════════
#
# HEIGHT FIX (D1/D2/D3) lives here and in §9. The regression target is now
# metric and anchor-relative so x/y/z residuals share one physical scale (D3),
# making the z/h channel weighting honest. Channel order is unchanged:
#       [dx, dy, dz, dlog_h, dlog_w, dlog_l, sin, cos]
# dx,dy : metres, residual to the cell centre
# dz    : metres, residual to the per-class z-anchor
# dlog_*: log-ratio of dimension to the per-class anchor
# sin,cos: absolute heading

# KITTI class anchors (metres). Consistent with the original head's KITTI priors.
CLASS_ANCHORS = {
    0: dict(z=-0.728, h=1.540, w=1.631, l=3.881),   # Car
    1: dict(z=-0.640, h=1.760, w=0.660, l=0.840),   # Pedestrian
    2: dict(z=-0.688, h=1.740, w=0.600, l=1.760),   # Cyclist
}


def _gaussian_radius(dh, dw, min_overlap=0.1):
    a1 = 1;    b1 = dh + dw;         c1 = dh*dw*(1-min_overlap)/(1+min_overlap)
    sq1 = math.sqrt(max(b1**2 - 4*a1*c1, 0)); r1 = (b1 - sq1) / 2
    a2 = 4;    b2 = 2*(dh + dw);     c2 = (1-min_overlap)*dh*dw
    sq2 = math.sqrt(max(b2**2 - 4*a2*c2, 0)); r2 = (b2 - sq2) / 2
    a3 = 4*min_overlap; b3 = -2*min_overlap*(dh+dw); c3 = (min_overlap-1)*dh*dw
    sq3 = math.sqrt(max(b3**2 - 4*a3*c3, 0)); r3 = (b3 + sq3) / 2
    return max(1, int(min(r1, r2, r3)))


def _draw_gaussian(hm, cx, cy, r):
    d = 2*r+1; s = d/6; m = r
    y_, x_ = np.ogrid[-m:m+1, -m:m+1]
    g = np.exp(-(x_*x_ + y_*y_) / (2*s*s)).astype(np.float32)
    g[g < np.finfo(g.dtype).eps * g.max()] = 0
    H, W = hm.shape
    l = min(cx, r); rb = min(W-cx, r+1); t = min(cy, r); b = min(H-cy, r+1)
    if min(rb-(-l), b-(-t)) > 0:
        np.maximum(hm[cy-t:cy+b, cx-l:cx+rb],
                   g[r-t:r+b,   r-l:r+rb],
                   out=hm[cy-t:cy+b, cx-l:cx+rb])


def build_targets(gt_boxes: np.ndarray, num_class: int = NUM_CLASSES,
                  H: int = F_NY, W: int = F_NX
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Metric, anchor-relative regression targets (height-fix encoding)."""
    vs = FINE_VS
    x0, y0 = PC_RANGE[0], PC_RANGE[1]
    hm  = np.zeros((num_class, H, W), np.float32)
    reg = np.zeros((8, H, W), np.float32)
    iou = np.zeros((1, H, W), np.float32)
    pos = np.zeros((1, H, W), np.float32)
    for box in gt_boxes:
        if len(box) < 8: continue
        x, y, z, l, w, h, ry, cls_id = box[:8]
        ci = int(cls_id) - 1
        if ci < 0 or ci >= num_class: continue
        cxf = (x - x0) / vs; cyf = (y - y0) / vs
        cxi, cyi = int(cxf), int(cyf)
        if not (0 <= cxi < W and 0 <= cyi < H): continue
        a = CLASS_ANCHORS[ci]
        cell_cx = (cxi + 0.5) * vs + x0
        cell_cy = (cyi + 0.5) * vs + y0
        min_ov = 0.1 if int(cls_id) == 1 else 0.25
        r = _gaussian_radius(max(1., l/vs), max(1., w/vs), min_overlap=min_ov)
        _draw_gaussian(hm[ci], cxi, cyi, r)
        reg[:, cyi, cxi] = [
            x - cell_cx,                                  # dx (m)
            y - cell_cy,                                  # dy (m)
            z - a['z'],                                   # dz (m, rel z-anchor)
            math.log(max(h, 1e-3)) - math.log(a['h']),    # dlog_h
            math.log(max(w, 1e-3)) - math.log(a['w']),    # dlog_w
            math.log(max(l, 1e-3)) - math.log(a['l']),    # dlog_l
            math.sin(ry), math.cos(ry)]
        iou[0, cyi, cxi] = 1.0
        pos[0, cyi, cxi] = 1.0
    return hm, reg, iou, pos


def decode_class_boxes(reg_ci: torch.Tensor, xi: torch.Tensor,
                       yi: torch.Tensor, ci: int) -> torch.Tensor:
    """Inverse of build_targets for a single class. reg_ci:(8,N) -> (N,7)."""
    vs = FINE_VS
    x0, y0 = PC_RANGE[0], PC_RANGE[1]
    a = CLASS_ANCHORS[ci]
    cell_cx = (xi.float() + 0.5) * vs + x0
    cell_cy = (yi.float() + 0.5) * vs + y0
    x = cell_cx + reg_ci[0]
    y = cell_cy + reg_ci[1]
    z = reg_ci[2] + a['z']
    h = (reg_ci[3] + math.log(a['h'])).exp()
    w = (reg_ci[4] + math.log(a['w'])).exp()
    l = (reg_ci[5] + math.log(a['l'])).exp()
    ry = torch.atan2(reg_ci[6], reg_ci[7])
    return torch.stack([x, y, z, l, w, h, ry], dim=1)

# ═══════════════════════════════════════════════════════════════════════════════
# §11  ROTATED BEV IoU + ROTATED NMS    (placed before §9 head)
# ═══════════════════════════════════════════════════════════════════════════════

def _polygon_clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    def inside(p, a, b):
        return (b[0]-a[0])*(p[1]-a[1]) > (b[1]-a[1])*(p[0]-a[0])
    def isect(p1, p2, a, b):
        x1, y1 = p1; x2, y2 = p2; x3, y3 = a; x4, y4 = b
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-12: return p1
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        return np.array([x1 + t*(x2-x1), y1 + t*(y2-y1)])
    out = list(subject); Nc = len(clip)
    for i in range(Nc):
        if not out: return np.zeros((0, 2))
        a = clip[i]; b = clip[(i + 1) % Nc]
        inp = out; out = []
        for j in range(len(inp)):
            curr = inp[j]; prev = inp[j - 1]
            ci_ = inside(curr, a, b); pi_ = inside(prev, a, b)
            if ci_:
                if not pi_: out.append(isect(prev, curr, a, b))
                out.append(curr)
            elif pi_:
                out.append(isect(prev, curr, a, b))
    return np.asarray(out) if out else np.zeros((0, 2))


def _poly_area(poly: np.ndarray) -> float:
    if len(poly) < 3: return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def box_corners_2d(x, y, l, w, ry) -> np.ndarray:
    c, s = math.cos(ry), math.sin(ry)
    local = np.array([[l/2, w/2], [-l/2, w/2], [-l/2, -w/2], [l/2, -w/2]], dtype=np.float64)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (R @ local.T).T + np.array([x, y])


def rotated_bev_iou(b1: np.ndarray, b2: np.ndarray) -> float:
    p1 = box_corners_2d(b1[0], b1[1], b1[3], b1[4], b1[6])
    p2 = box_corners_2d(b2[0], b2[1], b2[3], b2[4], b2[6])
    a1 = float(b1[3] * b1[4]); a2 = float(b2[3] * b2[4])
    inter = _poly_area(_polygon_clip(p1, p2))
    return inter / (a1 + a2 - inter + 1e-9)


def rotated_3d_iou(b1: np.ndarray, b2: np.ndarray) -> float:
    p1 = box_corners_2d(b1[0], b1[1], b1[3], b1[4], b1[6])
    p2 = box_corners_2d(b2[0], b2[1], b2[3], b2[4], b2[6])
    a_inter = _poly_area(_polygon_clip(p1, p2))
    z1l, z1h = b1[2] - b1[5]/2, b1[2] + b1[5]/2
    z2l, z2h = b2[2] - b2[5]/2, b2[2] + b2[5]/2
    h_inter = max(0.0, min(z1h, z2h) - max(z1l, z2l))
    inter_v = a_inter * h_inter
    v1 = b1[3] * b1[4] * b1[5]; v2 = b2[3] * b2[4] * b2[5]
    return float(inter_v / (v1 + v2 - inter_v + 1e-9))


def rotated_nms(boxes: torch.Tensor, scores: torch.Tensor,
                iou_thresh: float = 0.1, top_k: int = 500) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    dev = boxes.device
    boxes_np = boxes.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()
    order = np.argsort(-scores_np)[:top_k]
    suppressed = np.zeros(len(boxes_np), dtype=bool)
    keep: List[int] = []
    for idx in order:
        if suppressed[idx]: continue
        keep.append(int(idx))
        bi = boxes_np[idx]
        bi7 = np.concatenate([bi[:7], np.zeros(max(0, 7 - len(bi)))])
        for jdx in order:
            if suppressed[jdx] or jdx == idx: continue
            bj = boxes_np[jdx]
            bj7 = np.concatenate([bj[:7], np.zeros(max(0, 7 - len(bj)))])
            if rotated_bev_iou(bi7, bj7) > iou_thresh:
                suppressed[jdx] = True
    return torch.tensor(keep, dtype=torch.long, device=dev)


def soft_nms(boxes: torch.Tensor, scores: torch.Tensor,
             sigma: float = 0.1, score_thresh: float = 0.001,
             top_k: int = 500) -> torch.Tensor:
    boxes_np  = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy().copy()
    order     = np.argsort(-scores_np)[:top_k]
    kept: list = []
    while len(order):
        i = int(order[0]); kept.append(i); order = order[1:]
        if not len(order): break
        bi7 = boxes_np[i, :7]
        ious = np.array([rotated_bev_iou(bi7, boxes_np[j, :7]) for j in order])
        scores_np[order] *= np.exp(-(ious ** 2) / sigma)
        order = order[scores_np[order] > score_thresh]
    return torch.tensor(kept, dtype=torch.long, device=boxes.device)


def iou_rectified_score(score_cls: torch.Tensor, score_iou: torch.Tensor,
                        alpha: float = 0.68) -> torch.Tensor:
    return score_cls.pow(alpha) * score_iou.pow(1 - alpha)


# ═══════════════════════════════════════════════════════════════════════════════
# §9  HEAD  —  SGFCenterHead (height-fixed: D1 channel-axis weighting,
#              D2 per-class positive masks, D3 metric anchor-relative decode)
# ═══════════════════════════════════════════════════════════════════════════════

class SGFCenterHead(nn.Module):
    """
    CenterPoint-style head with the Car-3D height defects corrected:
      D1  channel-axis z/h weighting applied on the dense (B,8,H,W) tensor
          (the old boolean-mask flatten scrambled the per-channel weights).
      D2  per-class positive mask taken from each class's heatmap peaks, so the
          Car reg head is no longer trained on Ped/Cyc boxes.
      D3  metric anchor-relative targets (see §8) decoded via decode_class_boxes,
          so z/h gradients are commensurate with x/y; z_h_weight is now honest.
    Reg biases init to 0 (residual-to-anchor targets); cos→1 so ry≈0 at init.
    """
    def __init__(self, input_channels: int = 256, num_class: int = NUM_CLASSES,
                 loss_weights: Optional[dict] = None, z_h_weight: float = 2.0,
                 score_thresh: Sequence[float] = (0.30, 0.18, 0.18),
                 nms_iou_thresh: Sequence[float] = (0.10, 0.08, 0.08)):
        super().__init__()
        self.num_class    = num_class
        self.voxel_size   = (FINE_VS, FINE_VS, 4.0)
        self.pc_range     = tuple(PC_RANGE)
        w = loss_weights or {}
        self.w_hm  = float(w.get('hm',  1.0))
        self.w_reg = float(w.get('reg', 2.0))
        self.w_iou = float(w.get('iou', 0.5))
        # ch order [dx,dy,dz,dlog_h,dlog_w,dlog_l,sin,cos] -> z(2),h(3) boosted
        self.register_buffer('reg_ch_weights',
            torch.tensor([1., 1., z_h_weight, z_h_weight, 1., 1., 1., 1.]))
        self.score_thresh = tuple(score_thresh)
        self.nms_iou      = tuple(nms_iou_thresh)

        self.shared = nn.Sequential(cbr(input_channels, 128), cbr(128, 64))
        self.hm  = nn.Conv2d(64, num_class, 1)
        nn.init.constant_(self.hm.bias, -4.6)
        self.reg = nn.ModuleList([nn.Conv2d(64, 8, 1) for _ in range(num_class)])
        self.iou = nn.ModuleList([nn.Conv2d(64, 1, 1) for _ in range(num_class)])
        for ci in range(num_class):
            nn.init.zeros_(self.reg[ci].bias)            # residual targets -> 0
            with torch.no_grad():
                self.reg[ci].bias.data[7] = 1.0          # cos -> ry≈0 at init
            nn.init.constant_(self.iou[ci].bias, -4.6)

    def forward(self, feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = self.shared(feats)
        reg_all = torch.stack([self.reg[ci](f) for ci in range(self.num_class)], dim=1)
        iou_all = torch.stack([self.iou[ci](f) for ci in range(self.num_class)], dim=1)
        return {'hm': self.hm(f), 'reg': reg_all, 'iou': iou_all}

    @staticmethod
    def focal_loss(pred: torch.Tensor, gt: torch.Tensor,
                   gamma: float = 2.0, beta: float = 4.0) -> torch.Tensor:
        ps    = torch.clamp(torch.sigmoid(pred), 1e-6, 1 - 1e-6)
        total = pred.new_zeros(1)
        for ci, alpha in enumerate(FOCAL_ALPHA[:pred.shape[1]]):
            p = ps[:, ci]; g = gt[:, ci]
            n_pos = (g == 1).sum().float().clamp(1)
            pl = (1 - p).pow(gamma) * torch.log(p)     * (g == 1).float()
            nl = (1 - g).pow(beta)  * p.pow(gamma) * torch.log(1 - p) * (g < 1).float()
            total = total - alpha * (pl.sum() + nl.sum()) / n_pos
        return total / pred.shape[1]

    def compute_loss(self, pred, hm_t, reg_t, iou_t, pos_t,
                     gate_log: Optional[Dict[str, float]] = None,
                     aux_dg_logit: Optional[torch.Tensor] = None,
                     w_aux: float = 0.1
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_hm = self.focal_loss(pred['hm'], hm_t)
        l_reg = pred['reg'].new_zeros(1)
        l_iou = pred['iou'].new_zeros(1)
        ch_w = self.reg_ch_weights.view(1, 8, 1, 1)
        n_cls = 0
        for ci in range(self.num_class):
            # D2: per-class positives = this class's heatmap peaks (==1.0)
            cls_pos = (hm_t[:, ci] >= 1.0 - 1e-4)            # (B, H, W)
            npos = cls_pos.sum()
            if npos == 0: continue
            n_cls += 1
            reg_ci = pred['reg'][:, ci]                      # (B, 8, H, W)
            # D1: weight on the channel axis of the dense tensor, then select
            diff = F.smooth_l1_loss(reg_ci, reg_t, reduction='none')   # (B,8,H,W)
            per_cell = (diff * ch_w).sum(dim=1)              # (B, H, W)
            l_reg = l_reg + (per_cell * cls_pos).sum() / npos.clamp(1)
            iou_ci = pred['iou'][:, ci, 0]                   # (B, H, W)
            l_iou = l_iou + F.binary_cross_entropy_with_logits(
                iou_ci[cls_pos], iou_t.squeeze(1)[cls_pos])
        if n_cls > 0:
            l_reg = l_reg / n_cls
            l_iou = l_iou / n_cls

        total = self.w_hm * l_hm + self.w_reg * l_reg + self.w_iou * l_iou

        # auxiliary density-gate supervision (density fix): push the gate toward
        # object-foreground (max over class heatmaps), NOT raw occupancy.
        l_aux = pred['hm'].new_zeros(1)
        if aux_dg_logit is not None:
            with torch.no_grad():
                fg = hm_t.max(dim=1, keepdim=True).values            # (B,1,H,W) in [0,1]
                fg_t = F.adaptive_max_pool2d(fg, aux_dg_logit.shape[-2:])
            l_aux = F.binary_cross_entropy_with_logits(aux_dg_logit, fg_t)
            total = total + w_aux * l_aux

        tb = {'loss_hm':  float(l_hm.detach()),
              'loss_reg': float(l_reg.detach()),
              'loss_iou': float(l_iou.detach()),
              'loss_aux': float(l_aux.detach()),
              'loss':     float(total.detach())}
        if gate_log:
            for k, v in gate_log.items():
                tb[f'gate_{k}'] = float(v)
        return total, tb

    @torch.no_grad()
    def post_process(self, pred, use_soft_nms: bool = False,
                     iou_alpha: float = 0.68) -> List[Dict[str, torch.Tensor]]:
        hm    = torch.sigmoid(pred['hm'])
        reg   = pred['reg']
        iou   = torch.sigmoid(pred['iou'])
        hm_pool = F.max_pool2d(hm, 3, 1, 1)
        hm_peak = (hm == hm_pool).float() * hm
        B = hm.shape[0]
        out: List[Dict[str, torch.Tensor]] = []
        for bi in range(B):
            all_b, all_s, all_l = [], [], []
            for ci, (thr, nt) in enumerate(zip(self.score_thresh, self.nms_iou)):
                iou_ci = iou[bi, ci, 0]
                sm = iou_rectified_score(hm_peak[bi, ci], iou_ci.pow(0.32), alpha=iou_alpha)
                pos = (sm > thr).nonzero(as_tuple=False)
                if not len(pos): continue
                sc = sm[pos[:, 0], pos[:, 1]]
                if len(sc) > 500:
                    idx = sc.topk(500).indices; pos = pos[idx]; sc = sc[idx]
                r  = reg[bi, ci, :, pos[:, 0], pos[:, 1]]       # (8, N)
                boxes = decode_class_boxes(r, pos[:, 1], pos[:, 0], ci)  # (N,7)
                if use_soft_nms:
                    keep = soft_nms(boxes, sc, sigma=0.1, top_k=500)
                else:
                    keep = rotated_nms(boxes, sc, nt, top_k=500)
                if keep.numel() == 0: continue
                boxes = boxes[keep]; sc = sc[keep]
                all_b.append(boxes); all_s.append(sc)
                all_l.append(torch.full((len(sc),), ci + 1,
                                        dtype=torch.long, device=hm.device))
            if all_b:
                out.append({'pred_boxes':  torch.cat(all_b, 0),
                            'pred_scores': torch.cat(all_s, 0),
                            'pred_labels': torch.cat(all_l, 0)})
            else:
                dev = hm.device
                out.append({'pred_boxes':  torch.zeros(0, 7, device=dev),
                            'pred_scores': torch.zeros(0, device=dev),
                            'pred_labels': torch.zeros(0, dtype=torch.long, device=dev)})
        return out

# ═══════════════════════════════════════════════════════════════════════════════
# §10  TOP-LEVEL MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class SGFDualGridModel(nn.Module):
    def __init__(self, vfe_ch=64, neck_ch=128, head_ch=256, num_class=NUM_CLASSES,
                 save_gates: bool = False):
        super().__init__()
        self.vfe      = PillarVFE(out_ch=vfe_ch)
        self.scatter  = PillarScatter(ch=vfe_ch, ny=F_NY, nx=F_NX)
        self.backbone = SGFDualGridBackbone(in_ch=vfe_ch, neck_ch=neck_ch,
                                            out_ch=head_ch, save_gates=save_gates)
        self.head     = SGFCenterHead(input_channels=head_ch, num_class=num_class)

    def forward(self, batch: Dict) -> Dict:
        pillar_feats = self.vfe(batch['voxels'], batch['num_per_v'])
        fine_bev     = self.scatter(pillar_feats, batch['coords'], batch['batch_size'])
        feats, gate_log = self.backbone(fine_bev, batch['coarse_bev'], batch['density_map'])
        pred = self.head(feats)
        return {'fine_bev': fine_bev, 'feats': feats,
                'gate_log': gate_log,
                'dg_logit': self.backbone.neck.dg_logit, **pred}

    def count_params(self) -> Dict[str, int]:
        return {
            'vfe':      sum(p.numel() for p in self.vfe.parameters()),
            'backbone': sum(p.numel() for p in self.backbone.parameters()),
            'head':     sum(p.numel() for p in self.head.parameters()),
            'total':    sum(p.numel() for p in self.parameters()),
        }


# ───────────────────────────── DCGR two-stage (unchanged) ────────────────────

class DCGRHead(nn.Module):
    """Density-Conditioned Gate-guided Refinement (second stage)."""
    def __init__(self, max_pts: int = 64, pts_dim: int = 6,
                 mlp_chs: tuple = (64, 128, 256), gate_dim: int = 1,
                 roi_iou_thresh_pos: float = 0.55, roi_iou_thresh_neg: float = 0.35):
        super().__init__()
        self.max_pts        = max_pts
        self.iou_thresh_pos = roi_iou_thresh_pos
        self.iou_thresh_neg = roi_iou_thresh_neg
        layers: list = []
        in_ch = pts_dim
        for ch in mlp_chs:
            layers.extend([nn.Linear(in_ch, ch, bias=False),
                           nn.LayerNorm(ch), nn.ReLU(inplace=True)])
            in_ch = ch
        self.pointnet = nn.Sequential(*layers)
        roi_dim = mlp_chs[-1]
        self.gate_embed = nn.Sequential(nn.Linear(gate_dim, roi_dim), nn.Sigmoid())
        self.box_head = nn.Sequential(
            nn.Linear(roi_dim, 128, bias=False), nn.LayerNorm(128), nn.ReLU(True),
            nn.Linear(128, 8))
        nn.init.zeros_(self.box_head[-1].weight); nn.init.zeros_(self.box_head[-1].bias)
        self.iou_head = nn.Sequential(
            nn.Linear(roi_dim, 64, bias=False), nn.LayerNorm(64), nn.ReLU(True),
            nn.Linear(64, 1))
        self._roi_dim = roi_dim

    def _sample_gate(self, gate_map, props, stride: float = 2.0):
        B, C, H, W = gate_map.shape
        g_mean = gate_map.mean(1, keepdim=True)
        dev = gate_map.device
        results = []
        for bi in range(B):
            mask = props[:, 0].long() == bi
            if not mask.any(): continue
            xn = 2.0 * (props[mask, 1] - PC_RANGE[0]) / (FINE_VS * stride * W) - 1.0
            yn = 2.0 * (props[mask, 2] - PC_RANGE[1]) / (FINE_VS * stride * H) - 1.0
            grid = torch.stack([xn, yn], -1).view(1, 1, -1, 2)
            samp = F.grid_sample(g_mean[bi:bi+1], grid, mode='bilinear',
                                 padding_mode='border', align_corners=False).squeeze()
            results.append(samp.view(-1, 1))
        return torch.cat(results, 0) if results else props.new_zeros(0, 1)

    @staticmethod
    def _crop_and_encode(pts_np, box, max_pts, expand: float = 1.2):
        box_e = box.copy(); box_e[3] *= expand; box_e[4] *= expand
        inside = points_in_rotated_box(pts_np, box_e)
        p = pts_np[inside]
        if len(p) < 3: return None
        if len(p) > max_pts:
            p = p[np.random.choice(len(p), max_pts, replace=False)]
        cx, cy, cz, l, w, h, ry = box[:7]
        c, s = math.cos(-ry), math.sin(-ry)
        dx = p[:, 0] - cx; dy = p[:, 1] - cy; dz = p[:, 2] - cz
        dxr = c * dx - s * dy
        dyr = s * dx + c * dy
        rng = np.sqrt(p[:, 0]**2 + p[:, 1]**2).reshape(-1, 1)
        inten = p[:, 3:4] if p.shape[1] > 3 else np.zeros((len(p), 1))
        depth = np.sqrt(dxr**2 + dyr**2 + dz**2).reshape(-1, 1) / max(l, w, h, 1e-3)
        return np.concatenate([dxr.reshape(-1,1), dyr.reshape(-1,1),
                                dz.reshape(-1,1), inten, rng, depth], axis=1).astype(np.float32)

    @staticmethod
    def build_roi_targets(props_np, gt_boxes_np, cls_ids,
                          iou_thresh_pos: float = 0.55, iou_thresh_neg: float = 0.35):
        N = len(props_np); M = len(gt_boxes_np)
        delta_t = np.zeros((N, 8), np.float32)
        iou_t   = np.zeros(N, np.float32)
        pos_m   = np.zeros(N, dtype=bool)
        prop_cls = np.zeros(N, dtype=np.int64)
        if M == 0 or N == 0:
            return delta_t, iou_t, pos_m, prop_cls
        for i, prop in enumerate(props_np):
            best_iou = 0.0; best_j = -1
            for j, gt in enumerate(gt_boxes_np):
                iou = rotated_3d_iou(prop[:7], gt[:7])
                if iou > best_iou:
                    best_iou = iou; best_j = j
            iou_t[i] = best_iou
            if best_iou >= iou_thresh_pos and best_j >= 0:
                pos_m[i] = True
                gt = gt_boxes_np[best_j]
                prop_cls[i] = int(cls_ids[best_j]) if best_j < len(cls_ids) else 0
                dx = (gt[0] - prop[0]) / max(prop[3], 1e-3)
                dy = (gt[1] - prop[1]) / max(prop[4], 1e-3)
                dz = (gt[2] - prop[2]) / max(prop[5], 1e-3)
                dl = math.log(max(gt[3], 1e-3) / max(prop[3], 1e-3))
                dw = math.log(max(gt[4], 1e-3) / max(prop[4], 1e-3))
                dh = math.log(max(gt[5], 1e-3) / max(prop[5], 1e-3))
                da = gt[6] - prop[6]
                delta_t[i] = [dx, dy, dz, dl, dw, dh, math.sin(da), math.cos(da)]
        return delta_t, iou_t, pos_m, prop_cls

    def forward_roi(self, pts_batch, proposals, gate_map, gate_stride: float = 2.0):
        N = proposals.shape[0]
        dev = gate_map.device
        if N == 0:
            return {'delta_box': torch.zeros(0,8,device=dev),
                    'iou_pred':  torch.zeros(0,1,device=dev),
                    'valid_mask': torch.zeros(0,dtype=torch.bool,device=dev)}
        roi_feats  = torch.zeros(N, self._roi_dim, device=dev)
        valid_mask = torch.zeros(N, dtype=torch.bool, device=dev)
        for k in range(N):
            bi = int(proposals[k, 0].item())
            box_np = proposals[k, 1:].detach().cpu().numpy()
            pts_np = pts_batch[min(bi, len(pts_batch)-1)]
            enc = self._crop_and_encode(pts_np, box_np, self.max_pts)
            if enc is None: continue
            valid_mask[k] = True
            pt = torch.from_numpy(enc).to(dev)
            f = self.pointnet(pt)
            roi_feats[k] = f.max(0).values
        g_vals = self._sample_gate(gate_map, proposals, stride=gate_stride)
        if g_vals.numel() > 0:
            gate_scale = self.gate_embed(g_vals.to(dev))
            roi_feats  = roi_feats * gate_scale
        return {'delta_box': self.box_head(roi_feats),
                'iou_pred':  self.iou_head(roi_feats),
                'valid_mask': valid_mask, 'roi_features': roi_feats}

    def compute_loss(self, delta_pred, iou_pred, delta_t, iou_t, pos_m,
                     w_box: float = 1.0, w_iou: float = 0.5):
        if pos_m.sum() == 0:
            total = (delta_pred.sum() + iou_pred.sum()) * 0
            return total, {'dcgr_box': 0.0, 'dcgr_iou': 0.0, 'dcgr': 0.0}
        l_box = F.smooth_l1_loss(delta_pred[pos_m], delta_t[pos_m])
        l_iou = F.binary_cross_entropy_with_logits(iou_pred.squeeze(1), iou_t, reduction='mean')
        total = w_box * l_box + w_iou * l_iou
        return total, {'dcgr_box': float(l_box.detach()),
                       'dcgr_iou': float(l_iou.detach()), 'dcgr': float(total.detach())}

    @staticmethod
    def apply_deltas(proposals, delta_box):
        r = proposals.clone()
        l, w, h = proposals[:, 4], proposals[:, 5], proposals[:, 6]
        r[:, 1] = proposals[:, 1] + delta_box[:, 0] * l
        r[:, 2] = proposals[:, 2] + delta_box[:, 1] * w
        r[:, 3] = proposals[:, 3] + delta_box[:, 2] * h
        r[:, 4] = l * delta_box[:, 3].clamp(-1, 1).exp()
        r[:, 5] = w * delta_box[:, 4].clamp(-1, 1).exp()
        r[:, 6] = h * delta_box[:, 5].clamp(-1, 1).exp()
        r[:, 7] = proposals[:, 7] + torch.atan2(delta_box[:, 6], delta_box[:, 7])
        return r


class SGFTwoStageModel(nn.Module):
    GT_NOISE_XY  = 0.5
    GT_NOISE_ANG = 0.1

    def __init__(self, vfe_ch=64, neck_ch=128, head_ch=256, num_class=NUM_CLASSES,
                 dcgr_max_pts=64, dcgr_mlp=(64, 128, 256)):
        super().__init__()
        self.stage1 = SGFDualGridModel(vfe_ch=vfe_ch, neck_ch=neck_ch,
                                       head_ch=head_ch, num_class=num_class,
                                       save_gates=True)
        self.dcgr   = DCGRHead(max_pts=dcgr_max_pts, pts_dim=6, mlp_chs=dcgr_mlp)
        self.num_class = num_class

    def _make_gt_proposals(self, gt_boxes, bi):
        if len(gt_boxes) == 0:
            return torch.zeros(0, 8)
        noise_xy  = np.random.uniform(-self.GT_NOISE_XY,  self.GT_NOISE_XY,  (len(gt_boxes), 2))
        noise_ang = np.random.uniform(-self.GT_NOISE_ANG, self.GT_NOISE_ANG, (len(gt_boxes), 1))
        props = gt_boxes[:, :7].copy()
        props[:, :2]  += noise_xy
        props[:, 6:7] += noise_ang
        b_idx = np.full((len(props), 1), bi, dtype=np.float32)
        return torch.from_numpy(np.concatenate([b_idx, props], axis=1).astype(np.float32))

    def forward(self, batch: Dict) -> Dict:
        out1 = self.stage1(batch)
        gate_G = self.stage1.backbone.neck.gate_maps.get('G', None)
        pts_batch: list = batch.get('pts_batch', [])

        if self.training and 'gt_boxes' in batch and gate_G is not None:
            all_proposals = []
            for bi, gt in enumerate(batch['gt_boxes']):
                if len(gt):
                    all_proposals.append(self._make_gt_proposals(gt, bi))
            if all_proposals:
                proposals = torch.cat(all_proposals, 0).to(gate_G.device)
                dcgr_out = self.dcgr.forward_roi(pts_batch, proposals, gate_G, gate_stride=2.0)
                out1.update({'dcgr_delta_box': dcgr_out['delta_box'],
                             'dcgr_iou_pred':  dcgr_out['iou_pred'],
                             'dcgr_valid':     dcgr_out['valid_mask'],
                             'dcgr_proposals': proposals})
        elif not self.training and gate_G is not None:
            preds1 = self.stage1.head.post_process(
                {'hm': out1['hm'], 'reg': out1['reg'], 'iou': out1['iou']})
            all_proposals = []
            for bi, pd in enumerate(preds1):
                if pd['pred_boxes'].numel() > 0:
                    b_idx = torch.full((len(pd['pred_boxes']), 1), bi,
                                       dtype=torch.float32, device=pd['pred_boxes'].device)
                    all_proposals.append(torch.cat([b_idx, pd['pred_boxes']], 1))
            if all_proposals:
                proposals = torch.cat(all_proposals, 0)
                dcgr_out = self.dcgr.forward_roi(pts_batch, proposals, gate_G, gate_stride=2.0)
                refined = DCGRHead.apply_deltas(proposals, dcgr_out['delta_box'])
                final_score = torch.sigmoid(dcgr_out['iou_pred'].squeeze(1))
                out1['stage2_refined'] = refined
                out1['stage2_scores']  = final_score
                out1['stage1_preds']   = preds1
        return out1

    def count_params(self) -> Dict[str, int]:
        s1 = self.stage1.count_params()
        s2 = sum(p.numel() for p in self.dcgr.parameters())
        s1['dcgr'] = s2; s1['total'] += s2
        return s1


# ═══════════════════════════════════════════════════════════════════════════════
# §12  KITTI AP EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

def _gt_subset_by_difficulty(gts, cls_name, diff):
    diff_rank = {'Easy': 0, 'Moderate': 1, 'Hard': 2, 'Unknown': 99}
    rank_now  = diff_rank[diff]
    valid, ignore = [], []
    for g in gts:
        if g['cls_name'] != cls_name:
            continue
        gr = diff_rank.get(g['difficulty'], 99)
        if gr <= rank_now:
            valid.append(g)
        else:
            ignore.append(g)
    return valid, ignore


def _compute_ap_40point(rec, prec):
    if len(rec) == 0:
        return 0.0
    recall_thresh = np.linspace(1/40, 1.0, 40)
    ap = 0.0
    for r in recall_thresh:
        mask = rec >= r
        if mask.any():
            ap += float(prec[mask].max())
    return ap / 40.0 * 100.0


def evaluate_kitti(all_preds, all_gts, iou_mode: str = '3d'):
    assert iou_mode in ('3d', 'bev')
    iou_fn = rotated_3d_iou if iou_mode == '3d' else rotated_bev_iou
    iou_thr_map = IOU_THRESH_3D if iou_mode == '3d' else IOU_THRESH_BEV
    results: Dict[str, Dict[str, float]] = {}
    for ci, cls in enumerate(CLASS_NAMES):
        cls_id  = CLASS_TO_ID[cls]
        iou_thr = iou_thr_map[cls]
        results[cls] = {}
        for diff in ('Easy', 'Moderate', 'Hard'):
            score_list, tp_list, fp_list = [], [], []
            n_valid_gt = 0
            for fi, (pred, gts) in enumerate(zip(all_preds, all_gts)):
                valid_g, ignore_g = _gt_subset_by_difficulty(gts, cls, diff)
                n_valid_gt += len(valid_g)
                valid_boxes  = np.stack([g['box'] for g in valid_g], 0)  if valid_g  else np.zeros((0, 8))
                ignore_boxes = np.stack([g['box'] for g in ignore_g], 0) if ignore_g else np.zeros((0, 8))
                p_idx = np.where(pred['labels'] == cls_id)[0]
                if len(p_idx) == 0: continue
                p_box = pred['boxes'][p_idx]; p_sc = pred['scores'][p_idx]
                ord_ = np.argsort(-p_sc)
                p_box = p_box[ord_]; p_sc = p_sc[ord_]
                used_valid = np.zeros(len(valid_boxes), dtype=bool)
                for k in range(len(p_box)):
                    best_iou_v, best_jv = -1.0, -1
                    for j in range(len(valid_boxes)):
                        if used_valid[j]: continue
                        v = iou_fn(p_box[k], valid_boxes[j])
                        if v > best_iou_v: best_iou_v, best_jv = v, j
                    best_iou_i = -1.0
                    for j in range(len(ignore_boxes)):
                        v = iou_fn(p_box[k], ignore_boxes[j])
                        if v > best_iou_i: best_iou_i = v
                    is_tp = is_fp = is_ignore = False
                    if best_iou_v >= iou_thr:
                        used_valid[best_jv] = True; is_tp = True
                    elif best_iou_i >= iou_thr:
                        is_ignore = True
                    else:
                        is_fp = True
                    if is_ignore: continue
                    score_list.append(float(p_sc[k]))
                    tp_list.append(1 if is_tp else 0)
                    fp_list.append(1 if is_fp else 0)
            if n_valid_gt == 0 or not score_list:
                results[cls][diff] = 0.0; continue
            sc = np.array(score_list); tp = np.array(tp_list); fp = np.array(fp_list)
            ord_ = np.argsort(-sc)
            tp = tp[ord_]; fp = fp[ord_]
            tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
            rec  = tp_c / max(n_valid_gt, 1)
            prec = tp_c / np.maximum(tp_c + fp_c, 1)
            results[cls][diff] = _compute_ap_40point(rec, prec)
    return results


def format_ap_table(ap3d, apbev) -> str:
    L = []
    L.append("=" * 78)
    L.append(f"{'Class':<12} {'Metric':<8} {'IoU':>5} {'Easy':>10} {'Moderate':>10} {'Hard':>10}")
    L.append("-" * 78)
    for cls in CLASS_NAMES:
        for label, table, t_map in [('AP3D', ap3d, IOU_THRESH_3D),
                                     ('APBEV', apbev, IOU_THRESH_BEV)]:
            row = table.get(cls, {})
            L.append(f"{cls:<12} {label:<8} {t_map[cls]:>5.2f} "
                     f"{row.get('Easy', 0):>10.4f} "
                     f"{row.get('Moderate', 0):>10.4f} "
                     f"{row.get('Hard', 0):>10.4f}")
        L.append("-" * 78)
    mod_3d  = np.mean([ap3d[c].get('Moderate', 0)  for c in CLASS_NAMES])
    mod_bev = np.mean([apbev[c].get('Moderate', 0) for c in CLASS_NAMES])
    L.append(f"{'mAP@Mod':<12} {'AP3D':<8} {'':>5} {'':>10} {mod_3d:>10.4f}")
    L.append(f"{'mAP@Mod':<12} {'APBEV':<8} {'':>5} {'':>10} {mod_bev:>10.4f}")
    L.append("=" * 78)
    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════════════
# §13  VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def _setup_pub_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'savefig.edgecolor': 'white',
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'font.family': 'serif', 'font.size': 9,
        'axes.linewidth': 0.8, 'axes.edgecolor': '#333333',
        'axes.labelsize': 10, 'axes.titlesize': 11,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8, 'legend.frameon': True, 'legend.framealpha': 0.95,
        'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    })


def _box_to_polygon(box: np.ndarray) -> np.ndarray:
    return box_corners_2d(box[0], box[1], box[3], box[4], box[6])


# Eval visualization format (PNG only — do not use PDF for Slurm viz outputs).
VIZ_IMAGE_EXT = '.png'


def box_corners_3d(x, y, z, l, w, h, ry) -> np.ndarray:
    """8 corners (8, 3) in velodyne frame."""
    c, s = math.cos(ry), math.sin(ry)
    x_c = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2], dtype=np.float64)
    y_c = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2], dtype=np.float64)
    z_c = np.array([ h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2], dtype=np.float64)
    R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], dtype=np.float64)
    corners = R @ np.vstack([x_c, y_c, z_c])
    corners[0] += x; corners[1] += y; corners[2] += z
    return corners.T


def viz_bev_detection(pts, gt_boxes, pred_boxes, pred_scores, pred_labels,
                      save_path: str, title: str = '') -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    _setup_pub_style()
    pc = PC_RANGE
    ok = ((pts[:, 0] >= pc[0]) & (pts[:, 0] <= pc[3]) &
          (pts[:, 1] >= pc[1]) & (pts[:, 1] <= pc[4]))
    p = pts[ok]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(p[:, 0], p[:, 1], s=0.05, c='#888888', alpha=0.5,
               linewidths=0, rasterized=True)
    gt_patches = [Polygon(_box_to_polygon(b), closed=True) for b in gt_boxes]
    if gt_patches:
        ax.add_collection(PatchCollection(gt_patches, facecolor='none',
            edgecolor='#1b7e3d', linewidth=1.4, linestyle='-', label='GT'))
    pred_patches = [Polygon(_box_to_polygon(b), closed=True) for b in pred_boxes]
    if pred_patches:
        ax.add_collection(PatchCollection(pred_patches, facecolor='none',
            edgecolor='#c8302a', linewidth=1.0, linestyle='--'))
        for b, sc, lb in zip(pred_boxes, pred_scores, pred_labels):
            cname = CLASS_NAMES[int(lb) - 1] if 1 <= int(lb) <= len(CLASS_NAMES) else '?'
            ax.text(b[0], b[1], f'{cname[:3]}\n{sc:.2f}', fontsize=5.5, color='#c8302a',
                    ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.6))
    ax.set_xlim(pc[0], pc[3]); ax.set_ylim(pc[1], pc[4])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (m)  — forward'); ax.set_ylabel('y (m)  — left')
    if title: ax.set_title(title)
    ax.grid(True, linewidth=0.4, color='#cccccc')
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color='#888888', marker='.', linestyle='None', markersize=3, label='LiDAR points'),
        Line2D([], [], color='#1b7e3d', linestyle='-',  linewidth=1.4, label='Ground truth'),
        Line2D([], [], color='#c8302a', linestyle='--', linewidth=1.0, label='Prediction'),
    ]
    ax.legend(handles=handles, loc='upper right')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def viz_3d_detection(pts: np.ndarray,
                     gt_boxes: np.ndarray,
                     pred_boxes: np.ndarray,
                     pred_scores: np.ndarray,
                     pred_labels: np.ndarray,
                     save_path: str,
                     title: str = '',
                     max_pts: int = 12000) -> None:
    """3D scene: LiDAR + oriented 3D box wireframes (GT green, pred red)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    _setup_pub_style()
    pc = PC_RANGE
    ok = ((pts[:, 0] >= pc[0]) & (pts[:, 0] <= pc[3]) &
          (pts[:, 1] >= pc[1]) & (pts[:, 1] <= pc[4]) &
          (pts[:, 2] >= pc[2]) & (pts[:, 2] <= pc[5]))
    p = pts[ok]
    if len(p) > max_pts:
        idx = np.random.default_rng(0).choice(len(p), max_pts, replace=False)
        p = p[idx]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=0.05, c='#888888', alpha=0.35,
               linewidths=0, depthshade=False)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    def _draw_boxes(boxes, color, lw, ls='-'):
        for b in boxes:
            c = box_corners_3d(b[0], b[1], b[2], b[3], b[4], b[5], b[6])
            for i, j in edges:
                ax.plot([c[i,0],c[j,0]], [c[i,1],c[j,1]], [c[i,2],c[j,2]],
                        color=color, linewidth=lw, linestyle=ls, alpha=0.9)
    if len(gt_boxes):   _draw_boxes(gt_boxes,   '#1b7e3d', 1.4, '-')
    if len(pred_boxes): _draw_boxes(pred_boxes, '#c8302a', 1.0, '--')
    for b, sc, lb in zip(pred_boxes, pred_scores, pred_labels):
        cname = CLASS_NAMES[int(lb)-1] if 1 <= int(lb) <= len(CLASS_NAMES) else '?'
        ax.text(b[0], b[1], b[2]+b[5]*0.6, f'{cname[:3]} {sc:.2f}', fontsize=6, color='#c8302a')
    ax.set_xlim(pc[0], pc[3]); ax.set_ylim(pc[1], pc[4]); ax.set_zlim(pc[2], pc[5])
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    ax.view_init(elev=22, azim=-58)
    if title: ax.set_title(title)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color='#888888', marker='.', linestyle='None', markersize=4, label='LiDAR'),
        Line2D([], [], color='#1b7e3d', linewidth=1.4, label='GT 3D box'),
        Line2D([], [], color='#c8302a', linewidth=1.0, linestyle='--', label='Pred 3D box'),
    ], loc='upper left', fontsize=8)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def viz_gate_histograms(gate_maps: Dict[str, np.ndarray], save_path: str) -> None:
    import matplotlib.pyplot as plt
    _setup_pub_style()
    keys = [k for k in ('G_s', 'G_d', 'G', 'dg_logit') if k in gate_maps]
    if not keys:
        keys = [k for k in gate_maps if k.startswith('G') or k == 'dg_logit'][:4]
    if not keys:
        keys = list(gate_maps.keys())[:3]
    fig, axes = plt.subplots(1, max(1, len(keys)), figsize=(3.6 * max(1, len(keys)), 3.0))
    if len(keys) == 1:
        axes = [axes]
    colors = {'G_s': '#1f77b4', 'G_d': '#2ca02c', 'G': '#7f3aa3'}
    for ax, key in zip(axes, keys):
        color = colors.get(key, '#444444')
        v = gate_maps[key].ravel()
        ax.hist(v, bins=50, range=(0, 1), color=color, alpha=0.75,
                edgecolor='white', linewidth=0.4)
        ax.set_title(f'{key}  (mean={v.mean():.3f}, std={v.std():.3f})')
        ax.set_xlabel(key); ax.set_ylabel('Frequency')
        ax.axvline(0.5, color='#333333', linestyle=':', linewidth=0.6)
        ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# §14  DATASET + COLLATE
# ═══════════════════════════════════════════════════════════════════════════════

class KittiSGFDataset(Dataset):
    def __init__(self, kitti_root: str, split_file: str, training: bool,
                 augment: bool = True, keep_labels_for_eval: bool = False,
                 max_voxels: Optional[int] = None, gt_sampler: Optional[Any] = None,
                 data_subset: str = 'training'):
        self.root = Path(kitti_root)
        self.data_subset = data_subset
        if self.data_subset not in ('training', 'testing'):
            raise ValueError("data_subset must be 'training' or 'testing'")
        self.training = training
        self.augment_flag = augment and training
        self.keep_labels_for_eval = keep_labels_for_eval
        self.max_voxels = max_voxels or (MAX_VOXELS_TRAIN if training else MAX_VOXELS_EVAL)
        self.gt_sampler = gt_sampler if training else None
        with open(split_file, 'r') as f:
            self.sample_ids = [ln.strip() for ln in f if ln.strip()]

    def __len__(self): return len(self.sample_ids)

    def __getitem__(self, idx) -> Dict[str, Any]:
        sid = self.sample_ids[idx]
        sub = self.data_subset
        velo_p  = self.root / sub / 'velodyne' / f'{sid}.bin'
        calib_p = self.root / sub / 'calib'    / f'{sid}.txt'
        label_p = self.root / 'training' / 'label_2' / f'{sid}.txt'

        pts    = load_points_bin(str(velo_p))
        calib  = parse_calib(str(calib_p))
        labels = parse_label(str(label_p), calib) if sub == 'training' else []
        gt     = gt_array_for_training(labels)

        if self.gt_sampler is not None:
            pts, gt = self.gt_sampler(pts, gt)
        if self.augment_flag:
            pts, gt = augment(pts, gt)

        voxels, coords, num_per_v = voxelize_pillars(
            pts, max_voxels=self.max_voxels, max_pts=MAX_PTS)
        coarse_bev  = make_coarse_bev(pts)
        density_map = make_density_map(pts)

        if self.training:
            hm_t, reg_t, iou_t, pos_t = build_targets(gt, NUM_CLASSES, F_NY, F_NX)
        else:
            hm_t = reg_t = iou_t = pos_t = None

        out: Dict[str, Any] = {
            'voxels': voxels, 'coords': coords, 'num_per_v': num_per_v,
            'coarse_bev': coarse_bev, 'density_map': density_map,
            'gt_boxes': gt, 'sample_id': sid, 'pts': pts,
        }
        if self.training:
            out.update({'hm_t': hm_t, 'reg_t': reg_t, 'iou_t': iou_t, 'pos_t': pos_t})
        if self.keep_labels_for_eval:
            out['labels'] = labels
        return out


def collate_kitti(batch_list: List[Dict]) -> Dict[str, Any]:
    B = len(batch_list)
    voxels_l, coords_l, num_l = [], [], []
    for bi, s in enumerate(batch_list):
        v = s['voxels']; c = s['coords']; n = s['num_per_v']
        if v.shape[0] > 0:
            c = c.copy(); c[:, 0] = bi
            voxels_l.append(v); coords_l.append(c); num_l.append(n)
    if voxels_l:
        voxels    = np.concatenate(voxels_l, axis=0)
        coords    = np.concatenate(coords_l, axis=0)
        num_per_v = np.concatenate(num_l,    axis=0)
    else:
        voxels    = np.zeros((0, MAX_PTS, 4), np.float32)
        coords    = np.zeros((0, 3), np.int32)
        num_per_v = np.zeros((0,),   np.int32)

    coarse_bev  = np.stack([s['coarse_bev']  for s in batch_list], axis=0)
    density_map = np.stack([s['density_map'] for s in batch_list], axis=0)
    gt_boxes    = [s['gt_boxes'] for s in batch_list]
    sample_ids  = [s['sample_id'] for s in batch_list]

    out: Dict[str, Any] = {
        'voxels':       torch.from_numpy(voxels),
        'coords':       torch.from_numpy(coords).long(),
        'num_per_v':    torch.from_numpy(num_per_v).long(),
        'coarse_bev':   torch.from_numpy(coarse_bev),
        'density_map':  torch.from_numpy(density_map),
        'gt_boxes':     gt_boxes,
        'batch_size':   B,
        'sample_ids':   sample_ids,
    }
    if 'hm_t' in batch_list[0] and batch_list[0]['hm_t'] is not None:
        out['hm_t']  = torch.from_numpy(np.stack([s['hm_t']  for s in batch_list], 0))
        out['reg_t'] = torch.from_numpy(np.stack([s['reg_t'] for s in batch_list], 0))
        out['iou_t'] = torch.from_numpy(np.stack([s['iou_t'] for s in batch_list], 0))
        out['pos_t'] = torch.from_numpy(np.stack([s['pos_t'] for s in batch_list], 0))
    if 'labels' in batch_list[0]:
        out['labels'] = [s['labels'] for s in batch_list]
    if 'pts' in batch_list[0]:
        out['pts_batch'] = [s['pts'] for s in batch_list]
    return out


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


# ───────────────────────── production helpers (EMA/TTA/sampler/db) ───────────

class WeightEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def swap_in(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        msd = model.state_dict()
        backup = {k: msd[k].detach().clone() for k in self.shadow}
        for k, v in self.shadow.items():
            msd[k].copy_(v)
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        msd = model.state_dict()
        for k, v in backup.items():
            msd[k].copy_(v)


def flip_batch_y(batch: Dict) -> Dict:
    out = dict(batch)
    out['coarse_bev']  = torch.flip(batch['coarse_bev'],  dims=[2])
    out['density_map'] = torch.flip(batch['density_map'], dims=[2])
    voxels_f = batch['voxels'].clone()
    voxels_f[..., 1] = -voxels_f[..., 1]
    coords_f = batch['coords'].clone()
    coords_f[:, 1] = (F_NY - 1) - coords_f[:, 1]
    out['voxels'] = voxels_f; out['coords'] = coords_f
    return out


@torch.no_grad()
def tta_infer(model, batch: Dict) -> List[Dict[str, torch.Tensor]]:
    out1 = model(batch)
    p1 = model.head.post_process({'hm': out1['hm'], 'reg': out1['reg'], 'iou': out1['iou']})
    out2 = model(flip_batch_y(batch))
    p2 = model.head.post_process({'hm': out2['hm'], 'reg': out2['reg'], 'iou': out2['iou']})
    merged: List[Dict[str, torch.Tensor]] = []
    for s1, s2 in zip(p1, p2):
        b2 = s2['pred_boxes'].clone()
        if b2.numel():
            b2[:, 1] = -b2[:, 1]; b2[:, 6] = -b2[:, 6]
        ab   = torch.cat([s1['pred_boxes'],  b2],                 0)
        asc  = torch.cat([s1['pred_scores'], s2['pred_scores']],  0)
        alab = torch.cat([s1['pred_labels'], s2['pred_labels']],  0)
        keep_all: List[torch.Tensor] = []
        for cls in range(1, NUM_CLASSES + 1):
            mask = alab == cls
            if not mask.any(): continue
            cb = ab[mask]; cs = asc[mask]
            keep = rotated_nms(cb, cs, iou_thresh=0.10, top_k=500)
            keep_all.append(mask.nonzero(as_tuple=False).squeeze(1)[keep])
        if keep_all:
            kk = torch.cat(keep_all, 0)
            merged.append({'pred_boxes': ab[kk], 'pred_scores': asc[kk], 'pred_labels': alab[kk]})
        else:
            merged.append({'pred_boxes': ab[:0], 'pred_scores': asc[:0], 'pred_labels': alab[:0]})
    return merged


def make_class_balanced_sampler(dataset: KittiSGFDataset, boost_per_minority: float = 1.0):
    from torch.utils.data import WeightedRandomSampler
    sub = getattr(dataset, 'data_subset', 'training')
    weights = np.ones(len(dataset), dtype=np.float64)
    for i in range(len(dataset)):
        sid = dataset.sample_ids[i]
        label_p = dataset.root / sub / 'label_2' / f'{sid}.txt'
        calib_p = dataset.root / sub / 'calib'   / f'{sid}.txt'
        if not (label_p.exists() and calib_p.exists()): continue
        try:
            calib = parse_calib(str(calib_p)); labs = parse_label(str(label_p), calib)
        except Exception:
            continue
        for l in labs:
            if l['cls_id'] in (2, 3):
                weights[i] += boost_per_minority
    return WeightedRandomSampler(weights.tolist(), num_samples=len(dataset), replacement=True)


def build_gt_db(args):
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ds = KittiSGFDataset(args.kitti_root, args.train_split, training=False,
                         augment=False, keep_labels_for_eval=True)
    db: Dict[str, List[Dict]] = {n: [] for n in CLASS_NAMES}
    t0 = time.time()
    for i in range(len(ds)):
        s = ds[i]; pts = s['pts']
        for lab in s['labels']:
            if lab['cls_id'] < 1: continue
            box = lab['box']
            inside = points_in_rotated_box(pts, box)
            if int(inside.sum()) < 5: continue
            db[lab['cls_name']].append({'box': box.copy(), 'pts': pts[inside].copy(),
                                        'sample_id': s['sample_id']})
        if (i + 1) % 200 == 0:
            counts = {k: len(v) for k, v in db.items()}
            print(f"  [build_db] {i+1}/{len(ds)} ({time.time()-t0:.1f}s) counts={counts}")
    out_path = out_dir / 'gt_database.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(db, f)
    print(f"[build_db] saved {out_path}")
    print(f"[build_db] final counts: { {k: len(v) for k, v in db.items()} }")


# ═══════════════════════════════════════════════════════════════════════════════
# §15  TRAINER
# ═══════════════════════════════════════════════════════════════════════════════

class TextLogger:
    def __init__(self, jsonl_path: str, banner: str = ''):
        self.f = open(jsonl_path, 'a', buffering=1)
        if banner:
            print(banner); self.f.write(json.dumps({'banner': banner}) + '\n')
    def log(self, d: Dict[str, Any], pretty_prefix: str = ''):
        pretty = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in d.items())
        print(f"{pretty_prefix}{pretty}")
        sanitized = {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in d.items()}
        self.f.write(json.dumps(sanitized) + '\n')
    def close(self): self.f.close()


def train_one_epoch(model, loader, opt, sched, scaler, dev, epoch, logger,
                    log_every, use_amp, grad_clip: float = 2.0,
                    ema: Optional[WeightEMA] = None, two_stage: bool = False):
    model.train()
    t_ep = time.time()
    agg = {'loss': 0.0, 'loss_hm': 0.0, 'loss_reg': 0.0, 'loss_iou': 0.0, 'loss_aux': 0.0, 'dcgr': 0.0, 'n': 0}
    for it, batch in enumerate(loader):
        batch = move_batch_to_device(batch, dev)
        opt.zero_grad(set_to_none=True)
        with amp_autocast(use_amp):
            out = model(batch)
            head = model.stage1.head if two_stage else model.head
            loss, tb = head.compute_loss(
                pred={'hm': out['hm'], 'reg': out['reg'], 'iou': out['iou']},
                hm_t=batch['hm_t'], reg_t=batch['reg_t'],
                iou_t=batch['iou_t'], pos_t=batch['pos_t'], gate_log=out['gate_log'],
                aux_dg_logit=out.get('dg_logit'))
            if two_stage and 'dcgr_delta_box' in out:
                props_np = out['dcgr_proposals'].detach().cpu().numpy()
                dcgr_loss_total = loss.new_zeros(1)
                for bi in range(batch['batch_size']):
                    mask_bi = props_np[:, 0] == bi
                    if not mask_bi.any(): continue
                    gt_np = batch['gt_boxes'][bi]
                    cls_ids = gt_np[:, 7].astype(int) if gt_np.ndim > 1 and gt_np.shape[1] > 7 else np.zeros(len(gt_np), int)
                    dt, it_t, pm, _ = DCGRHead.build_roi_targets(
                        props_np[mask_bi, 1:], gt_np[:, :7] if len(gt_np) else np.zeros((0,7)), cls_ids)
                    dt_t  = torch.from_numpy(dt).to(dev)
                    iou_tt= torch.from_numpy(it_t).to(dev)
                    pm_t  = torch.from_numpy(pm).to(dev)
                    idx   = np.where(mask_bi)[0]
                    dl, _ = model.dcgr.compute_loss(
                        out['dcgr_delta_box'][idx], out['dcgr_iou_pred'][idx], dt_t, iou_tt, pm_t)
                    dcgr_loss_total = dcgr_loss_total + dl
                loss = loss + 0.5 * dcgr_loss_total
                tb['dcgr'] = float(dcgr_loss_total.detach())

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        if sched is not None: sched.step()
        if ema is not None: ema.update(model)

        agg['loss'] += tb['loss']; agg['loss_hm'] += tb['loss_hm']
        agg['loss_reg'] += tb['loss_reg']; agg['loss_iou'] += tb['loss_iou']
        agg['loss_aux'] += tb.get('loss_aux', 0.0); agg['dcgr'] += tb.get('dcgr', 0.0); agg['n'] += 1
        if it % log_every == 0:
            row = {**tb, 'epoch': epoch, 'iter': it,
                   'lr': float(opt.param_groups[0]['lr']), 'gnorm': float(gn)}
            logger.log(row, pretty_prefix=f"[train ep{epoch:03d} it{it:05d}] ")
    avg = {k: (agg[k] / max(agg['n'], 1)) for k in ('loss', 'loss_hm', 'loss_reg', 'loss_iou', 'loss_aux', 'dcgr')}
    avg['epoch_time'] = time.time() - t_ep; avg['epoch'] = epoch
    logger.log(avg, pretty_prefix=f"[train ep{epoch:03d} SUMMARY] ")
    return avg


def train(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train] device = {dev}")
    torch.backends.cudnn.benchmark = True
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_log = TextLogger(str(out_dir / 'train_log.jsonl'),
                           banner=f'SGF-DualGrid(BCAF-DH) training start  out_dir={out_dir}')
    val_log   = TextLogger(str(out_dir / 'val_log.jsonl'), banner='SGF-DualGrid(BCAF-DH) validation')

    gt_sampler = None
    if args.gt_db_path and os.path.exists(args.gt_db_path):
        print(f"[train] loading GT database from {args.gt_db_path}")
        db = load_gt_database(args.gt_db_path)
        print(f"[train] GT-DB counts: { {k: len(v) for k, v in db.items()} }")
        samples_per_class = {'Car': args.gt_sample_car, 'Pedestrian': args.gt_sample_ped,
                             'Cyclist': args.gt_sample_cyc}
        gt_sampler = (lambda pts, gt, _db=db, _spc=samples_per_class: gt_sample(pts, gt, _db, _spc))
        print(f"[train] GT-sampling enabled: {samples_per_class}")
    elif args.gt_db_path:
        print(f"[train] WARNING: --gt_db_path={args.gt_db_path} not found; no GT-sampling")

    ds_train = KittiSGFDataset(args.kitti_root, args.train_split, training=True,
                               augment=True, gt_sampler=gt_sampler)

    if args.cls_balanced:
        print("[train] building class-balanced sampler...")
        t0 = time.time()
        sampler = make_class_balanced_sampler(ds_train)
        print(f"[train] class-balanced sampler ready ({time.time()-t0:.1f}s)")
        loader_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, collate_fn=collate_kitti, pin_memory=True,
            persistent_workers=(args.num_workers > 0), drop_last=True)
    else:
        loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_kitti, pin_memory=True,
            persistent_workers=(args.num_workers > 0), drop_last=True)

    if len(loader_train) == 0:
        raise RuntimeError(f"[train] Training DataLoader has 0 batches "
                           f"(len={len(ds_train)}, bs={args.batch_size}, drop_last=True).")

    do_val = bool(args.val_split)
    if do_val:
        ds_val = KittiSGFDataset(args.kitti_root, args.val_split, training=False,
                                 augment=False, keep_labels_for_eval=True)
        loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
            num_workers=max(2, args.num_workers // 2), collate_fn=collate_kitti,
            pin_memory=True, drop_last=False)
        print(f"[train] train size={len(ds_train)}  val size={len(ds_val)}")
    else:
        loader_val = None
        print(f"[train] train size={len(ds_train)}  (no val)")

    model = (SGFTwoStageModel() if args.two_stage else SGFDualGridModel()).to(dev)
    pc = model.count_params()
    if args.two_stage:
        print(f"[train] TWO-STAGE: total={pc['total']:,} (stage1={pc['total']-pc['dcgr']:,} dcgr={pc['dcgr']:,})")
    else:
        print(f"[train] params total={pc['total']:,} (vfe={pc['vfe']:,} bb={pc['backbone']:,} head={pc['head']:,})")

    ema = None
    if args.ema:
        ema = WeightEMA(model, decay=args.ema_decay)
        print(f"[train] EMA enabled (decay={args.ema_decay})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * max(len(loader_train), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps,
        pct_start=0.2, div_factor=10, final_div_factor=100)
    use_amp = (dev.type == 'cuda' and args.amp)
    scaler  = amp_grad_scaler(use_amp)

    start_epoch = 0; best_metric = -1.0
    if args.resume and os.path.exists(args.resume):
        print(f"[train] resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ckpt['model']); opt.load_state_dict(ckpt['opt'])
        if 'scaler' in ckpt and use_amp: scaler.load_state_dict(ckpt['scaler'])
        if 'sched' in ckpt: sched.load_state_dict(ckpt['sched'])
        if ema is not None and ckpt.get('ema'):
            for k, v in ckpt['ema'].items():
                if k in ema.shadow: ema.shadow[k].copy_(v)
        start_epoch = ckpt.get('epoch', -1) + 1
        best_metric = ckpt.get('best_metric', -1.0)
        print(f"[train] resumed at epoch {start_epoch}, best_metric={best_metric:.4f}")

    for ep in range(start_epoch, args.epochs):
        train_one_epoch(model, loader_train, opt, sched, scaler, dev, ep,
                        train_log, args.log_every, use_amp, ema=ema,
                        two_stage=getattr(args, 'two_stage', False))
        ckpt = {'model': model.state_dict(), 'opt': opt.state_dict(),
                'sched': sched.state_dict(), 'scaler': scaler.state_dict(),
                'ema': ema.shadow if ema is not None else None,
                'epoch': ep, 'best_metric': best_metric}
        torch.save(ckpt, out_dir / 'last.pt')

        if do_val and (((ep + 1) % args.val_interval == 0) or (ep + 1 == args.epochs)):
            print(f"[val ep{ep:03d}] running validation"
                  f"{' with EMA' if ema is not None else ''}{' + TTA' if args.tta else ''}...")
            backup = ema.swap_in(model) if ema is not None else None
            try:
                metric_3d_mod, ap3d, apbev = _run_eval(
                    model, loader_val, dev, tta=args.tta,
                    soft_nms=getattr(args, 'soft_nms', False),
                    two_stage=getattr(args, 'two_stage', False))
            finally:
                if ema is not None and backup is not None:
                    ema.restore(model, backup)
            row = {'epoch': ep, 'mAP3D_mod': metric_3d_mod,
                   **{f'AP3D_{c}_Mod':  ap3d[c]['Moderate']  for c in CLASS_NAMES},
                   **{f'APBEV_{c}_Mod': apbev[c]['Moderate'] for c in CLASS_NAMES}}
            val_log.log(row, pretty_prefix=f"[val ep{ep:03d}] ")
            if metric_3d_mod > best_metric:
                best_metric = metric_3d_mod
                if ema is not None:
                    backup2 = ema.swap_in(model)
                    torch.save({'model': model.state_dict(), 'epoch': ep,
                                'best_metric': best_metric, 'ema_used': True}, out_dir / 'best.pt')
                    ema.restore(model, backup2)
                else:
                    torch.save({'model': model.state_dict(), 'epoch': ep,
                                'best_metric': best_metric, 'ema_used': False}, out_dir / 'best.pt')
                print(f"[val ep{ep:03d}] NEW BEST mAP@Mod={best_metric:.4f} saved to best.pt")

    train_log.close(); val_log.close()
    print(f"[train] DONE.  best mAP@Mod = {best_metric:.4f}  out_dir={out_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# §16  EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _run_eval(model, loader, dev, tta=False, soft_nms=False, two_stage=False):
    model.eval()
    all_preds: List[Dict[str, np.ndarray]] = []
    all_gts:   List[List[Dict]]            = []
    for batch in loader:
        batch = move_batch_to_device(batch, dev)
        with torch.no_grad():
            if tta:
                pred_list = tta_infer(model.stage1 if two_stage else model, batch)
            elif two_stage:
                out = model(batch)
                if 'stage2_refined' in out:
                    pred_list = []
                    for bi in range(batch['batch_size']):
                        mask = out['stage2_refined'][:, 0].long() == bi
                        if mask.any():
                            r = out['stage2_refined'][mask, 1:]; s = out['stage2_scores'][mask]
                            l = out['stage1_preds'][bi]['pred_labels'][:mask.sum()]
                            pred_list.append({'pred_boxes': r, 'pred_scores': s, 'pred_labels': l})
                        else:
                            pred_list.append({'pred_boxes': torch.zeros(0,7,device=dev),
                                              'pred_scores': torch.zeros(0,device=dev),
                                              'pred_labels': torch.zeros(0,dtype=torch.long,device=dev)})
                else:
                    out_s1 = model.stage1(batch)
                    pred_list = model.stage1.head.post_process(
                        {'hm': out_s1['hm'], 'reg': out_s1['reg'], 'iou': out_s1['iou']},
                        use_soft_nms=soft_nms)
            else:
                out = model(batch)
                pred_list = model.head.post_process(
                    {'hm': out['hm'], 'reg': out['reg'], 'iou': out['iou']}, use_soft_nms=soft_nms)
        for bi, pd in enumerate(pred_list):
            all_preds.append({'boxes': pd['pred_boxes'].cpu().numpy(),
                              'scores': pd['pred_scores'].cpu().numpy(),
                              'labels': pd['pred_labels'].cpu().numpy()})
            all_gts.append(batch['labels'][bi])
    ap3d  = evaluate_kitti(all_preds, all_gts, iou_mode='3d')
    apbev = evaluate_kitti(all_preds, all_gts, iou_mode='bev')
    mAP_3D_mod = float(np.mean([ap3d[c]['Moderate'] for c in CLASS_NAMES]))
    return mAP_3D_mod, ap3d, apbev


def evaluate(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[eval] device = {dev}")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ds = KittiSGFDataset(args.kitti_root, args.val_split, training=False,
                         augment=False, keep_labels_for_eval=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_kitti, pin_memory=True)
    print(f"[eval] dataset size = {len(ds)}")
    ts = getattr(args, 'two_stage', False); sn = getattr(args, 'soft_nms', False)
    model = (SGFTwoStageModel() if ts else SGFDualGridModel()).to(dev)
    print(f"[eval] loading {args.ckpt}{' (TTA)' if args.tta else ''}"
          f"{' (two-stage)' if ts else ''}{' (soft-NMS)' if sn else ''}")
    ckpt = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ckpt['model'])
    if ckpt.get('ema_used'): print("[eval] checkpoint contains EMA-averaged weights")
    if getattr(args, 'ablate_density_gate', False):
        model.backbone.neck.dgate.ablate_density_gate = True
        print("[eval] ABLATION: density gate G_d ≡ 1 (identity on fused features)")
    if getattr(args, 'ablate_semantic_gate', False):
        print("[eval] NOTE: MidFusion has no G_s; use --ablate_density_gate")
    t0 = time.time()
    mAP_3D_mod, ap3d, apbev = _run_eval(model, loader, dev, tta=args.tta, soft_nms=sn, two_stage=ts)
    print(f"[eval] elapsed = {time.time()-t0:.1f}s")
    table = format_ap_table(ap3d, apbev)
    print("\n" + table + "\n")
    suffix = ('_2s' if ts else '') + ('_tta' if args.tta else '') + ('_snms' if sn else '')
    eval_path = out_dir / f'eval_results{suffix}.json'
    with open(eval_path, 'w') as f:
        json.dump({'AP3D': ap3d, 'APBEV': apbev, 'mAP3D_Mod': mAP_3D_mod,
                   'ckpt': args.ckpt, 'tta': args.tta, 'two_stage': ts, 'soft_nms': sn,
                   'val_split': args.val_split,
                   'ablate_density_gate': bool(getattr(args, 'ablate_density_gate', False))}, f, indent=2)
    with open(out_dir / f'eval_table{suffix}.txt', 'w') as f:
        f.write(table + "\n")
    print(f"[eval] saved {eval_path} and eval_table{suffix}.txt")


# ═══════════════════════════════════════════════════════════════════════════════
# §17  VISUALIZER + TEST EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def visualize(args):
    from sgf_viz_common import (frame_detection_score, gate_maps_to_numpy,
                                resolve_viz_subdir, select_viz_indices,
                                viz_semantic_gate_grid)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[viz] device = {dev}")
    viz_sub = resolve_viz_subdir(args)
    out_dir = Path(args.out_dir) / viz_sub; out_dir.mkdir(parents=True, exist_ok=True)
    ds = KittiSGFDataset(args.kitti_root, args.val_split, training=False,
                         augment=False, keep_labels_for_eval=True)
    n_out = min(args.num_viz, len(ds))
    viz_best = bool(getattr(args, 'viz_best', False))
    scan_n = len(ds) if viz_best else n_out
    if viz_best and getattr(args, 'viz_scan_max', 0) > 0:
        scan_n = min(scan_n, int(args.viz_scan_max))
    print(f"[viz] scan={scan_n}  save={n_out}  viz_best={viz_best}  "
          f"format=PNG  panel=semantic-6  subdir={viz_sub}")
    model = SGFDualGridModel(save_gates=True).to(dev)
    ckpt = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ckpt['model']); model.eval()
    ranked: List[Tuple[float, int]] = []
    for i in range(scan_n):
        sample = ds[i]
        batch = move_batch_to_device(collate_kitti([sample]), dev)
        out = model(batch)
        pred_list = model.head.post_process({'hm': out['hm'], 'reg': out['reg'], 'iou': out['iou']})
        pd = pred_list[0]
        pb = pd['pred_boxes'].cpu().numpy()
        ranked.append((frame_detection_score(sample['gt_boxes'], pb, rotated_3d_iou), i))
    pick = select_viz_indices(ranked, n_out, viz_best)
    score_by_idx = {idx: sc for sc, idx in ranked}
    if viz_best and pick:
        print("[viz] best frames: " + ", ".join(
            f"{ds[i]['sample_id']}({score_by_idx.get(i, 0.0):.3f})" for i in pick[:8]))
    for i in pick:
        sample = ds[i]
        batch = move_batch_to_device(collate_kitti([sample]), dev)
        out = model(batch)
        pred_list = model.head.post_process({'hm': out['hm'], 'reg': out['reg'], 'iou': out['iou']})
        pd = pred_list[0]
        pb = pd['pred_boxes'].cpu().numpy(); ps = pd['pred_scores'].cpu().numpy()
        pl = pd['pred_labels'].cpu().numpy()
        gt_boxes = sample['gt_boxes']; pts = sample['pts']; sid = sample['sample_id']
        ext = VIZ_IMAGE_EXT
        viz_bev_detection(pts, gt_boxes, pb, ps, pl,
            save_path=str(out_dir / f'{sid}_bev_detection{ext}'),
            title=f'KITTI {sid}  —  green: GT,  red: pred')
        viz_3d_detection(pts, gt_boxes, pb, ps, pl,
            save_path=str(out_dir / f'{sid}_3d_detection{ext}'),
            title=f'KITTI {sid}  —  3D boxes (green GT, red pred)')
        density_map = sample['density_map'][0]
        gate_arrays = gate_maps_to_numpy(model.backbone.neck.gate_maps)
        viz_semantic_gate_grid(
            density_map, gate_arrays,
            save_path=str(out_dir / f'{sid}_density_gates{ext}'),
            title=f'KITTI {sid}  —  MidFusion density gate heatmaps',
            pc_range=PC_RANGE, model_tag='MidFusion')
        if gate_arrays:
            viz_gate_histograms(gate_arrays,
                save_path=str(out_dir / f'{sid}_gate_histograms{ext}'))
        print(f"[viz] {sid}: score={score_by_idx.get(i, 0.0):.3f}  GT={len(gt_boxes)}  pred={len(pb)}")
    print(f"[viz] DONE. PNGs at {out_dir}")


@torch.no_grad()
def export_kitti_test(args) -> None:
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[export_test] device = {dev}")
    out_root = Path(args.out_dir) / 'kitti_test_submit'; data_dir = out_root / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    ds = KittiSGFDataset(args.kitti_root, args.test_split, training=False,
                         augment=False, keep_labels_for_eval=False, data_subset='testing')
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_kitti,
                        pin_memory=True, drop_last=False)
    ts = getattr(args, 'two_stage', False); sn = getattr(args, 'soft_nms', False)
    model = (SGFTwoStageModel() if ts else SGFDualGridModel()).to(dev)
    print(f"[export_test] loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ckpt['model']); model.eval()
    kroot = Path(args.kitti_root); n_written = 0
    for batch in loader:
        batch = move_batch_to_device(batch, dev)
        with torch.no_grad():
            if args.tta:
                pred_list = tta_infer(model.stage1 if ts else model, batch)
            elif ts:
                out = model(batch)
                if 'stage2_refined' in out:
                    pred_list = []
                    for bi in range(batch['batch_size']):
                        mask = out['stage2_refined'][:, 0].long() == bi
                        if mask.any():
                            r = out['stage2_refined'][mask, 1:]; s = out['stage2_scores'][mask]
                            l = out['stage1_preds'][bi]['pred_labels'][:mask.sum()]
                            pred_list.append({'pred_boxes': r, 'pred_scores': s, 'pred_labels': l})
                        else:
                            pred_list.append({'pred_boxes': torch.zeros(0,7,device=dev),
                                              'pred_scores': torch.zeros(0,device=dev),
                                              'pred_labels': torch.zeros(0,dtype=torch.long,device=dev)})
                else:
                    out_s1 = model.stage1(batch)
                    pred_list = model.stage1.head.post_process(
                        {'hm': out_s1['hm'], 'reg': out_s1['reg'], 'iou': out_s1['iou']}, use_soft_nms=sn)
            else:
                out = model(batch)
                pred_list = model.head.post_process(
                    {'hm': out['hm'], 'reg': out['reg'], 'iou': out['iou']}, use_soft_nms=sn)
        for bi, sid in enumerate(batch['sample_ids']):
            pd = pred_list[bi]
            pb = pd['pred_boxes'].detach().cpu().numpy()
            ps = pd['pred_scores'].detach().cpu().numpy()
            pl = pd['pred_labels'].detach().cpu().numpy().astype(int)
            calib = parse_calib(str(kroot / 'testing' / 'calib' / f'{sid}.txt'))
            lines: List[str] = []
            for j in range(pb.shape[0]):
                lb = int(pl[j])
                if lb < 1 or lb > len(CLASS_NAMES): continue
                lines.append(velo_box_to_kitti_line(pb[j], CLASS_NAMES[lb - 1], float(ps[j]), calib))
            with open(data_dir / f'{sid}.txt', 'w') as f:
                f.write('\n'.join(lines) + ('\n' if lines else ''))
            n_written += 1
    with open(out_root / 'export_manifest.json', 'w') as f:
        json.dump({'frames': n_written, 'ckpt': args.ckpt, 'test_split': args.test_split,
                   'submit_data_dir': str(data_dir)}, f, indent=2)
    print(f"[export_test] DONE. Wrote {n_written} files under {data_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# §18  SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════════

def smoke_test(args):
    print("=" * 70); print("SGF-DualGrid (BCAF-DH) SMOKE TEST"); print("=" * 70)
    pts = load_points_bin(args.lidar)
    print(f"[1] points: {pts.shape}")
    if args.calib and os.path.exists(args.calib):
        calib = parse_calib(args.calib)
    else:
        calib = {'V2C': np.eye(4, np.float32), 'C2V': np.eye(4, np.float32),
                 'P2': np.zeros((3, 4), np.float32)}
        print("    no calib — using identity")
    labels = parse_label(args.label, calib) if (args.label and os.path.exists(args.label)) else []
    gt = gt_array_for_training(labels)
    print(f"    labels parsed: {len(labels)} ; detection-class GTs: {len(gt)}")

    voxels, coords, num = voxelize_pillars(pts, max_voxels=MAX_VOXELS_TRAIN)
    coarse  = make_coarse_bev(pts); density = make_density_map(pts)
    hm_t, reg_t, iou_t, pos_t = build_targets(gt, NUM_CLASSES, F_NY, F_NX)
    print(f"[2] voxelization: V={voxels.shape[0]} mean_pts/pillar={num.mean() if len(num) else 0:.2f}")
    print(f"    coarse_bev={coarse.shape} density={density.shape}")
    print(f"    hm_t pos count = {int((pos_t>0).sum())}")

    coords_b = coords.copy(); coords_b[:, 0] = 0
    batch = {'voxels': torch.from_numpy(voxels), 'coords': torch.from_numpy(coords_b).long(),
             'num_per_v': torch.from_numpy(num).long(),
             'coarse_bev': torch.from_numpy(coarse).unsqueeze(0),
             'density_map': torch.from_numpy(density).unsqueeze(0),
             'hm_t': torch.from_numpy(hm_t).unsqueeze(0), 'reg_t': torch.from_numpy(reg_t).unsqueeze(0),
             'iou_t': torch.from_numpy(iou_t).unsqueeze(0), 'pos_t': torch.from_numpy(pos_t).unsqueeze(0),
             'gt_boxes': [gt], 'batch_size': 1, 'sample_ids': ['smoke']}
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("    CUDA requested but unavailable — using CPU"); args.device = 'cpu'
    dev = torch.device(args.device); batch = move_batch_to_device(batch, dev)

    model = SGFDualGridModel().to(dev)
    print(f"[3] params total={model.count_params()['total']:,}")
    model.train()
    t0 = time.time(); out = model(batch)
    if dev.type == 'cuda': torch.cuda.synchronize()
    print(f"[4] forward time : {time.time()-t0:.3f}s  feats={tuple(out['feats'].shape)}")
    print(f"    gate_log     : {out['gate_log']}")
    loss, tb = model.head.compute_loss(
        {'hm': out['hm'], 'reg': out['reg'], 'iou': out['iou']},
        batch['hm_t'], batch['reg_t'], batch['iou_t'], batch['pos_t'],
        gate_log=out['gate_log'], aux_dg_logit=out.get('dg_logit'))
    print(f"[5] loss components: {tb}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    opt.zero_grad(); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
    print(f"    backward grad-norm = {float(gn):.4f}, step OK")
    model.eval()
    pred_list = model.head.post_process({'hm': out['hm'], 'reg': out['reg'], 'iou': out['iou']})
    print(f"[6] inference + rotated NMS: {pred_list[0]['pred_boxes'].shape[0]} boxes")
    print("=" * 70); print("SMOKE TEST PASSED"); print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# §19  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="SGF-DualGrid (BCAF-DH) production runner")
    p.add_argument('--mode', choices=['smoke', 'train', 'eval', 'viz', 'build_db', 'export_test'], required=True)
    p.add_argument('--lidar', type=str, default=None)
    p.add_argument('--label', type=str, default=None)
    p.add_argument('--calib', type=str, default=None)
    p.add_argument('--kitti_root',  type=str, default=None)
    p.add_argument('--train_split', type=str, default=None)
    p.add_argument('--val_split',   type=str, default=None)
    p.add_argument('--test_split',  type=str, default=None)
    p.add_argument('--batch_size',  type=int, default=4)
    p.add_argument('--num_workers', type=int, default=6)
    p.add_argument('--epochs',      type=int, default=80)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--amp',         action='store_true')
    p.add_argument('--log_every',   type=int, default=20)
    p.add_argument('--val_interval', type=int, default=2)
    p.add_argument('--resume',      type=str, default=None)
    p.add_argument('--ckpt',     type=str, default=None)
    p.add_argument('--num_viz',  type=int, default=12)
    p.add_argument('--viz_best', action='store_true',
                   help='Visualize val frames with highest GT–pred 3D IoU match quality')
    p.add_argument('--viz_subdir', type=str, default='viz_semantic',
                   help='Subfolder under out_dir for visualization PNGs')
    p.add_argument('--viz_scan_max', type=int, default=0,
                   help='Max val frames to scan for --viz_best (0 = full val split)')
    p.add_argument('--gt_db_path',  type=str, default=None)
    p.add_argument('--gt_sample_car', type=int, default=15)
    p.add_argument('--gt_sample_ped', type=int, default=10)
    p.add_argument('--gt_sample_cyc', type=int, default=10)
    p.add_argument('--ema',         action='store_true')
    p.add_argument('--ema_decay',   type=float, default=0.999)
    p.add_argument('--tta',         action='store_true')
    p.add_argument('--cls_balanced', action='store_true')
    p.add_argument('--two_stage',   action='store_true')
    p.add_argument('--soft_nms',    action='store_true')
    p.add_argument('--ablate_density_gate', action='store_true',
                   help='Eval ablation: force G_d ≡ 1 (density gate disabled)')
    p.add_argument('--ablate_semantic_gate', action='store_true',
                   help='No-op on MidFusion (no G_s); kept for CLI parity')
    p.add_argument('--out_dir',  type=str, default='./runs/sgfd_bcaf_dh')
    p.add_argument('--device',   type=str, default='cuda')
    p.add_argument('--seed',     type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    if args.mode == 'smoke':
        if not args.lidar: print("ERROR: --lidar required", file=sys.stderr); sys.exit(2)
        smoke_test(args)
    elif args.mode == 'train':
        if not args.kitti_root or not args.train_split:
            print("ERROR: --kitti_root and --train_split required", file=sys.stderr); sys.exit(2)
        train(args)
    elif args.mode == 'eval':
        if not (args.kitti_root and args.val_split and args.ckpt):
            print("ERROR: --kitti_root --val_split --ckpt required", file=sys.stderr); sys.exit(2)
        evaluate(args)
    elif args.mode == 'viz':
        if not (args.kitti_root and args.val_split and args.ckpt):
            print("ERROR: --kitti_root --val_split --ckpt required", file=sys.stderr); sys.exit(2)
        visualize(args)
    elif args.mode == 'build_db':
        if not (args.kitti_root and args.train_split):
            print("ERROR: --kitti_root --train_split required", file=sys.stderr); sys.exit(2)
        build_gt_db(args)
    elif args.mode == 'export_test':
        if not (args.kitti_root and args.test_split and args.ckpt and args.out_dir):
            print("ERROR: --kitti_root --test_split --ckpt --out_dir required", file=sys.stderr); sys.exit(2)
        export_kitti_test(args)


if __name__ == '__main__':
    main()