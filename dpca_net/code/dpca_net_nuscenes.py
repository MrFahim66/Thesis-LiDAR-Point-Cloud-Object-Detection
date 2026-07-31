"""
SGF-DualGrid MidFusion (BCAF-DH) for nuScenes — 10-class LiDAR 3D detection.

MODEL: identical to modelv4_midFusion.py (best KITTI variant, 80.79 mAP3D@Mod).
  - BCAFFPNNeck: bidirectional cross-stream attention + DensityGate
  - Auxiliary BCE density-gate supervision
  - D1/D2/D3 height-fixed SGFCenterHead

DATA: nuScenes official protocol (10 detection classes, ego-frame LiDAR).
  - JSON tables + quaternion box transforms
  - Cross-scene train/val splits supported
  - 40-point AP evaluator with nuScenes difficulty (num_lidar_pts)
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

_PORT_CODE = Path('/home/frahman8/scratch/SGF_PortV2/code')
if str(_PORT_CODE) not in sys.path:
    sys.path.insert(0, str(_PORT_CODE))
from sgf_port_data import normalize_point_intensity, resolve_torch_device, HM_BIAS_INIT  # noqa: E402
from dataset_configs import nuscenes as ds_cfg  # noqa: E402
from dataset_configs._common import expand_per_class  # noqa: E402
from sgf_opcd_protocol import opcd_augment, NUSC_ROT_RANGE, make_nuscenes_cbgs_sampler  # noqa: E402
from sgf_eval_official import nusc_gt_eval_fields  # noqa: E402
from sgf_gtdb import make_gt_sampler  # noqa: E402


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
# §1  CONFIG  (from dataset_configs.nuscenes)
# ═══════════════════════════════════════════════════════════════════════════════

FINE_VS    = ds_cfg.FINE_VS
COARSE_VS  = ds_cfg.COARSE_VS
PC_RANGE   = ds_cfg.PC_RANGE
REG_ENCODING = ds_cfg.REG_ENCODING
MAX_PTS    = ds_cfg.MAX_PTS
NUM_CLASSES = ds_cfg.NUM_CLASSES
CLASS_NAMES = ds_cfg.CLASS_NAMES
CLASS_TO_ID = ds_cfg.CLASS_TO_ID
F_NX = ds_cfg.F_NX
F_NY = ds_cfg.F_NY
C_NX = ds_cfg.C_NX
C_NY = ds_cfg.C_NY
DEFAULT_NUM_SWEEPS = ds_cfg.DEFAULT_NUM_SWEEPS
MAX_VOXELS_TRAIN = ds_cfg.MAX_VOXELS_TRAIN
MAX_VOXELS_EVAL  = ds_cfg.MAX_VOXELS_EVAL
NUSC_DIFFICULTY_LIMITS = ds_cfg.NUSC_DIFFICULTY_LIMITS
IOU_THRESH_3D  = ds_cfg.IOU_THRESH_3D
IOU_THRESH_BEV = ds_cfg.IOU_THRESH_BEV
FOCAL_ALPHA = ds_cfg.FOCAL_ALPHA
NUSC_CAT_MAP = ds_cfg.NUSC_CAT_MAP
CLASS_ANCHORS = ds_cfg.CLASS_ANCHORS
_SMALL_CLS_IDX = ds_cfg._SMALL_CLS_IDX
IOU_ALPHA_DEFAULT = ds_cfg.IOU_ALPHA_DEFAULT
EVAL_SCORE_THRESH_DEFAULT = ds_cfg.EVAL_SCORE_THRESH_DEFAULT


def _per_class_thresh(val, n: int = NUM_CLASSES, default: float = 0.01):
    if isinstance(val, (int, float)):
        return tuple([float(val)] * n)
    t = tuple(val)
    if len(t) < n:
        return t + tuple([t[-1]] * (n - len(t)))
    return t[:n]

# ═══════════════════════════════════════════════════════════════════════════════
# §2  NUSCENES JSON PARSING + QUATERNION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """q = [w, x, y, z] → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1;  w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def _quat_to_yaw(q: np.ndarray) -> float:
    """Yaw (rotation around z) from [w,x,y,z] quaternion."""
    w, x, y, z = q
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


from functools import lru_cache


@lru_cache(maxsize=4)
def load_nusc_tables(nusc_root: str, version: str = 'v1.0-mini') -> Dict[str, Any]:
    """Load all nuScenes JSON tables into lookup dicts (cached — shared across datasets)."""
    print(f"[dataset] parsing nuScenes JSON tables ({version})…", flush=True)
    vdir = Path(nusc_root) / version
    tables: Dict[str, Any] = {}
    for name in ('sample', 'sample_data', 'sample_annotation',
                 'ego_pose', 'calibrated_sensor', 'sensor',
                 'instance', 'category', 'scene', 'log', 'attribute'):
        with open(vdir / f'{name}.json') as f:
            tables[name] = json.load(f)

    tables['ego_pose_by_token']          = {r['token']: r for r in tables['ego_pose']}
    tables['calib_sensor_by_token']      = {r['token']: r for r in tables['calibrated_sensor']}
    tables['category_by_token']          = {r['token']: r['name'] for r in tables['category']}
    tables['attribute_by_token']       = {r['token']: r['name'] for r in tables['attribute']}
    tables['instance_by_token']          = {r['token']: r for r in tables['instance']}
    tables['sensor_by_token']            = {r['token']: r for r in tables['sensor']}
    tables['sd_by_token']                = {sd['token']: sd for sd in tables['sample_data']}

    # Index sample_data by sample_token → {channel: record}
    # IMPORTANT: only keyframe entries (is_key_frame=True) and only LIDAR_TOP.
    # Sweep entries also carry sample_token of the nearest keyframe and would
    # overwrite the correct samples/LIDAR_TOP/... path with sweeps/LIDAR_TOP/...
    sd_by_sample: Dict[str, Dict[str, Any]] = {}
    for sd in tables['sample_data']:
        if not sd.get('is_key_frame', False):
            continue          # skip sweeps — we only need keyframe LiDAR
        calib = tables['calib_sensor_by_token'].get(sd['calibrated_sensor_token'], {})
        sensor = tables['sensor_by_token'].get(calib.get('sensor_token', ''), {})
        channel = sensor.get('channel', '')
        if channel != 'LIDAR_TOP':
            continue          # skip camera / RADAR sensors — LiDAR only
        st = sd['sample_token']
        sd_by_sample.setdefault(st, {})['LIDAR_TOP'] = sd
    tables['sd_by_sample'] = sd_by_sample

    # Index annotations by sample_token
    anns_by_sample: Dict[str, List[Any]] = {}
    for ann in tables['sample_annotation']:
        st = ann['sample_token']
        anns_by_sample.setdefault(st, []).append(ann)
    tables['anns_by_sample'] = anns_by_sample

    # Resolve category name for each annotation (via instance → category)
    for ann in tables['sample_annotation']:
        inst = tables['instance_by_token'].get(ann['instance_token'], {})
        cat_tok = inst.get('category_token', '')
        ann['category_name'] = tables['category_by_token'].get(cat_tok, '')

    return tables


def load_lidar_bin(filepath: str) -> np.ndarray:
    """Load nuScenes LiDAR .pcd.bin → (N,4) [x,y,z,intensity] in sensor frame."""
    pts = np.fromfile(filepath, dtype=np.float32).reshape(-1, 5)
    return pts[:, :4]   # drop ring_index; keep x,y,z,intensity


def transform_pts_lidar_to_ego(pts_lidar: np.ndarray,
                                calib_sensor: Dict) -> np.ndarray:
    """LiDAR sensor frame → ego vehicle frame."""
    t = np.array(calib_sensor['translation'], dtype=np.float64)
    R = _quat_to_rotmat(np.array(calib_sensor['rotation'], dtype=np.float64))
    xyz = pts_lidar[:, :3].astype(np.float64) @ R.T + t
    out = np.concatenate([xyz.astype(np.float32), pts_lidar[:, 3:4]], axis=1)
    return out


def _lidar_to_ref_ego(pts_lidar: np.ndarray,
                      calib_src: Dict,
                      ego_src: Dict,
                      ego_ref: Dict) -> np.ndarray:
    """Transform LiDAR points from one sweep into the reference keyframe ego frame."""
    pts_ego_src = transform_pts_lidar_to_ego(pts_lidar, calib_src)
    R_src = _quat_to_rotmat(np.array(ego_src['rotation'], dtype=np.float64))
    t_src = np.array(ego_src['translation'], dtype=np.float64)
    xyz_g = pts_ego_src[:, :3] @ R_src.T + t_src
    R_ref = _quat_to_rotmat(np.array(ego_ref['rotation'], dtype=np.float64))
    t_ref = np.array(ego_ref['translation'], dtype=np.float64)
    xyz_ref = (xyz_g - t_ref) @ R_ref
    return np.concatenate([xyz_ref.astype(np.float32), pts_lidar[:, 3:4]], axis=1)


def _filter_pc_range(pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] == 0:
        return pts
    m = ((pts[:, 0] >= PC_RANGE[0]) & (pts[:, 0] < PC_RANGE[3]) &
         (pts[:, 1] >= PC_RANGE[1]) & (pts[:, 1] < PC_RANGE[4]) &
         (pts[:, 2] >= PC_RANGE[2]) & (pts[:, 2] < PC_RANGE[5]))
    return pts[m]


def load_lidar_multisweep(sd_keyframe: Dict,
                        tables: Dict[str, Any],
                        nusc_root: str,
                        num_sweeps: int = DEFAULT_NUM_SWEEPS) -> np.ndarray:
    """
    Load keyframe + (num_sweeps-1) previous LIDAR_TOP sweeps, fused in keyframe ego.
    Matches CenterPoint / MMDetection3D nuScenes training convention (~0.5 s window).
    """
    ego_ref = tables['ego_pose_by_token'][sd_keyframe['ego_pose_token']]
    sd_by_token = tables['sd_by_token']
    chunks: List[np.ndarray] = []
    sd = sd_keyframe
    for _ in range(max(1, num_sweeps)):
        lidar_path = Path(nusc_root) / sd['filename']
        pts = load_lidar_bin(str(lidar_path))
        calib = tables['calib_sensor_by_token'][sd['calibrated_sensor_token']]
        ego = tables['ego_pose_by_token'][sd['ego_pose_token']]
        pts_ref = _lidar_to_ref_ego(pts, calib, ego, ego_ref)
        chunks.append(_filter_pc_range(pts_ref))
        prev = sd.get('prev', '')
        if not prev or prev not in sd_by_token:
            break
        sd = sd_by_token[prev]
    if not chunks:
        return np.zeros((0, 4), np.float32)
    return np.concatenate(chunks, axis=0)


def ann_to_ego_box(ann: Dict, ego_pose: Dict,
                   calib_sensor: Dict) -> Optional[np.ndarray]:
    """
    Convert one annotation to a box in ego frame: [x,y,z, l,w,h, ry, cls_id].
    Returns None if category is not in NUSC_CAT_MAP.
    """
    cls_name = NUSC_CAT_MAP.get(ann['category_name'])
    if cls_name is None:
        return None

    # Transform center: global → ego
    t_ego = np.array(ego_pose['translation'], dtype=np.float64)
    q_ego = np.array(ego_pose['rotation'],    dtype=np.float64)
    R_ego = _quat_to_rotmat(q_ego)
    center_g = np.array(ann['translation'],    dtype=np.float64)
    center_e = R_ego.T @ (center_g - t_ego)

    # Yaw in ego frame
    q_box_g = np.array(ann['rotation'],        dtype=np.float64)
    q_box_e = _quat_mul(_quat_conj(q_ego), q_box_g)
    yaw_e   = _quat_to_yaw(q_box_e)

    # nuScenes size: [width, length, height]
    w, l, h = ann['size']
    cls_id  = CLASS_TO_ID[cls_name]

    box = np.array([center_e[0], center_e[1], center_e[2],
                    float(l), float(w), float(h),
                    yaw_e, float(cls_id)], dtype=np.float32)
    return box


def nusc_difficulty(num_lidar_pts: int) -> Optional[str]:
    if num_lidar_pts >= NUSC_DIFFICULTY_LIMITS['Easy']:     return 'Easy'
    if num_lidar_pts >= NUSC_DIFFICULTY_LIMITS['Moderate']: return 'Moderate'
    if num_lidar_pts >= NUSC_DIFFICULTY_LIMITS['Hard']:     return 'Hard'
    return None   # skip (< 1 pts — invisible)


def load_nusc_sample(token: str, nusc_root: str,
                     tables: Dict[str, Any],
                     keep_labels: bool = False,
                     num_sweeps: int = DEFAULT_NUM_SWEEPS,
                     ) -> Dict[str, Any]:
    """Load one nuScenes sample → dict with pts (ego), gt_boxes, labels."""
    # Find LIDAR_TOP sample_data
    sd_map = tables['sd_by_sample'].get(token, {})
    sd_lidar = sd_map.get('LIDAR_TOP')
    if sd_lidar is None:
        raise ValueError(f"No LIDAR_TOP for sample {token}")

    ego_pose     = tables['ego_pose_by_token'][sd_lidar['ego_pose_token']]
    calib_sensor = tables['calib_sensor_by_token'][sd_lidar['calibrated_sensor_token']]

    if num_sweeps <= 1:
        lidar_path = Path(nusc_root) / sd_lidar['filename']
        pts_lidar = load_lidar_bin(str(lidar_path))
        pts_ego = _filter_pc_range(transform_pts_lidar_to_ego(pts_lidar, calib_sensor))
    else:
        pts_ego = load_lidar_multisweep(sd_lidar, tables, nusc_root, num_sweeps)

    annotations = tables['anns_by_sample'].get(token, [])
    boxes, labels = [], []
    for ann in annotations:
        box = ann_to_ego_box(ann, ego_pose, calib_sensor)
        if box is None:
            continue
        boxes.append(box)
        if keep_labels:
            diff = nusc_difficulty(ann.get('num_lidar_pts', 0))
            if diff is None:
                diff = 'Hard'
            cls_name = NUSC_CAT_MAP[ann['category_name']]
            meta = nusc_gt_eval_fields(ann, ego_pose, tables['attribute_by_token'])
            labels.append({
                'cls_name':      cls_name,
                'difficulty':    diff,
                'box':           box[:7].tolist(),
                'num_lidar_pts': ann.get('num_lidar_pts', 0),
                'velocity':      meta['velocity'].tolist(),
                'attribute':     meta['attribute'],
                'bbox':          [0, 0, 1, 1],
                'truncated':     0.0,
                'occluded':      0,
                'alpha':         0.0,
            })

    gt = np.stack(boxes, 0).astype(np.float32) if boxes else np.zeros((0, 8), np.float32)
    out: Dict[str, Any] = {'pts': pts_ego, 'gt_boxes': gt,
                           'sample_id': token[:8], 'sample_token': token}
    if keep_labels:
        out['labels'] = labels
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# §3  DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def augment(pts: np.ndarray, boxes: np.ndarray,
            flip_prob: float = 0.5,
            rot_range: float = NUSC_ROT_RANGE,
            scale_range: Tuple[float, float] = (0.95, 1.05)
            ) -> Tuple[np.ndarray, np.ndarray]:
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

def voxelize_pillars(pts: np.ndarray, max_voxels: int = MAX_VOXELS_TRAIN,
                     max_pts: int = MAX_PTS
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        sel  = np.zeros(V, dtype=bool); sel[keep] = True
        valid_pt = sel[inv]
        pts  = pts[valid_pt]; inv = inv[valid_pt]
        old2new = -np.ones(V, dtype=np.int32)
        old2new[keep] = np.arange(len(keep), dtype=np.int32)
        inv  = old2new[inv]; uniq = uniq[keep]; cnt = cnt[keep]; V = len(uniq)

    voxels    = np.zeros((V, max_pts, 4), np.float32)
    num_per_v = np.zeros((V,), np.int32)
    order     = np.argsort(inv, kind='stable')
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
    coords[:, 1] = uniq // F_NX
    coords[:, 2] = uniq %  F_NX
    return voxels.astype(np.float32), coords, num_per_v


# ═══════════════════════════════════════════════════════════════════════════════
# §5  COARSE BEV + DENSITY MAP
# ═══════════════════════════════════════════════════════════════════════════════

def make_coarse_bev(pts: np.ndarray) -> np.ndarray:
    pc = PC_RANGE
    x, y, z, i_ = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    ok = ((x >= pc[0]) & (x < pc[3]) & (y >= pc[1]) & (y < pc[4]) &
          (z >= pc[2]) & (z < pc[5]))
    x, y, z, i_ = x[ok], y[ok], z[ok], i_[ok]
    if len(x) == 0:
        return np.zeros((4, C_NY, C_NX), np.float32)
    xi = np.floor((x - pc[0]) / COARSE_VS).astype(np.int32).clip(0, C_NX - 1)
    yi = np.floor((y - pc[1]) / COARSE_VS).astype(np.int32).clip(0, C_NY - 1)
    flt = yi * C_NX + xi
    z_n = ((z - pc[2]) / (pc[5] - pc[2])).clip(0, 1)
    nc = C_NY * C_NX
    cnt  = np.bincount(flt, minlength=nc).astype(np.float32)
    s_z  = np.bincount(flt, weights=z_n.astype(np.float64), minlength=nc).astype(np.float32)
    s_i  = np.bincount(flt, weights=i_.astype(np.float64),  minlength=nc).astype(np.float32)
    mx   = np.zeros(nc, np.float32); np.maximum.at(mx, flt, z_n.astype(np.float32))
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
    ld  = np.clip(np.log1p(cnt) / math.log1p(MAX_PTS), 0.0, 1.0)
    return ld.reshape(1, F_NY, F_NX)


# ═══════════════════════════════════════════════════════════════════════════════
# §6  PILLAR FEATURE ENCODER + PILLAR SCATTER
# ═══════════════════════════════════════════════════════════════════════════════

class PillarVFE(nn.Module):
    """Dynamic Pillar Feature Encoding with max+mean dual pooling. 10-dim input."""
    def __init__(self, out_ch: int = 64):
        super().__init__()
        in_dim = 10; half = out_ch // 2
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
                self.gate_maps = {
                    'den_p2': den_p1,
                    'G_d': G_p1,
                    'G': Gm,
                    'dg_logit': dg_p1,
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
# §8  HEATMAP TARGETS + METRIC ANCHOR-RELATIVE ENCODING (nuScenes anchors)
# ═══════════════════════════════════════════════════════════════════════════════

# nuScenes class anchors loaded from dataset_configs.nuscenes (computed from train GT)


def _gaussian_radius(dh, dw, min_overlap=0.1):
    a1=1; b1=dh+dw; c1=dh*dw*(1-min_overlap)/(1+min_overlap)
    sq1=math.sqrt(max(b1*b1-4*a1*c1,0)); r1=(b1-sq1)/2
    a2=4; b2=2*(dh+dw); c2=(1-min_overlap)*dh*dw
    sq2=math.sqrt(max(b2*b2-4*a2*c2,0)); r2=(b2-sq2)/2
    a3=4*min_overlap; b3=-2*min_overlap*(dh+dw); c3=(min_overlap-1)*dh*dw
    sq3=math.sqrt(max(b3*b3-4*a3*c3,0)); r3=(b3+sq3)/2
    return max(1, int(min(r1,r2,r3)))


def _draw_gaussian(hm, cx, cy, r):
    d=2*r+1; s=d/6; m=r
    y_,x_=np.ogrid[-m:m+1,-m:m+1]
    g=np.exp(-(x_*x_+y_*y_)/(2*s*s)).astype(np.float32)
    g[g < np.finfo(g.dtype).eps*g.max()] = 0
    H,W=hm.shape
    l=min(cx,r); rb=min(W-cx,r+1); t=min(cy,r); b=min(H-cy,r+1)
    if min(rb-(-l), b-(-t)) > 0:
        np.maximum(hm[cy-t:cy+b, cx-l:cx+rb], g[r-t:r+b, r-l:r+rb],
                   out=hm[cy-t:cy+b, cx-l:cx+rb])


def build_targets(gt_boxes: np.ndarray, num_class: int = NUM_CLASSES,
                  H: int = F_NY, W: int = F_NX
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vs = FINE_VS; x0, y0 = PC_RANGE[0], PC_RANGE[1]
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
        cell_cx = (cxi + 0.5) * vs + x0; cell_cy = (cyi + 0.5) * vs + y0
        min_ov = 0.1 if ci in _SMALL_CLS_IDX else 0.25
        r = _gaussian_radius(max(1., l/vs), max(1., w/vs), min_overlap=min_ov)
        _draw_gaussian(hm[ci], cxi, cyi, r)
        reg[:, cyi, cxi] = [
            x - cell_cx, y - cell_cy,
            z - a['z'],
            math.log(max(h, 1e-3)) - math.log(a['h']),
            math.log(max(w, 1e-3)) - math.log(a['w']),
            math.log(max(l, 1e-3)) - math.log(a['l']),
            math.sin(ry), math.cos(ry)]
        iou[0, cyi, cxi] = 1.0
        pos[0, cyi, cxi] = 1.0
    return hm, reg, iou, pos


def decode_class_boxes(reg_ci: torch.Tensor, xi: torch.Tensor,
                       yi: torch.Tensor, ci: int) -> torch.Tensor:
    vs = FINE_VS; x0, y0 = PC_RANGE[0], PC_RANGE[1]
    a = CLASS_ANCHORS[ci]
    cell_cx = (xi.float() + 0.5) * vs + x0
    cell_cy = (yi.float() + 0.5) * vs + y0
    x = cell_cx + reg_ci[0]; y = cell_cy + reg_ci[1]
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
                 score_thresh: Sequence[float] = None,
                 nms_iou_thresh: Sequence[float] = None):
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
        st = score_thresh if score_thresh is not None else ds_cfg.SCORE_THRESH
        nt = nms_iou_thresh if nms_iou_thresh is not None else ds_cfg.NMS_IOU
        self.score_thresh = tuple(expand_per_class(st, num_class))
        self.nms_iou      = tuple(expand_per_class(nt, num_class))

        self.shared = nn.Sequential(cbr(input_channels, 128), cbr(128, 64))
        self.hm  = nn.Conv2d(64, num_class, 1)
        nn.init.constant_(self.hm.bias, HM_BIAS_INIT)
        self.reg = nn.ModuleList([nn.Conv2d(64, 8, 1) for _ in range(num_class)])
        self.iou = nn.ModuleList([nn.Conv2d(64, 1, 1) for _ in range(num_class)])
        for ci in range(num_class):
            nn.init.zeros_(self.reg[ci].bias)            # residual targets -> 0
            with torch.no_grad():
                self.reg[ci].bias.data[7] = 1.0          # cos -> ry≈0 at init
            nn.init.constant_(self.iou[ci].bias, HM_BIAS_INIT)

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
                     iou_alpha: float = IOU_ALPHA_DEFAULT) -> List[Dict[str, torch.Tensor]]:
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
                 save_gates: bool = False, score_thresh=None,
                 nms_iou_thresh=None):
        super().__init__()
        self.vfe      = PillarVFE(out_ch=vfe_ch)
        self.scatter  = PillarScatter(ch=vfe_ch, ny=F_NY, nx=F_NX)
        self.backbone = SGFDualGridBackbone(in_ch=vfe_ch, neck_ch=neck_ch,
                                            out_ch=head_ch, save_gates=save_gates)
        kw = {}
        if score_thresh is not None:
            kw['score_thresh'] = score_thresh
        if nms_iou_thresh is not None:
            kw['nms_iou_thresh'] = nms_iou_thresh
        self.head     = SGFCenterHead(input_channels=head_ch, num_class=num_class, **kw)

    def forward(self, batch: Dict) -> Dict:
        with amp_autocast(False):
            pillar_feats = self.vfe(batch['voxels'].float(), batch['num_per_v'])
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
# §12  AP EVALUATOR  (40-point, nuScenes difficulty by num_lidar_pts)
# ═══════════════════════════════════════════════════════════════════════════════

def _gt_subset_by_difficulty(gts, cls_name, diff):
    diff_rank = {'Easy': 0, 'Moderate': 1, 'Hard': 2, 'Unknown': 99}
    rank_now = diff_rank[diff]; valid, ignore = [], []
    for g in gts:
        if g['cls_name'] != cls_name: continue
        gr = diff_rank.get(g['difficulty'], 99)
        if gr <= rank_now: valid.append(g)
        else: ignore.append(g)
    return valid, ignore


def _compute_ap_40point(rec, prec):
    if len(rec) == 0: return 0.0
    recall_thresh = np.linspace(1/40, 1.0, 40); ap = 0.0
    for r in recall_thresh:
        mask = rec >= r
        if mask.any(): ap += float(prec[mask].max())
    return ap / 40.0 * 100.0


def evaluate_nusc(all_preds, all_gts, iou_mode='3d'):
    iou_fn = rotated_3d_iou if iou_mode == '3d' else rotated_bev_iou
    iou_thr = IOU_THRESH_3D if iou_mode == '3d' else IOU_THRESH_BEV
    results: Dict[str, Dict[str, float]] = {}
    for ci, cls in enumerate(CLASS_NAMES):
        thr = iou_thr[cls]; results[cls] = {}
        for diff in ('Easy', 'Moderate', 'Hard'):
            tp_list, fp_list, scores_list, n_gt = [], [], [], 0
            for pred, gts in zip(all_preds, all_gts):
                valid_gt, ignore_gt = _gt_subset_by_difficulty(gts, cls, diff)
                n_gt += len(valid_gt)
                pd = pred
                if len(pd['pred_boxes']) == 0: continue
                mask = pd['pred_labels'] == (ci + 1)
                if not mask.any(): continue
                pb = pd['pred_boxes'][mask].numpy()
                ps = pd['pred_scores'][mask].numpy()
                matched = np.zeros(len(valid_gt), dtype=bool)
                for bi in np.argsort(-ps):
                    best_iou = 0; best_j = -1
                    for j, g in enumerate(valid_gt):
                        if matched[j]: continue
                        iou = iou_fn(pb[bi], np.array(g['box']))
                        if iou > best_iou: best_iou = iou; best_j = j
                    in_ignore = any(iou_fn(pb[bi], np.array(g['box'])) >= thr for g in ignore_gt)
                    if best_iou >= thr and best_j >= 0:
                        tp_list.append(1); fp_list.append(0)
                        matched[best_j] = True
                    elif in_ignore:
                        continue
                    else:
                        tp_list.append(0); fp_list.append(1)
                    scores_list.append(float(ps[bi]))
            if n_gt == 0: results[cls][diff] = 0.0; continue
            if not scores_list: results[cls][diff] = 0.0; continue
            order = np.argsort(-np.array(scores_list))
            tp_c = np.cumsum(np.array(tp_list)[order])
            fp_c = np.cumsum(np.array(fp_list)[order])
            rec  = tp_c / max(n_gt, 1)
            prec = tp_c / np.maximum(tp_c + fp_c, 1)
            results[cls][diff] = _compute_ap_40point(rec, prec)
    return results


def print_ap_table(results: Dict[str, Dict[str, float]]):
    header = f"{'Class':<14} {'Easy':>8} {'Moderate':>10} {'Hard':>8}"
    print(header); print('-' * len(header))
    mAPs = []
    for cls in CLASS_NAMES:
        r = results.get(cls, {})
        e = r.get('Easy', 0); m = r.get('Moderate', 0); h = r.get('Hard', 0)
        print(f"{cls:<14} {e:8.2f} {m:10.2f} {h:8.2f}")
        mAPs.append(m)
    print('-' * len(header))
    print(f"{'mAP@Mod':<14} {''*8} {np.mean(mAPs):10.2f}")
    return float(np.mean(mAPs))


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

class NuscSGFDataset(Dataset):
    """nuScenes dataset loader for SGF-DualGrid MidFusion."""
    def __init__(self, nusc_root: str, split_file: str, training: bool,
                 version: str = 'v1.0-trainval',
                 augment: bool = True,
                 keep_labels_for_eval: bool = False,
                 max_voxels: Optional[int] = None,
                 num_sweeps: Optional[int] = None,
                 cache_dir: Optional[str] = None,
                 require_cache: bool = False,
                 tables: Optional[Dict[str, Any]] = None,
                 gt_sampler=None):
        self.nusc_root = nusc_root
        self._nusc_version = version
        self.training  = training
        self.augment_flag = augment and training
        self.gt_sampler = gt_sampler if training else None
        self.keep_labels  = keep_labels_for_eval
        self.max_voxels   = max_voxels or (MAX_VOXELS_TRAIN if training else MAX_VOXELS_EVAL)
        self.num_sweeps = (num_sweeps if num_sweeps is not None
                           else (DEFAULT_NUM_SWEEPS if training else 1))
        from sgf_port_data import (default_nusc_npz_cache, nusc_frame_cache_path,
                                   verify_nusc_npz_cache)
        self.cache_dir = cache_dir or default_nusc_npz_cache()
        self.require_cache = require_cache

        with open(split_file) as f:
            self.sample_tokens = [ln.strip() for ln in f if ln.strip()]
        if self.cache_dir:
            n_cached = verify_nusc_npz_cache(
                self.cache_dir, self.sample_tokens, self.num_sweeps, self.require_cache)
            print(f"[dataset] npz cache {n_cached}/{len(self.sample_tokens)} at {self.cache_dir}",
                  flush=True)
        if tables is not None:
            self.tables = tables
        elif require_cache:
            self.tables = None
            print("[dataset] npz-only mode (require_cache); defer nuScenes JSON tables", flush=True)
        else:
            self.tables = load_nusc_tables(nusc_root, version)
        print(f"[dataset] {len(self.sample_tokens)} samples from {split_file}  "
              f"num_sweeps={self.num_sweeps}", flush=True)

    def _ensure_tables(self) -> Dict[str, Any]:
        if self.tables is None:
            self.tables = load_nusc_tables(self.nusc_root, self._nusc_version)
        return self.tables

    def __len__(self): return len(self.sample_tokens)

    def _load_sample_data(self, token: str) -> Dict[str, Any]:
        from sgf_port_data import (load_nusc_frame_npz, nusc_frame_cache_path,
                                   save_nusc_frame_npz)
        keep_lab = self.keep_labels or (not self.training)
        cache_path = (nusc_frame_cache_path(self.cache_dir, token, self.num_sweeps)
                      if self.cache_dir else None)
        if cache_path and cache_path.is_file():
            try:
                data = load_nusc_frame_npz(cache_path)
                if not keep_lab:
                    data.pop('labels', None)
                return data
            except (EOFError, OSError, ValueError, KeyError):
                pass  # corrupt cache removed; rebuild from LiDAR below
        if self.require_cache:
            raise FileNotFoundError(f"Missing npz cache: {cache_path}")
        data = load_nusc_sample(token, self.nusc_root, self._ensure_tables(),
                                keep_labels=keep_lab, num_sweeps=self.num_sweeps)
        if cache_path:
            save_nusc_frame_npz(cache_path, data['pts'], data['gt_boxes'],
                              data.get('labels', []), data['sample_id'])
        return data

    def __getitem__(self, idx) -> Dict[str, Any]:
        token = self.sample_tokens[idx]
        data  = self._load_sample_data(token)
        pts    = normalize_point_intensity(data['pts'])
        gt     = gt_array_for_training(data['gt_boxes'])

        if self.gt_sampler is not None:
            pts, gt = self.gt_sampler(pts, gt)
        if self.augment_flag:
            pts, gt = augment(pts, gt)

        voxels, coords, num_per_v = voxelize_pillars(pts, max_voxels=self.max_voxels)
        coarse_bev  = make_coarse_bev(pts)
        density_map = make_density_map(pts)

        if self.training:
            hm_t, reg_t, iou_t, pos_t = build_targets(gt, NUM_CLASSES, F_NY, F_NX)
        else:
            hm_t = reg_t = iou_t = pos_t = None

        out: Dict[str, Any] = {
            'voxels': voxels, 'coords': coords, 'num_per_v': num_per_v,
            'coarse_bev': coarse_bev, 'density_map': density_map,
            'gt_boxes': gt, 'sample_id': data['sample_id'], 'pts': pts,
            'sample_token': token,
        }
        if self.training:
            out.update({'hm_t': hm_t, 'reg_t': reg_t, 'iou_t': iou_t, 'pos_t': pos_t})
        if self.keep_labels or (not self.training):
            out['labels'] = data.get('labels', [])
        return out


def collate_nusc(batch_list: List[Dict]) -> Dict[str, Any]:
    B = len(batch_list)
    voxels_l, coords_l, num_l = [], [], []
    for bi, s in enumerate(batch_list):
        v=s['voxels']; c=s['coords'].copy(); c[:,0]=bi
        voxels_l.append(v); coords_l.append(c); num_l.append(s['num_per_v'])
    if voxels_l:
        voxels    = np.concatenate(voxels_l, 0)
        coords    = np.concatenate(coords_l, 0)
        num_per_v = np.concatenate(num_l,    0)
    else:
        voxels    = np.zeros((0, MAX_PTS, 4), np.float32)
        coords    = np.zeros((0, 3), np.int32)
        num_per_v = np.zeros((0,), np.int32)
    coarse_bev  = np.stack([s['coarse_bev']  for s in batch_list], 0)
    density_map = np.stack([s['density_map'] for s in batch_list], 0)
    out: Dict[str, Any] = {
        'voxels':      torch.from_numpy(voxels),
        'coords':      torch.from_numpy(coords).long(),
        'num_per_v':   torch.from_numpy(num_per_v).long(),
        'coarse_bev':  torch.from_numpy(coarse_bev),
        'density_map': torch.from_numpy(density_map),
        'gt_boxes':    [s['gt_boxes']  for s in batch_list],
        'sample_ids':    [s['sample_id'] for s in batch_list],
        'sample_tokens': [s.get('sample_token', s['sample_id']) for s in batch_list],
        'batch_size':  B,
    }
    if 'hm_t' in batch_list[0] and batch_list[0]['hm_t'] is not None:
        out['hm_t']  = torch.from_numpy(np.stack([s['hm_t']  for s in batch_list],0))
        out['reg_t'] = torch.from_numpy(np.stack([s['reg_t'] for s in batch_list],0))
        out['iou_t'] = torch.from_numpy(np.stack([s['iou_t'] for s in batch_list],0))
        out['pos_t'] = torch.from_numpy(np.stack([s['pos_t'] for s in batch_list],0))
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
    def __init__(self, path):
        self._f = open(path, 'a')
    def log(self, row, pretty_prefix=''):
        if pretty_prefix:
            parts = []
            for k, v in row.items():
                if isinstance(v, float): parts.append(f'{k}={v:.4f}')
                else: parts.append(f'{k}={v}')
            print(pretty_prefix + '  '.join(parts), flush=True)
        self._f.write(json.dumps(row) + '\n'); self._f.flush()
    def close(self): self._f.close()


class WeightEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1-self.decay) * v.float()
    def swap_in(self, model):
        backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.to(next(model.parameters()).device)
                               for k, v in self.shadow.items()}, strict=False)
        return backup
    def restore(self, model, backup): model.load_state_dict(backup)


def train_one_epoch(model, loader, opt, sched, scaler, dev, epoch, logger, use_amp=False):
    model.train(); t0 = time.time(); total_loss = 0; n = 0
    for i, batch in enumerate(loader):
        batch = move_batch_to_device(batch, dev)
        with amp_autocast(use_amp):
            out  = model(batch)
            loss, tb = model.head.compute_loss(
                {'hm': out['hm'], 'reg': out['reg'], 'iou': out['iou']},
                hm_t=batch['hm_t'], reg_t=batch['reg_t'],
                iou_t=batch['iou_t'], pos_t=batch['pos_t'],
                gate_log=out.get('gate_log'),
                aux_dg_logit=out.get('dg_logit'))
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
        total_loss += float(loss.detach()); n += 1
        if (i+1) % 20 == 0:
            logger.log({'epoch': epoch, 'step': i+1, **tb},
                       pretty_prefix=f'[train ep{epoch:03d} step{i+1:04d}] ')
    elapsed = time.time() - t0
    print(f'[train ep{epoch:03d}] avg_loss={total_loss/max(n,1):.4f}  t={elapsed:.0f}s')
    return total_loss / max(n, 1)


@torch.no_grad()
def _run_eval(model, loader, dev, soft_nms_flag=False,
              eval_score_thresh=0.05, progress_every=25, tag='val',
              iou_alpha: float = IOU_ALPHA_DEFAULT):
    from sgf_eval_official import eval_score_thresh_context, summarize_det_lists
    model.eval(); all_preds, all_gts = [], []
    t0 = time.time()
    with eval_score_thresh_context(model, eval_score_thresh, NUM_CLASSES):
        for bi, batch in enumerate(loader):
            if progress_every and bi > 0 and bi % progress_every == 0:
                print(f'[{tag}] batch {bi}/{len(loader)} …', flush=True)
            batch = move_batch_to_device(batch, dev)
            out = model(batch)
            preds = model.head.post_process(
                {'hm': out['hm'], 'reg': out['reg'], 'iou': out['iou']},
                use_soft_nms=soft_nms_flag, iou_alpha=iou_alpha)
            for pd in preds:
                all_preds.append({k: v.cpu() for k, v in pd.items()})
            for lbs in batch.get('labels', [[] for _ in range(batch['batch_size'])]):
                all_gts.append(lbs)
    stats = summarize_det_lists(all_preds, all_gts)
    print(f"[{tag}] inference {time.time()-t0:.0f}s  frames={stats['frames']}  "
          f"gt={stats['n_gt']}  pred={stats['n_pred']}  "
          f"eval_score_thresh={eval_score_thresh}", flush=True)
    return all_preds, all_gts


def _nusc_root(args) -> str:
    """Resolve nuScenes dataroot from --nusc_root or legacy --kitti_root."""
    root = getattr(args, 'nusc_root', None) or getattr(args, 'kitti_root', None)
    if not root:
        raise ValueError('nuScenes root required (--nusc_root or --kitti_root)')
    return root


def train(args):
    dev = resolve_torch_device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    root = _nusc_root(args)
    ver = getattr(args, 'nusc_version', 'v1.0-trainval')
    ns = getattr(args, 'num_sweeps', DEFAULT_NUM_SWEEPS)
    val_ns = getattr(args, 'val_num_sweeps', None)
    if val_ns is None:
        val_ns = 1
    cache_dir = getattr(args, 'cache_dir', None)
    require_cache = getattr(args, 'require_cache', False)
    require_cache_val = getattr(args, 'require_cache_val', False)
    do_inline_val = 0 < args.val_interval <= args.epochs

    gt_sampler = _build_gt_sampler(args)
    train_ds = NuscSGFDataset(root, args.train_split, training=True,
                               version=ver, augment=True, max_voxels=MAX_VOXELS_TRAIN,
                               num_sweeps=ns, cache_dir=cache_dir,
                               require_cache=require_cache, gt_sampler=gt_sampler)
    shared_tables = train_ds.tables
    val_ds = val_loader = None
    if do_inline_val:
        val_ds = NuscSGFDataset(root, args.val_split, training=False,
                                   version=ver, keep_labels_for_eval=True,
                                   max_voxels=MAX_VOXELS_EVAL, num_sweeps=val_ns,
                                   cache_dir=cache_dir, require_cache=require_cache_val,
                                   tables=shared_tables)
    nw = max(0, args.num_workers)
    val_nw = getattr(args, 'val_num_workers', None)
    if val_nw is None:
        val_nw = max(2, nw // 2) if nw > 0 else 4
    val_bs = getattr(args, 'val_batch_size', None) or args.batch_size
    dl_kw = dict(collate_fn=collate_nusc, pin_memory=True)
    if nw > 0:
        dl_kw.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, drop_last=True, **dl_kw)
    if getattr(args, 'balanced_resampling', False) or getattr(args, 'cls_balanced', False):
        print('[train] OpenPCDet CBGS class-balanced resampling enabled', flush=True)
        sampler = make_nuscenes_cbgs_sampler(train_ds, NUM_CLASSES)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=nw, drop_last=True, **dl_kw)
    if do_inline_val:
        val_dl_kw = dict(collate_fn=collate_nusc, pin_memory=True)
        if val_nw > 0:
            val_dl_kw.update(persistent_workers=True, prefetch_factor=2)
        val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False,
                                num_workers=val_nw, drop_last=False, **val_dl_kw)
    else:
        print('[train] inline val disabled — official eval via submit_eval.sh', flush=True)
    print(f'[train] cache={cache_dir}  require_cache_train={require_cache}  '
          f'require_cache_val={require_cache_val}  train_sweeps={ns}  val_sweeps={val_ns}', flush=True)

    st = getattr(args, 'score_thresh', 0.01)
    model = SGFDualGridModel(score_thresh=_per_class_thresh(st)).to(dev)
    params = model.count_params()
    print(f"[train] model params: total={params['total']:,}  "
          f"vfe={params['vfe']:,}  backbone={params['backbone']:,}  head={params['head']:,}"
          f"  score_thresh={st}")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=getattr(args, 'weight_decay', 1e-4))
    steps = len(train_loader) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                 pct_start=getattr(args, 'pct_start', 0.2),
                                                 div_factor=getattr(args, 'div_factor', 10.),
                                                 final_div_factor=getattr(args, 'final_div_factor', 100.))
    scaler = amp_grad_scaler(args.amp)
    ema    = WeightEMA(model, decay=args.ema_decay) if args.ema else None

    train_log = JsonLogger(str(out_dir/'train_log.jsonl'))
    val_log   = JsonLogger(str(out_dir/'val_log.jsonl'))
    best_metric = -1.0; start_ep = 1

    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt['model'])
        start_ep = ckpt.get('epoch', 0) + 1
        best_metric = ckpt.get('best_metric', -1.0)
        print(f"[train] resumed from {args.resume} ep={start_ep-1}  best={best_metric:.4f}")

    print(f"[train] {len(train_ds)} train"
          f"{f', {len(val_ds)} val' if val_ds is not None else ''} — "
          f"{args.epochs} ep, batch {args.batch_size}, lr {args.lr}")

    for ep in range(start_ep, args.epochs + 1):
        train_one_epoch(model, train_loader, opt, sched, scaler, dev, ep,
                        train_log, use_amp=args.amp)
        if ema: ema.update(model)

        # Checkpoint last
        ckpt = {'model': model.state_dict(), 'epoch': ep, 'best_metric': best_metric,
                'reg_encoding': REG_ENCODING}
        torch.save(ckpt, out_dir/'last.pt')

        if do_inline_val and ep % args.val_interval == 0:
            eval_st = getattr(args, 'eval_score_thresh', EVAL_SCORE_THRESH_DEFAULT)
            iou_a = getattr(args, 'iou_alpha', None) or IOU_ALPHA_DEFAULT
            print(f'[val ep{ep:03d}] official nuScenes eval (eval_score_thresh={eval_st}, iou_alpha={iou_a})…', flush=True)
            if ema: backup = ema.swap_in(model)
            all_preds, all_gts = _run_eval(model, val_loader, dev,
                                           eval_score_thresh=eval_st,
                                           tag=f'val ep{ep:03d}', iou_alpha=iou_a)
            if ema: ema.restore(model, backup)
            from sgf_eval_official import run_nuscenes_official_metrics
            official = run_nuscenes_official_metrics(all_preds, all_gts, print_table=True)
            metric_3d_mod = float(official['mean_ap'])
            print(f'[val ep{ep:03d}] official nuScenes mAP = {metric_3d_mod:.4f}')
            row = {'epoch': ep, 'official_mAP': metric_3d_mod,
                   **{f'mAP_{c}': official['mean_dist_aps'].get(c, 0.) for c in CLASS_NAMES}}
            val_log.log(row, pretty_prefix=f'[val ep{ep:03d}] ')

            if metric_3d_mod > best_metric:
                best_metric = metric_3d_mod
                if ema: backup2 = ema.swap_in(model)
                torch.save({'model': model.state_dict(), 'epoch': ep,
                            'best_metric': best_metric, 'ema_used': ema is not None,
                            'reg_encoding': REG_ENCODING}, out_dir/'best.pt')
                if ema: ema.restore(model, backup2)
                import json as _json
                (out_dir/'best.meta.json').write_text(
                    _json.dumps({'reg_encoding': REG_ENCODING, 'arch': 'midFusion_nusc',
                                 'epoch': ep, 'best_metric': float(best_metric)}))
                print(f"[val ep{ep:03d}] NEW BEST official mAP={best_metric:.4f} → best.pt")

    if not (out_dir / 'best.pt').is_file():
        if ema:
            bk = ema.swap_in(model)
        torch.save({'model': model.state_dict(), 'epoch': args.epochs,
                    'best_metric': best_metric, 'ema_used': ema is not None,
                    'reg_encoding': REG_ENCODING}, out_dir / 'best.pt')
        if ema:
            ema.restore(model, bk)
        import json as _json
        (out_dir / 'best.meta.json').write_text(
            _json.dumps({'reg_encoding': REG_ENCODING, 'arch': 'midFusion_nusc',
                         'epoch': args.epochs, 'best_metric': float(best_metric)}))
        print('[train] saved best.pt from final epoch (train-only / OpenPCDet protocol)', flush=True)

    train_log.close(); val_log.close()
    print(f"[train] DONE.  best official mAP = {best_metric:.4f}  out_dir={out_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# §16  EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(args):
    dev = resolve_torch_device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    root = _nusc_root(args)
    ver = getattr(args, 'nusc_version', 'v1.0-trainval')
    val_ns = getattr(args, 'val_num_sweeps', None)
    if val_ns is None:
        val_ns = 1
    cache_dir = getattr(args, 'cache_dir', None)
    require_cache_val = getattr(args, 'require_cache_val', False)
    ds = NuscSGFDataset(root, args.val_split, training=False,
                        version=ver, keep_labels_for_eval=True,
                        max_voxels=MAX_VOXELS_EVAL, num_sweeps=val_ns,
                        cache_dir=cache_dir, require_cache=require_cache_val)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_nusc,
                        pin_memory=True, drop_last=False)
    st = getattr(args, 'score_thresh', 0.01)
    eval_st = getattr(args, 'eval_score_thresh', EVAL_SCORE_THRESH_DEFAULT)
    iou_a = getattr(args, 'iou_alpha', None) or IOU_ALPHA_DEFAULT
    model = SGFDualGridModel(score_thresh=_per_class_thresh(st)).to(dev)
    ckpt  = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt['model']); model.eval()
    print(f"[eval] loaded {args.ckpt}  reg_encoding={ckpt.get('reg_encoding','?')}  "
          f"score_thresh={st}  eval_score_thresh={eval_st}  iou_alpha={iou_a}")
    all_preds, all_gts = _run_eval(model, loader, dev,
                                   soft_nms_flag=getattr(args,'soft_nms',False),
                                   eval_score_thresh=eval_st, tag='eval', iou_alpha=iou_a)
    from sgf_eval_official import run_nuscenes_official_metrics, format_nuscenes_eval_table
    official = run_nuscenes_official_metrics(all_preds, all_gts, print_table=True)
    metric = float(official['mean_ap'])
    nds = float(official.get('NDS', 0.0))
    results = evaluate_nusc(all_preds, all_gts, iou_mode='3d')
    print("\n[eval] diagnostic IoU AP (KITTI-style E/M/H):")
    print_ap_table(results)
    apbev = evaluate_nusc(all_preds, all_gts, iou_mode='bev')
    print("\n[eval] diagnostic BEV IoU:")
    print_ap_table(apbev)
    import json as _json
    out = {
        'official_nuscenes': official,
        'official_mAP': metric,
        'NDS': nds,
        'diagnostic_AP3D': results,
        'diagnostic_APBEV': apbev,
    }
    (out_dir/'eval_results.json').write_text(_json.dumps(out, indent=2))
    (out_dir/'eval_table.txt').write_text(format_nuscenes_eval_table(official))
    print(f"[eval] nuScenes mAP={metric:.2f}%  NDS={nds:.4f}  → {out_dir}/eval_results.json")


# ═══════════════════════════════════════════════════════════════════════════════
# §17  VISUALIZER
# ═══════════════════════════════════════════════════════════════════════════════

def visualize(args):
    from sgf_viz_common import (gate_maps_to_numpy, resolve_viz_subdir,
                                viz_semantic_gate_grid)

    dev = resolve_torch_device(args.device)
    viz_sub = resolve_viz_subdir(args)
    out_dir = Path(args.out_dir) / viz_sub
    out_dir.mkdir(parents=True, exist_ok=True)
    root = _nusc_root(args)
    ver = getattr(args, 'nusc_version', 'v1.0-trainval')
    val_ns = getattr(args, 'val_num_sweeps', None)
    if val_ns is None:
        val_ns = 1
    cache_dir = getattr(args, 'cache_dir', None)
    require_cache_val = getattr(args, 'require_cache_val', False)
    ds = NuscSGFDataset(root, args.val_split, training=False,
                         version=ver, keep_labels_for_eval=True,
                         max_voxels=MAX_VOXELS_EVAL, num_sweeps=val_ns,
                         cache_dir=cache_dir, require_cache=require_cache_val)
    n = min(args.num_viz, len(ds))
    model = SGFDualGridModel(save_gates=True).to(dev)
    ckpt  = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt['model']); model.eval()
    ext = VIZ_IMAGE_EXT
    print(f"[viz] producing {n} samples  panel=semantic-6  subdir={viz_sub}  format=PNG")
    for i in range(n):
        sample = ds[i]
        batch  = move_batch_to_device(collate_nusc([sample]), dev)
        out    = model(batch)
        pd     = model.head.post_process({'hm': out['hm'], 'reg': out['reg'], 'iou': out['iou']})[0]
        pb = pd['pred_boxes'].cpu().numpy(); ps = pd['pred_scores'].cpu().numpy()
        pl = pd['pred_labels'].cpu().numpy()
        gt_boxes = sample['gt_boxes']; pts = sample['pts']; sid = sample['sample_id']

        viz_bev_detection(pts, gt_boxes, pb, ps, pl,
            save_path=str(out_dir / f'{sid}_bev_detection{ext}'),
            title=f'nuScenes {sid}  —  green: GT,  red: pred')
        viz_3d_detection(pts, gt_boxes, pb, ps, pl,
            save_path=str(out_dir / f'{sid}_3d_detection{ext}'),
            title=f'nuScenes {sid}  —  3D boxes (green GT, red pred)')
        density_map = sample['density_map'][0]
        gate_arrays = gate_maps_to_numpy(model.backbone.neck.gate_maps)
        viz_semantic_gate_grid(
            density_map, gate_arrays,
            save_path=str(out_dir / f'{sid}_density_gates{ext}'),
            title=f'nuScenes {sid}  —  MidFusion density gate heatmaps',
            pc_range=PC_RANGE, model_tag='MidFusion')
        if gate_arrays:
            viz_gate_histograms(gate_arrays,
                save_path=str(out_dir / f'{sid}_gate_histograms{ext}'))
        print(f"[viz] {sid}: GT={len(gt_boxes)}  pred={len(pb)}")
    print(f"[viz] DONE → {out_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# §18  SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════════

def smoke_test(args):
    """Quick forward pass with random points to verify the pipeline."""
    dev = resolve_torch_device(args.device)
    print(f"[smoke] device={dev}")
    # Random points in ego frame (360° nuScenes range)
    rng = np.random.default_rng(0)
    N = 25000
    x = rng.uniform(PC_RANGE[0], PC_RANGE[3], N).astype(np.float32)
    y = rng.uniform(PC_RANGE[1], PC_RANGE[4], N).astype(np.float32)
    z = rng.uniform(PC_RANGE[2], PC_RANGE[5], N).astype(np.float32)
    intensity = rng.uniform(0, 1, N).astype(np.float32)
    pts = np.stack([x, y, z, intensity], axis=1)

    # Dummy gt box (Car)
    gt = np.array([[5., 2., 0.8, 4.6, 2.0, 1.7, 0.3, 1.]], dtype=np.float32)

    voxels, coords, num_per_v = voxelize_pillars(pts, max_voxels=2000)
    coarse_bev  = make_coarse_bev(pts)
    density_map = make_density_map(pts)

    batch = {
        'voxels':     torch.from_numpy(voxels).to(dev),
        'coords':     torch.from_numpy(coords).long().to(dev),
        'num_per_v':  torch.from_numpy(num_per_v).long().to(dev),
        'coarse_bev': torch.from_numpy(coarse_bev).unsqueeze(0).to(dev),
        'density_map':torch.from_numpy(density_map).unsqueeze(0).to(dev),
        'gt_boxes':   [gt],
        'batch_size': 1,
    }
    hm_t, reg_t, iou_t, pos_t = build_targets(gt)
    batch.update({
        'hm_t':  torch.from_numpy(hm_t).unsqueeze(0).to(dev),
        'reg_t': torch.from_numpy(reg_t).unsqueeze(0).to(dev),
        'iou_t': torch.from_numpy(iou_t).unsqueeze(0).to(dev),
        'pos_t': torch.from_numpy(pos_t).unsqueeze(0).to(dev),
    })
    model = SGFDualGridModel().to(dev)
    p = model.count_params()
    print(f"[smoke] model params: total={p['total']:,}")
    out = model(batch)
    loss, tb = model.head.compute_loss(
        {'hm': out['hm'], 'reg': out['reg'], 'iou': out['iou']},
        hm_t=batch['hm_t'], reg_t=batch['reg_t'],
        iou_t=batch['iou_t'], pos_t=batch['pos_t'],
        gate_log=out.get('gate_log'), aux_dg_logit=out.get('dg_logit'))
    print(f"[smoke] loss={float(loss):.4f}  "
          f"hm={tb['loss_hm']:.4f}  reg={tb['loss_reg']:.4f}  iou={tb['loss_iou']:.4f}")
    print(f"[smoke] F_NX={F_NX} F_NY={F_NY} C_NX={C_NX} C_NY={C_NY}")
    print(f"[smoke] hm={tuple(out['hm'].shape)}  reg={tuple(out['reg'].shape)}")
    print("[smoke] PASS ✓")


def train_smoke(args):
    """Run a few real training steps (catches NaN loss before a long Slurm job)."""
    dev = resolve_torch_device(args.device)
    use_amp = bool(getattr(args, 'amp', False))
    root = _nusc_root(args)
    ver = getattr(args, 'nusc_version', 'v1.0-trainval')
    train_ds = NuscSGFDataset(root, args.train_split, training=True, version=ver,
                                augment=True, max_voxels=MAX_VOXELS_TRAIN,
                                num_sweeps=getattr(args, 'num_sweeps', DEFAULT_NUM_SWEEPS))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate_nusc, drop_last=True)
    model = SGFDualGridModel().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=getattr(args, 'weight_decay', 1e-4))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(3, len(loader)), pct_start=.04,
        div_factor=10, final_div_factor=100)
    scaler = amp_grad_scaler(use_amp)
    n_steps = int(getattr(args, 'train_smoke_steps', 1))
    print(f"[train_smoke] device={dev}  amp={use_amp}  batches={len(loader)}  steps={n_steps}", flush=True)
    ok = 0
    for step, batch in enumerate(loader):
        if step >= n_steps:
            break
        print(f"[train_smoke] step {step+1}/{n_steps}: forward…", flush=True)
        batch = move_batch_to_device(batch, dev)
        with amp_autocast(use_amp):
            out = model(batch)
            loss, tb = model.head.compute_loss(
                {'hm': out['hm'], 'reg': out['reg'], 'iou': out['iou']},
                hm_t=batch['hm_t'], reg_t=batch['reg_t'],
                iou_t=batch['iou_t'], pos_t=batch['pos_t'],
                gate_log=out.get('gate_log'), aux_dg_logit=out.get('dg_logit'))
        if not torch.isfinite(loss):
            print(f"[train_smoke] FAIL step {step+1}: non-finite loss {tb}")
            sys.exit(1)
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
        ok += 1
        print(f"[train_smoke] step {step+1}: loss={float(loss):.4f}  hm={tb['loss_hm']:.4f}", flush=True)
    print(f"[train_smoke] PASS ({ok} steps)", flush=True)
    return ok


def data_smoke_test(args):
    """Load one real nuScenes keyframe (+ sweeps) and run a forward pass."""
    dev = resolve_torch_device(args.device)
    root = _nusc_root(args)
    ver = getattr(args, 'nusc_version', 'v1.0-trainval')
    split = args.val_split or args.train_split
    if not split:
        raise ValueError('--train_split or --val_split required for data_smoke')
    print(f"[data_smoke] root={root}  version={ver}  split={split}")
    ds = NuscSGFDataset(root, split, training=True, version=ver,
                        augment=False, max_voxels=8000,
                        num_sweeps=getattr(args, 'num_sweeps', DEFAULT_NUM_SWEEPS))
    sample = ds[0]
    pts_n = sample['pts'].shape[0]
    gt_n = sample['gt_boxes'].shape[0]
    print(f"[data_smoke] sample={sample['sample_id']}  pts={pts_n}  gt={gt_n}")
    batch = move_batch_to_device(collate_nusc([sample]), dev)
    model = SGFDualGridModel().to(dev)
    out = model(batch)
    loss, tb = model.head.compute_loss(
        {'hm': out['hm'], 'reg': out['reg'], 'iou': out['iou']},
        hm_t=batch['hm_t'], reg_t=batch['reg_t'],
        iou_t=batch['iou_t'], pos_t=batch['pos_t'],
        gate_log=out.get('gate_log'), aux_dg_logit=out.get('dg_logit'))
    print(f"[data_smoke] loss={float(loss):.4f}  hm={tb['loss_hm']:.4f}  "
          f"pts={pts_n}  sweeps={ds.num_sweeps}")
    print("[data_smoke] PASS ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# §19  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="SGF-DualGrid MidFusion nuScenes runner")
    p.add_argument('--mode', choices=['smoke', 'data_smoke', 'train_smoke', 'train', 'eval', 'viz'],
                   required=True)
    p.add_argument('--nusc_root', type=str, default=None,
                   help='nuScenes dataroot (contains v1.0-trainval/, samples/, sweeps/)')
    p.add_argument('--kitti_root',  type=str, default=None,
                   help='Alias for --nusc_root (legacy name)')
    p.add_argument('--nusc_version', type=str, default='v1.0-trainval')
    p.add_argument('--num_sweeps', type=int, default=DEFAULT_NUM_SWEEPS,
                   help='LiDAR sweeps fused per sample during training (default 10)')
    p.add_argument('--cache_dir', type=str, default=None,
                   help='NPZ cache for fused nuScenes frames (default: NUSC_NPZ_CACHE)')
    p.add_argument('--require_cache', action='store_true',
                   help='Fail if NPZ cache entry missing (skip LiDAR disk reads)')
    p.add_argument('--train_split', type=str, default=None)
    p.add_argument('--val_split',   type=str, default=None)
    p.add_argument('--batch_size',  type=int, default=4)
    p.add_argument('--val_batch_size', type=int, default=None,
                   help='Val/eval batch size (default: same as --batch_size)')
    p.add_argument('--num_workers', type=int, default=6)
    p.add_argument('--epochs',      type=int, default=20)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--amp',         action='store_true')
    p.add_argument('--log_every',   type=int, default=10)
    p.add_argument('--val_interval',type=int, default=2)
    p.add_argument('--resume',      type=str, default=None)
    p.add_argument('--ckpt',        type=str, default=None)
    p.add_argument('--num_viz',     type=int, default=12)
    p.add_argument('--viz_subdir',  type=str, default='viz_semantic',
                   help='Subfolder under out_dir for viz PNGs (KITTI convention)')
    p.add_argument('--ema',         action='store_true')
    p.add_argument('--ema_decay',   type=float, default=0.999)
    p.add_argument('--cls_balanced',action='store_true')
    p.add_argument('--soft_nms',    action='store_true')
    p.add_argument('--score_thresh', type=float, default=0.01,
                   help='Training/inference score threshold (per class)')
    p.add_argument('--eval_score_thresh', type=float, default=EVAL_SCORE_THRESH_DEFAULT,
                   help='Low threshold for val/eval AP (full PR curve)')
    p.add_argument('--out_dir',     type=str,
                   default='/home/frahman8/scratch/SGF_Port/nuscenes/Runs/sgfd_nusc')
    p.add_argument('--device',      type=str, default='cuda')
    p.add_argument('--seed',        type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if str(args.device).lower().startswith('cuda'): torch.cuda.manual_seed_all(args.seed)

    if args.mode == 'smoke':
        smoke_test(args)
    elif args.mode == 'data_smoke':
        if not (args.nusc_root or args.kitti_root) or not (args.train_split or args.val_split):
            print("ERROR: --nusc_root and --train_split or --val_split required", file=sys.stderr)
            sys.exit(2)
        data_smoke_test(args)
    elif args.mode == 'train_smoke':
        if not (args.nusc_root or args.kitti_root) or not args.train_split:
            print("ERROR: --nusc_root and --train_split required", file=sys.stderr)
            sys.exit(2)
        train_smoke(args)
    elif args.mode == 'train':
        if not (args.nusc_root or args.kitti_root) or not args.train_split:
            print("ERROR: --nusc_root and --train_split required", file=sys.stderr)
            sys.exit(2)
        train(args)
    elif args.mode == 'eval':
        if not (args.nusc_root or args.kitti_root) or not args.val_split or not args.ckpt:
            print("ERROR: --nusc_root --val_split --ckpt required", file=sys.stderr)
            sys.exit(2)
        evaluate(args)
    elif args.mode == 'viz':
        if not (args.nusc_root or args.kitti_root) or not args.val_split or not args.ckpt:
            print("ERROR: --nusc_root --val_split --ckpt required", file=sys.stderr)
            sys.exit(2)
        visualize(args)


if __name__ == '__main__':
    main()
