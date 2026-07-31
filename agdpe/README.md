# AG-DPE (Thesis Ch. 4)

Density-aware Adaptive Encoding for LiDAR 3D object detection.

## Contents

| Path | Description |
|------|-------------|
| `code/model_agdpe.py` | KITTI training / eval / viz entry |
| `code/agdpe_waymo.py` | Waymo runner |
| `code/agdpe_nuscenes.py` | nuScenes runner |
| `logs/kitti_62600156/` | Train + val JSONL |
| `logs/nuscenes_64331453/` | nuScenes train + val JSONL |
| `logs/waymo_64331451/` | Waymo train + val JSONL |
| `results/kitti/` | `eval_results.json`, `eval_table.txt` |
| `results/nuscenes/` | nuScenes eval (mAP 0.696, NDS 0.737) |
| `results/waymo/` | Waymo eval (AP L1 87.30, APH L2 81.70) |
| `figures/` | Chapter 4 figures |

## Primary runs

| Dataset | Job ID | Moderate / primary metric |
|---------|--------|---------------------------|
| KITTI | `62600156` | mAP₃D Mod **66.53** |
| nuScenes | `64331453` | mAP **0.696**, NDS **0.737** |
| Waymo | `64331451` | AP L2 **83.20**, APH L2 **81.70** |

Checkpoints (`.pt`) are not packaged. Point `--ckpt` at your trained `best.pt`.

## Example

```bash
python agdpe/code/model_agdpe.py --mode eval \
  --kitti_root /path/to/KITTI \
  --val_split  /path/to/val.txt \
  --ckpt       /path/to/best.pt \
  --out_dir    agdpe/results/kitti
```
