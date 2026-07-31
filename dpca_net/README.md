# DPCA-Net (Thesis Ch. 5)

Dual-grid cross-stream fusion (mid-level DPCA). Implementation based on the historical `SGFDualGrid` MidFusion architecture.

## Contents

| Path | Description |
|------|-------------|
| `code/dpca_net_kitti.py` | KITTI entry (MidFusion / BCAF-DH) |
| `code/dpca_net_waymo.py` | Waymo entry |
| `code/dpca_net_nuscenes.py` | nuScenes entry |
| `configs/` | Dataset YAML configs |
| `logs/kitti_62292752/` | KITTI train + val JSONL |
| `logs/nuscenes_64331452/` | nuScenes train + val JSONL |
| `logs/waymo_64331450/` | Waymo train + val JSONL |
| `results/kitti/` | Eval + `ablation/` fusion-placement JSONs |
| `results/nuscenes/` | nuScenes eval (70.0% mAP, 73.9 NDS) |
| `results/waymo/` | Waymo eval (88.05 AP L1, 82.52 APH L2) |
| `figures/` | Chapter 5 figures |

## Primary runs

| Dataset | Job ID | Moderate / primary metric |
|---------|--------|---------------------------|
| KITTI | `62292752` | mAP₃D Mod **75.58** |
| nuScenes | `64331452` | mAP **70.0%**, NDS **73.9%** |
| Waymo | `64331450` | AP L1 **88.05**, APH L2 **82.52** |

Ablation peers: `results/kitti/ablation/` (Early / Mid / Head / no-density).

Checkpoints (`.pt`) are not packaged. Point `--ckpt` at your trained `best.pt`.

## Example

```bash
python dpca_net/code/dpca_net_kitti.py --mode eval \
  --kitti_root /path/to/KITTI \
  --val_split  /path/to/val.txt \
  --ckpt       /path/to/best.pt \
  --out_dir    dpca_net/results/kitti
```
