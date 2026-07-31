# Metrics reference

Evaluation JSON and tables under each method's `results/` directory. Training curves are in the matching `logs/` folder.

## AG-DPE

### KITTI validation — job `62600156`

| Source | `agdpe/results/kitti/eval_results.json` |
|--------|----------------------------------------|
| Overall Moderate mAP₃D | **66.53** |
| Car / Ped / Cyc (Mod AP₃D) | 78.96 / 55.03 / 65.59 |

### nuScenes — job `64331453`

| Source | `agdpe/results/nuscenes/eval_results.json` |
|--------|---------------------------------------------|
| mAP | **0.696** |
| mASE | **0.229** |
| NDS | **0.737** |

Logs: `agdpe/logs/nuscenes_64331453/`

### Waymo — job `64331451`

| Source | `agdpe/results/waymo/eval_results.json` |
|--------|------------------------------------------|
| AP L1 / APH L1 | **87.30** / **85.70** |
| AP L2 / APH L2 | **83.20** / **81.70** |

Logs: `agdpe/logs/waymo_64331451/`

---

## DPCA-Net (MidFusion)

### KITTI validation — job `62292752`

| Source | `dpca_net/results/kitti/eval_results.json` |
|--------|---------------------------------------------|
| Overall Moderate mAP₃D | **75.58** |
| Car / Ped / Cyc (Mod AP₃D) | 79.37 / 69.82 / 77.55 |

### nuScenes — job `64331452`

| Source | `dpca_net/results/nuscenes/eval_results.json` |
|--------|------------------------------------------------|
| mAP | **70.0%** |
| mASE | **0.229** |
| NDS | **73.9%** |

Logs: `dpca_net/logs/nuscenes_64331452/`

### Waymo — job `64331450`

| Source | `dpca_net/results/waymo/eval_results.json` |
|--------|---------------------------------------------|
| AP L1 / APH L1 | **88.05** / **86.50** |
| AP L2 / APH L2 | **84.08** / **82.52** |

Logs: `dpca_net/logs/waymo_64331450/`

---

## Ablation runs (KITTI)

JSON summaries in `dpca_net/results/kitti/ablation/`:

| File | Fusion placement |
|------|------------------|
| `MidFusion_62292752.json` | Mid-level DPCA (default) |
| `HeadFusion_62292757.json` | Head fusion |
| `EarlyFusion_62292743.json` | Early fusion |
| `noDensity_62292740.json` | Mid fusion, density gate off |

---

## Log / eval index

| Method | Dataset | Logs | Eval |
|--------|---------|------|------|
| AG-DPE | KITTI | `agdpe/logs/kitti_62600156/` | `agdpe/results/kitti/` |
| AG-DPE | nuScenes | `agdpe/logs/nuscenes_64331453/` | `agdpe/results/nuscenes/` |
| AG-DPE | Waymo | `agdpe/logs/waymo_64331451/` | `agdpe/results/waymo/` |
| DPCA-Net | KITTI | `dpca_net/logs/kitti_62292752/` | `dpca_net/results/kitti/` |
| DPCA-Net | nuScenes | `dpca_net/logs/nuscenes_64331452/` | `dpca_net/results/nuscenes/` |
| DPCA-Net | Waymo | `dpca_net/logs/waymo_64331450/` | `dpca_net/results/waymo/` |
