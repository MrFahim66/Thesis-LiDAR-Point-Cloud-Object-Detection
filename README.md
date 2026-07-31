# DPCA-Net & AG-DPE

Official code release for **DPCA-Net** (Dual-grid Pillar Cross-stream Attention Network) and **AG-DPE** (Adaptive Geometry- and Density-aware Pillar Encoding) — LiDAR-only 3D object detectors evaluated on KITTI, nuScenes, and Waymo Open Dataset.

Both methods extend pillar-based BEV detection with lightweight fusion modules that adapt to spatial point density and multi-scale context. This repository contains model implementations, training/evaluation configs, result JSONs, training logs, and figures from the associated MSc thesis.

---

## Highlights

| Method | Core idea | Best overall result (reported) |
|--------|-----------|--------------------------------|
| **AG-DPE** | Density-guided gating + density-aware attention on a single pillar grid | KITTI val mAP₃D (Mod) **66.53%** |
| **DPCA-Net** | Fine + coarse dual-grid encoding with mid-level bidirectional cross-attention | KITTI val mAP₃D (Mod) **75.58%** |

Both models are evaluated cross-dataset on nuScenes and Waymo under each benchmark's official LiDAR-only protocol.

---

## Results

### KITTI validation (Moderate)

**AG-DPE** — job `62600156` · [`agdpe/results/kitti/`](agdpe/results/kitti/)

| Class | Car | Pedestrian | Cyclist | **mAP₃D** |
|-------|-----|------------|---------|-----------|
| Moderate AP₃D | 78.96 | 55.03 | 65.59 | **66.53** |

**DPCA-Net (MidFusion)** — job `62292752` · [`dpca_net/results/kitti/`](dpca_net/results/kitti/)

| Class | Car | Pedestrian | Cyclist | **mAP₃D** |
|-------|-----|------------|---------|-----------|
| Moderate AP₃D | 79.37 | 69.82 | 77.55 | **75.58** |

Fusion-placement ablations (Early / Mid / Head / no-density): [`dpca_net/results/kitti/ablation/`](dpca_net/results/kitti/ablation/)

### nuScenes (LiDAR-only)

| Method | Job | mAP | mASE ↓ | NDS |
|--------|-----|-----|--------|-----|
| AG-DPE | 64331453 | 0.696 | 0.229 | 0.737 |
| DPCA-Net | 64331452 | 70.0% | 0.229 | 73.9% |

Eval: `agdpe/results/nuscenes/` · `dpca_net/results/nuscenes/`

### Waymo Open Dataset (LiDAR-only)

| Method | Job | AP L1 | APH L1 | AP L2 | APH L2 |
|--------|-----|-------|--------|-------|--------|
| AG-DPE | 64331451 | 87.30 | 85.70 | 83.20 | 81.70 |
| DPCA-Net | 64331450 | 88.05 | 86.50 | 84.08 | 82.52 |

Eval: `agdpe/results/waymo/` · `dpca_net/results/waymo/`

Full metric breakdowns: [`docs/METRICS.md`](docs/METRICS.md)

---

## Repository layout

```
.
├── agdpe/                  # AG-DPE (density-adaptive pillar encoding)
│   ├── code/               # KITTI, nuScenes, Waymo runners
│   ├── logs/               # train_log.jsonl + val_log.jsonl per job
│   ├── results/            # eval_results.json + eval_table.txt
│   └── figures/            # paper / thesis figures (Ch. 4)
├── dpca_net/               # DPCA-Net (dual-grid mid-fusion)
│   ├── code/
│   ├── configs/            # dataset YAML configs
│   ├── logs/
│   ├── results/
│   └── figures/            # paper / thesis figures (Ch. 5)
├── datasets/               # shared dataset config templates
├── figures/                # intro BEV panels, pred-vs-GT visualizations
├── docs/                   # metrics reference, GitHub setup notes
└── sgf_classification/     # Ch. 3 SiGF-Net classification figures
```

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 + CUDA (training/eval)
- NumPy, Matplotlib (visualization)
- KITTI / nuScenes / Waymo devkit or preprocessed NPZ caches (see `datasets/`)

Checkpoints (`.pt`) are **not** bundled due to size. Download or train locally and point `--ckpt` to your `best.pt`.

---

## Quick start

### Smoke test (no data)

```bash
python agdpe/code/model_agdpe.py --mode smoke --device cuda
python dpca_net/code/dpca_net_kitti.py --mode smoke --device cuda
```

### KITTI evaluation

```bash
# AG-DPE
python agdpe/code/model_agdpe.py --mode eval \
  --kitti_root /path/to/KITTI \
  --val_split  /path/to/val.txt \
  --ckpt       /path/to/agdpe_best.pt \
  --out_dir    agdpe/results/kitti

# DPCA-Net (MidFusion)
python dpca_net/code/dpca_net_kitti.py --mode eval \
  --kitti_root /path/to/KITTI \
  --val_split  /path/to/val.txt \
  --ckpt       /path/to/dpca_best.pt \
  --out_dir    dpca_net/results/kitti
```

### nuScenes / Waymo

```bash
python agdpe/code/agdpe_nuscenes.py --mode eval --ckpt /path/to/best.pt ...
python dpca_net/code/dpca_net_waymo.py --mode eval --ckpt /path/to/best.pt ...
```

See per-method READMEs: [`agdpe/README.md`](agdpe/README.md) · [`dpca_net/README.md`](dpca_net/README.md)

---

## Training logs

Each run folder under `logs/` contains:

| File | Contents |
|------|----------|
| `train_log.jsonl` | Per-step loss components, learning rate, gate statistics |
| `val_log.jsonl` | Periodic validation metrics (dataset-specific AP / NDS / Waymo L1–L2) |

Example job IDs: KITTI `62600156` / `62292752`; nuScenes `64331453` / `64331452`; Waymo `64331451` / `64331450`.

---

## Citation

If you use this code or results, please cite:

```bibtex
@mastersthesis{rahman2026lidar,
  author  = {Fahim Ur Rahman},
  title   = {Efficient LiDAR-Based 3D Object Detection via Density-Aware and Dual-Grid Fusion Architectures},
  school  = {Lakehead University},
  year    = {2026},
  type    = {MSc Thesis}
}
```


---

## License

See [`LICENSE`](LICENSE). Thesis figures remain © the author unless otherwise noted.

---

## Acknowledgements

Built on pillar-based 3D detection pipelines inspired by PointPillars and OpenPCDet training protocols. Benchmarks: [KITTI](http://www.cvlibs.net/datasets/kitti/), [nuScenes](https://www.nuscenes.org/), [Waymo Open Dataset](https://waymo.com/open/).
