# DCS Artifact — Drift-Diagnosis-Conditioned Stacking

Code and data accompanying the paper. This repository contains the two
deployment datasets (**Ransomware** and **Cryptominer2**) and the complete
DCS implementation.

## Layout

```
Ransomware/                 Ransomware dataset (labels + classifier scores)
  balanced_lab_dataset_1to2.csv    Lab ground truth (hash, label)
  balanced_wild_dataset_1to2.csv   Wild ground truth (hash, label, first_submission_date)
  lab_results/                     Per-classifier lab scores (hash, confidence)
  real_world_results/              Per-classifier wild scores
miner2/                     Cryptominer-2 dataset (same structure)
  lab_label.csv / wild_label.csv   Ground truth (hash, label, first_submission)
  lab/ , wild/                     Per-classifier scores
stacking.py                 Core pipeline (dataset assembly, isotonic calibration, LR)
eval_single.py              Utilities (AUC, score loading)
run_dcs.py                  Full DCS procedure (construction → diagnosis → tiers)
requirements.txt
```

## Data format

- `label`: 1 = malicious, 0 = benign.
- Sample timestamps (`first_submission_date`) are VirusTotal first-submission
  records (Unix seconds); the lab/wild split is a strict temporal partition.
- Classifier identifiers (`PF`, `TIF`, `EHc`, ...) denote feature-view roles
  as defined in the paper appendix.
- The third dataset used in the paper (Cryptominer-1) is not redistributed here.

## Reproducing DCS

```bash
pip install -r requirements.txt
python3 run_dcs.py
```

The script reproduces the full pipeline on both datasets:
MVS-LR construction (calibration → validity filtering → error-correlation
clustering), drift diagnosis (domain-discriminator AUC + pilot ECE), and the
conditional Tier-1 adaptation. Expected output (wild-domain AUC):

| Dataset | Backbone (full V) | MVS-LR (S) | Diagnosis | Tier 1 |
|---|---|---|---|---|
| Ransomware | .9329 | **.9341** | d_cov=.713, d_con=.178 | freeze → **.9341** |
| Cryptominer2 | .9783 | .9773 | d_cov=.683, d_con=.050 | enable IW → **.9817** |

## Notes

- DCS hyperparameters are fixed across datasets:
  (τ_A, λ_d, τ_D, τ_C, τ_cov, κ, λ_2) = (0.55, 1.0, 0.6, 0.05, 0.65, 20, 1e-3).
- The diagnostic pilot is stratified-random sampled (n = 500); acquisition
  experiments report five-seed means as in the paper.
