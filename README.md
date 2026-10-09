# DCS Artifact

Data and code for the paper. Contains the two deployment datasets
(Ransomware, Cryptominer-2) and the DCS reference implementation.

## Contents

```
Ransomware/   labels + base-classifier scores (lab / wild)
miner2/       Cryptominer-2, same structure
stacking.py   core pipeline (calibration, LR stacking)
eval_single.py
run_dcs.py    DCS: MVS-LR construction + drift diagnosis + Tier-1 adaptation
run_tier2.py  Tier-2: budget-conditional label acquisition (B=500)
requirements.txt
```

## Run

```bash
pip install -r requirements.txt
python3 run_dcs.py     # construction, diagnosis, Tier 1
python3 run_tier2.py   # Tier 2 (declines or acquires per diagnosis)
```

Expected output (wild-domain AUC):

| Dataset | MVS-LR | Diagnosis | Tier 1 | Tier 2 |
|---|---|---|---|---|
| Ransomware | .9341 | d_cov=.713, d_con=.178 | freeze → .9341 | decline → .9341 |
| Cryptominer-2 | .9773 | d_cov=.683, d_con=.050 | enable IW → .9817 | acquire → .9846±.0009 |

Hyperparameters (fixed across datasets):
(τ_A, λ_d, τ_D, τ_C, τ_cov, κ, λ_2) = (0.55, 1.0, 0.6, 0.05, 0.65, 20, 1e-3).
Tier 2: B=500, 5 rounds, margin-based acquisition, five seeds.
