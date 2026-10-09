# -*- coding: utf-8 -*-
"""
Tier-2 reference implementation:
budget-conditional label acquisition (margin-based, 5 rounds x 100).
Runs on top of the Tier-1 state produced by run_dcs.py.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stacking import build_dataset, fit_lr, predict_lr, roc_auc
from run_dcs import (w1_dist, fit_lr_np_weighted, predict_m,
                     hier_cluster_prune, TAU_C, TAU_COV, KAPPA, LAM_D, LAM_2)

BUDGET, ROUNDS, N_SEEDS = 500, 5, 5

def stratified_lab(yl, n, seed):
    rng = np.random.RandomState(seed)
    pos = np.where(yl == 1)[0]; neg = np.where(yl == 0)[0]
    kk = min(n // 2, len(pos), len(neg))
    nk = min(n - kk, len(neg)); kk = min(kk, n - nk)
    parts = []
    if kk > 0: parts.append(rng.choice(pos, kk, replace=False))
    if nk > 0: parts.append(rng.choice(neg, nk, replace=False))
    return np.concatenate(parts) if parts else np.array([], dtype=int)

def ece10(p, y):
    edges = np.linspace(0, 1, 11); e = 0.0
    for i in range(10):
        m = (p >= edges[i]) & (p < edges[i+1]) if i < 9 else (p >= edges[9]) & (p <= edges[10])
        if m.sum(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

def main():
    for ds in ["Ransomware", "miner2"]:
        d = build_dataset(ds)
        names = d["retained"]; k = len(names)
        Xl, yl = np.array(d["Xl"]), np.array(d["yl"])
        Xw, yw = np.array(d["Xw"]), np.array(d["yw"])

        # construction (same as run_dcs)
        delta = np.array([w1_dist(Xl[:, j], Xw[:, j]) for j in range(k)])
        A_lab = np.array([roc_auc(Xl[:, j].tolist(), yl.tolist()) for j in range(k)])
        A_star = A_lab * np.exp(-LAM_D * delta)
        E_mat = np.column_stack([(Xl[:, j] >= 0.5) != (yl == 1) for j in range(k)]).astype(int)
        Ec = E_mat - E_mat.mean(0)
        sd_ = np.sqrt((Ec**2).sum(0)); sd_[sd_ == 0] = 1
        R = (Ec.T @ Ec) / np.outer(sd_, sd_)
        S_alg, _ = hier_cluster_prune(A_star, R, names)

        # diagnosis
        Xdom = np.vstack([Xl, Xw])
        ydom = np.concatenate([np.zeros(len(Xl)), np.ones(len(Xw))])
        sw = np.where(ydom == 1, 1.0/(ydom==1).sum(), 1.0/(ydom==0).sum())
        sw = sw/sw.sum()*len(ydom)
        m_dom = fit_lr_np_weighted(Xdom, ydom, sw)
        w_raw = predict_m(m_dom, Xl)
        w_lab = np.clip(w_raw / np.clip(1 - w_raw, 1e-12, None), 1/KAPPA, KAPPA)

        f_V = fit_lr(Xl.tolist(), yl.tolist(), lam=LAM_2, lr=0.05, iters=800)
        p_full = np.array(predict_lr(f_V, Xw.tolist()))
        G_con = ece10(p_full, yw)

        tier1_enable = (G_con < TAU_C)

        if not tier1_enable:
            print(f"{ds}: concept risk {G_con:.4f} >= tau_C -> Tier 2 declines acquisition")
            f0 = fit_lr(Xl[:, S_alg].tolist(), yl.tolist(), lam=LAM_2, lr=0.05, iters=800)
            auc0 = roc_auc(predict_lr(f0, Xw[:, S_alg].tolist()), yw.tolist())
            print(f"   retains MVS-LR: {auc0:.4f}")
            continue

        # inherited state from Tier 1 (enable => full valid representation)
        f_cur = fit_lr_np_weighted(Xl, yl, w_lab)
        print(f"{ds}: concept risk {G_con:.4f} < tau_C -> acquire (B={BUDGET}, {ROUNDS} rounds)")

        aucs = []
        for sd_i in range(N_SEEDS):
            seed = 7 + sd_i * 13
            labeled = np.array([], dtype=int); avail = np.arange(len(yw))
            per = BUDGET // ROUNDS
            m_cur = f_cur
            for r in range(ROUNDS):
                s = np.array(predict_m(m_cur, Xw[avail]))
                take = avail[np.argsort(np.abs(s - 0.5))[:per]]
                labeled = np.concatenate([labeled, take]); avail = np.setdiff1d(avail, take)
                lab_ix = stratified_lab(yl, min(2 * len(labeled), len(Xl)), seed + r)
                Xmix = np.vstack([Xw[labeled], Xl[lab_ix]])
                ymix = np.concatenate([yw[labeled], yl[lab_ix]])
                wmix = np.concatenate([np.ones(len(labeled)), w_lab[lab_ix]])
                m_cur = fit_lr_np_weighted(Xmix, ymix, wmix)
            s_final = np.array(predict_m(m_cur, Xw[avail]))
            aucs.append(roc_auc(s_final.tolist(), yw[avail].tolist()))
        aucs = np.array(aucs)
        print(f"   Tier 2: {aucs.mean():.4f} +/- {aucs.std():.4f}  (seeds: {np.round(aucs,4).tolist()})")

if __name__ == "__main__":
    main()
