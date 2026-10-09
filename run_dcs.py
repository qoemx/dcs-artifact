# -*- coding: utf-8 -*-
"""
DCS reference implementation:
MVS-LR construction, drift diagnosis, and Tier-1 adaptation.
Run from the repository root:  python3 run_dcs.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stacking import build_dataset, fit_lr, predict_lr, roc_auc

TAU_A, LAM_D, TAU_D, TAU_C, TAU_COV, KAPPA = 0.55, 1.0, 0.6, 0.05, 0.65, 20.0
LAM_2 = 1e-3

def w1_dist(a, b):
    a = np.sort(np.asarray(a, float)); b = np.sort(np.asarray(b, float))
    allv = np.concatenate([a, b])
    return float(np.mean(np.abs(np.searchsorted(a, allv, 'right')/len(a)
                                - np.searchsorted(b, allv, 'right')/len(b))))

def fit_lr_np_weighted(X, y, sw, lam=LAM_2, lr=0.05, iters=800):
    """Weighted loss minimization (true weighting, not resampling)."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1
    Xs = (X - mu) / sd
    sw = np.asarray(sw, float); sw = sw / sw.mean()
    w = np.zeros(X.shape[1]); b = 0.0
    mw = np.zeros_like(w); vw = np.zeros_like(w); mb = vb = 0.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    n = len(y)
    for t in range(1, iters + 1):
        p = 1 / (1 + np.exp(-np.clip(Xs @ w + b, -40, 40)))
        d = ((p - y) * sw) / n
        gw = Xs.T @ d + lam * w; gb = d.sum()
        mb = b1*mb + (1-b1)*gb; vb = b2*vb + (1-b2)*gb*gb
        b -= lr * (mb/(1-b1**t)) / (eps + np.sqrt(vb/(1-b2**t)))
        mw = b1*mw + (1-b1)*gw; vw = b2*vw + (1-b2)*gw*gw
        w -= lr * (mw/(1-b1**t)) / (eps + np.sqrt(vw/(1-b2**t)))
    return dict(w=w, b=b, mu=mu, sd=sd)

def predict_m(m, X):
    Xs = (np.asarray(X, float) - m["mu"]) / m["sd"]
    return 1 / (1 + np.exp(-np.clip(Xs @ m["w"] + m["b"], -40, 40)))

def hier_cluster_prune(A_star, R, names):
    """Hierarchical clustering (average linkage on 1-R) + per-cluster argmax A*."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    k = len(names)
    dist = 1.0 - R
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0, None)
    Z = linkage(squareform(dist, checks=False), method="average")
    # cut height 1-tau_D (R >= tau_D in same cluster)
    labels = fcluster(Z, t=1.0 - TAU_D, criterion="distance")
    S = []
    for c in np.unique(labels):
        mem = [j for j in range(k) if labels[j] == c]
        best = max(mem, key=lambda j: A_star[j])
        S.append(best)
    return sorted(S), labels

def main():
    rows = []
    for ds in ["Ransomware", "miner2"]:
        d = build_dataset(ds)
        names = d["retained"]; k = len(names)
        Xl, yl = np.array(d["Xl"]), np.array(d["yl"])
        Xw, yw = np.array(d["Xw"]), np.array(d["yw"])

        # Phase 1 done in build_dataset (calibration + filtering)

        # ---- Phase 2: retention scores ----
        delta = np.array([w1_dist(Xl[:, j], Xw[:, j]) for j in range(k)])
        A_lab = np.array([roc_auc(Xl[:, j].tolist(), yl.tolist()) for j in range(k)])
        A_star = A_lab * np.exp(-LAM_D * delta)

        # ---- Phase 3: clustering + pruning ----
        E_mat = np.column_stack([(Xl[:, j] >= 0.5) != (yl == 1) for j in range(k)]).astype(int)
        Ec = E_mat - E_mat.mean(0)
        sd_ = np.sqrt((Ec**2).sum(0)); sd_[sd_ == 0] = 1
        R = (Ec.T @ Ec) / np.outer(sd_, sd_)
        S_alg, labels = hier_cluster_prune(A_star, R, names)
        S_full = list(range(k))

        # ---- Phase 4: backbone (pruned vs full) ----
        f_prune = fit_lr(Xl[:, S_alg].tolist(), yl.tolist(), lam=LAM_2, lr=0.05, iters=800)
        auc_prune = roc_auc(predict_lr(f_prune, Xw[:, S_alg].tolist()), yw.tolist())
        f_full = fit_lr(Xl.tolist(), yl.tolist(), lam=LAM_2, lr=0.05, iters=800)
        auc_full = roc_auc(predict_lr(f_full, Xw.tolist()), yw.tolist())

        print(f"\n=== {ds} ===")
        print(f"  clustering: {len(set(labels))} cluster → keep {[names[j] for j in S_alg]}")
        print(f"  backbone full: {auc_full:.4f} | pruned: {auc_prune:.4f} (Δ{auc_prune-auc_full:+.4f})")

        # ---- Phase 5: diagnosis ----
        Zl = Xl; Zw = Xw
        Xdom = np.vstack([Zl, Zw])
        ydom = np.concatenate([np.zeros(len(Zl)), np.ones(len(Zw))])
        # balanced discriminator (equal priors)
        sw_dom = np.where(ydom == 1, 1.0/max((ydom==1).sum(),1), 1.0/max((ydom==0).sum(),1))
        sw_dom = sw_dom / sw_dom.sum() * len(ydom)
        m_dom = fit_lr_np_weighted(Xdom, ydom, sw_dom)
        p_dom = predict_m(m_dom, Xdom)
        # 5-fold CV AUC
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(Xdom))
        aucs_dom = []
        for f in range(5):
            te = perm[f::5]
            pte = predict_m(m_dom, Xdom[te])  # full-fit CV approximation
            aucs_dom.append(roc_auc(pte.tolist(), ydom[te].tolist()))
        G_cov = float(np.mean(aucs_dom))
        # ECE on full wild (retrospective, paper protocol)
        p0 = np.array(predict_lr(f_full, Xw.tolist()))
        edges = np.linspace(0, 1, 11); ece = 0.0
        for i in range(10):
            m_ = (p0 >= edges[i]) & (p0 < edges[i+1]) if i < 9 else (p0 >= edges[9]) & (p0 <= edges[10])
            if m_.sum(): ece += m_.mean() * abs(yw[m_].mean() - p0[m_].mean())
        G_con = float(ece)

        # ---- Tier 1: conditional IW ----
        tier1_open = (G_con < TAU_C) and (G_cov > TAU_COV)
        auc_t1 = auc_full
        t1_action = "freeze"
        if tier1_open:
            w_raw = predict_m(m_dom, Xl)  # D(x) on lab
            w = np.clip(w_raw / np.clip(1 - w_raw, 1e-12, None), 1/KAPPA, KAPPA)
            f_iw = fit_lr_np_weighted(Xl, yl, w)
            auc_t1 = roc_auc(predict_m(f_iw, Xw).tolist(), yw.tolist())
            t1_action = "enable IW"

        print(f"  G_cov={G_cov:.3f} G_con={G_con:.3f} → Tier1: {t1_action}")
        print(f"  Tier1 AUC: {auc_t1:.4f} (vs backbone {auc_full:+.4f})")

        rows.append((ds, auc_full, auc_prune, G_cov, G_con, t1_action, auc_t1))

    print("\n===== Summary =====")
    print(f"{'dataset':<12} {'backbone':>9} {'pruned':>9} {'Gcov':>6} {'Gcon':>6} {'Tier1':>8} {'T1 AUC':>8}")
    for r in rows:
        print(f"{r[0]:<12} {r[1]:>9.4f} {r[2]:>9.4f} {r[3]:>6.3f} {r[4]:>6.3f} "
              f"{('IW' if 'IW' in r[5] else 'freeze'):>8} {r[6]:>8.4f}")

if __name__ == "__main__":
    main()
