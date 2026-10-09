# -*- coding: utf-8 -*-
"""
Stacking pipeline: calibration + baselines + LR stacking
Trained on lab labels only; wild data is used for evaluation only.
All methods are evaluated on the common-coverage intersection of retained classifiers.
"""
import os, math, bisect, csv, time, importlib.util

BASE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ev", os.path.join(BASE, "eval_single.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

VALID_AUC = 0.55   # drop classifiers below this lab AUC

# ---------- numeric utilities ----------
def mean(a): return sum(a) / len(a) if a else 0.0
def std(a):
    if len(a) < 2: return 0.0
    m = mean(a); return math.sqrt(sum((x - m) ** 2 for x in a) / len(a))
def sigmoid(z): return 1.0 / (1.0 + math.exp(-z)) if z > -40 else 0.0 if z < -40 else 1.0

def roc_auc(scores, labels):
    return ev.roc_auc(scores, labels)

def metrics_at(scores, labels, thr):
    tp = fp = tn = fn = 0
    for s, y in zip(scores, labels):
        p = 1 if s >= thr else 0
        if y == 1 and p == 1: tp += 1
        elif y == 0 and p == 1: fp += 1
        elif y == 0 and p == 0: tn += 1
        else: fn += 1
    n = tp + fp + tn + fn or 1
    acc = (tp + tn) / n
    bacc = (tp / (tp + fn) + tn / (tn + fp)) / 2 if (tp + fn) and (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) else 0.0
    return acc, bacc, prec, tpr, f1

def best_threshold(probs, labels, step=0.02):
    best_t, best_f1 = 0.5, -1
    t = 0.1
    while t <= 0.9:
        _, _, _, _, f1 = metrics_at(probs, labels, t)
        if f1 > best_f1: best_f1, best_t = f1, t
        t += step
    return best_t

# ---------- isotonic calibration (PAVA) ----------
def pava_blocks(y):
    blocks = []
    for yi in y:
        blocks.append([1, yi])
        while len(blocks) >= 2 and blocks[-2][1] / blocks[-2][0] > blocks[-1][1] / blocks[-1][0] + 1e-15:
            b2 = blocks.pop(); b1 = blocks.pop()
            blocks.append([b1[0] + b2[0], b1[1] + b2[1]])
    return [(s, v / s) for s, v in blocks]

def isotonic_fit(x, y):
    idx = sorted(range(len(x)), key=lambda i: x[i])
    xs = [x[i] for i in idx]; ys = [y[i] for i in idx]
    blocks = pava_blocks(ys); fitted = []
    for sz, mv in blocks: fitted.extend([mv] * sz)
    def pred(c):
        if not xs or not math.isfinite(c): return 0.5
        if c <= xs[0]: return fitted[0]
        if c >= xs[-1]: return fitted[-1]
        i = bisect.bisect_right(xs, c)
        x0, x1 = xs[i - 1], xs[i]; y0, y1 = fitted[i - 1], fitted[i]
        return y0 if x1 == x0 else y0 + (y1 - y0) * (c - x0) / (x1 - x0)
    return pred

# ---------- logistic regression (Adam, L2) ----------
def fit_lr(X, y, lam=1e-3, lr=0.05, iters=800):
    n = len(X); k = len(X[0])
    mu = [mean([X[i][j] for i in range(n)]) for j in range(k)]
    sd = [std([X[i][j] for i in range(n)]) or 1.0 for j in range(k)]
    Xs = [[(X[i][j] - mu[j]) / sd[j] for j in range(k)] for i in range(n)]
    w = [0.0] * k; b = 0.0
    mw = [0.0] * k; mb = 0.0; vw = [0.0] * k; vb = 0.0; b1 = 0.9; b2 = 0.999; eps = 1e-8
    for t in range(1, iters + 1):
        gw = [0.0] * k; gb = 0.0
        for i in range(n):
            z = b + sum(w[j] * Xs[i][j] for j in range(k))
            p = sigmoid(z); d = (p - y[i]) / n
            gb += d
            xi = Xs[i]
            for j in range(k): gw[j] += d * xi[j]
        for j in range(k): gw[j] += lam * w[j]   # L2
        # Adam
        mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb * gb
        b -= lr * (mb / (1 - b1 ** t)) / (eps + math.sqrt(vb / (1 - b2 ** t)))
        for j in range(k):
            mw[j] = b1 * mw[j] + (1 - b1) * gw[j]; vw[j] = b2 * vw[j] + (1 - b2) * gw[j] * gw[j]
            w[j] -= lr * (mw[j] / (1 - b1 ** t)) / (eps + math.sqrt(vw[j] / (1 - b2 ** t)))
    return {"w": w, "b": b, "mu": mu, "sd": sd}

def predict_lr(m, X):
    out = []
    for x in X:
        z = m["b"] + sum(m["w"][j] * (x[j] - m["mu"][j]) / m["sd"][j] for j in range(len(x)))
        out.append(sigmoid(z))
    return out

# ---------- regression tree (CART, depth-limited) ----------
def fit_tree(X, y, depth=3, min_leaf=40, n_thresh=12):
    n = len(X); idx = list(range(n))
    cols = [[X[i][j] for i in range(n)] for j in range(len(X[0]))]

    def sse_of(ids):
        if not ids: return 0.0
        m = mean([y[i] for i in ids])
        return sum((y[i] - m) ** 2 for i in ids)

    def build(ids, d):
        val = mean([y[i] for i in ids])
        if d == 0 or len(ids) < 2 * min_leaf:
            return {"leaf": True, "val": val}
        cur = sse_of(ids); best = None
        for j in range(len(X[0])):
            vals = sorted(set(cols[j][i] for i in ids))
            if len(vals) < 2: continue
            step = max(1, len(vals) // n_thresh)
            qs = vals[::step]
            for q in qs[1:]:
                L = [i for i in ids if cols[j][i] <= q]; R = [i for i in ids if cols[j][i] > q]
                if len(L) < min_leaf or len(R) < min_leaf: continue
                s = sse_of(L) + sse_of(R)
                if best is None or s < best[0]:
                    best = (s, j, q, L, R)
        if best is None or best[0] >= cur - 1e-9:
            return {"leaf": True, "val": val}
        _, j, q, L, R = best
        return {"leaf": False, "f": j, "t": q, "l": build(L, d - 1), "r": build(R, d - 1)}

    tree = build(idx, depth)

    def pred(x):
        node = tree
        while not node["leaf"]:
            node = node["l"] if x[node["f"]] <= node["t"] else node["r"]
        return node["val"]
    return pred

# ---------- gradient boosting (log-loss) ----------
def fit_gbm(X, y, n_trees=40, lr=0.1, depth=3, min_leaf=40, n_thresh=12):
    py = mean(y); F0 = math.log(py / (1 - py)) if 0 < py < 1 else 0.0
    n = len(X); F = [F0] * n; trees = []
    for _ in range(n_trees):
        p = [sigmoid(f) for f in F]; g = [y[i] - p[i] for i in range(n)]
        tree = fit_tree(X, g, depth, min_leaf, n_thresh)
        for i in range(n): F[i] += lr * tree(X[i])
        trees.append(tree)
    def predict(Xq):
        out = []
        for x in Xq:
            f = F0 + lr * sum(t(x) for t in trees); out.append(sigmoid(f))
        return out
    return predict

# ---------- data pipeline ----------
def build_canon(gt_path):
    """Map any id form (filename/hash) to canonical hash; canonical -> label."""
    key2c = {}; c2l = {}
    for row in csv.DictReader(open(gt_path, encoding="utf-8-sig")):
        lab = int(row["label"])
        canon = (row.get("hash") or row.get("filename") or "").strip()
        if not canon: continue
        c2l[canon] = lab
        for col in ("filename", "hash"):
            v = row.get(col)
            if v: key2c[v.strip()] = canon
    return key2c, c2l


def build_dataset(ds):
    cfg = ev.DATASETS[ds]
    kl2c, lab_c2l = build_canon(os.path.join(BASE, cfg["lab_gt"]))
    kw2c, wild_c2l = build_canon(os.path.join(BASE, cfg["wild_gt"]))
    clfs = []
    for m, labf, wildf in cfg["methods"]:
        rl_raw, _, _ = ev.load_result(os.path.join(BASE, cfg["lab_dir"], labf))
        rw_raw, _, _ = ev.load_result(os.path.join(BASE, cfg["wild_dir"], wildf))
        # normalize to canonical id
        rl = {kl2c[i]: v for i, v in rl_raw.items() if i in kl2c}
        rw = {kw2c[i]: v for i, v in rw_raw.items() if i in kw2c}
        # lab calibration + AUC
        lab_ids = [i for i in rl if i in lab_c2l and math.isfinite(rl[i])]
        scL = [rl[i] for i in lab_ids]; yL = [lab_c2l[i] for i in lab_ids]
        aucL = ev.roc_auc(scL, yL) if len(set(yL)) > 1 else 0.5
        cal = isotonic_fit(scL, yL)
        clfs.append({"m": m, "rl": rl, "rw": rw, "cal": cal, "aucL": aucL})
    retained = [c for c in clfs if c["aucL"] >= VALID_AUC]
    dropped = [c["m"] for c in clfs if c["aucL"] < VALID_AUC]
    # canonical intersection sample set
    common_lab = set(i for i in retained[0]["rl"] if math.isfinite(retained[0]["rl"][i]))
    for c in retained[1:]:
        common_lab &= set(i for i in c["rl"] if math.isfinite(c["rl"][i]))
    common_lab &= set(lab_c2l)
    common_wild = set(i for i in retained[0]["rw"] if math.isfinite(retained[0]["rw"][i]))
    for c in retained[1:]:
        common_wild &= set(i for i in c["rw"] if math.isfinite(c["rw"][i]))
    common_wild &= set(wild_c2l)
    common_lab = sorted(common_lab); common_wild = sorted(common_wild)

    def matrix(ids, c2l, domain):
        X, y, per_clf_cal = [], [], []
        for c in retained:
            src = c["rl"] if domain == "lab" else c["rw"]
            per_clf_cal.append([c["cal"](src[i]) for i in ids])
        for r in range(len(ids)):
            X.append([per_clf_cal[j][r] for j in range(len(retained))])
            y.append(c2l[ids[r]])
        return X, y, per_clf_cal

    Xl, yl, _ = matrix(common_lab, lab_c2l, "lab")
    Xw, yw, calw = matrix(common_wild, wild_c2l, "wild")
    return {
        "retained": [c["m"] for c in retained], "dropped": dropped,
        "aucL": {c["m"]: c["aucL"] for c in retained},
        "n_lab": len(common_lab), "n_wild": len(common_wild),
        "Xl": Xl, "yl": yl, "Xw": Xw, "yw": yw, "calw": calw,
    }


def run():
    results = []
    for ds in ["Ransomware", "miner1", "miner2"]:
        t0 = time.time()
        d = build_dataset(ds)
        k = len(d["retained"])
        print("=" * 100)
        print(f"dataset {ds}：keep {k} classifiers {d['retained']}")
        if d["dropped"]: print(f"  drop(lab-AUC<{VALID_AUC}): {d['dropped']}")
        print(f"  intersection: lab={d['n_lab']}  wild={d['n_wild']}  ({time.time()-t0:.1f}s)")
        Xl, yl, Xw, yw = d["Xl"], d["yl"], d["Xw"], d["yw"]
        calw = d["calw"]   # list[k] of list[n_wild] calibrationsplit

        # --- baseline: best single classifier (highest lab AUC) ---
        best = max(d["retained"], key=lambda m: d["aucL"][m])
        bi = d["retained"].index(best)
        best_lab = [Xl[r][bi] for r in range(len(yl))]      # calibrated lab scoresumn)
        best_wild = calw[bi]                                 # calibrated wild scoresnsplit
        # --- baseline: mean fusion ---
        mean_lab = [mean([Xl[r][j] for j in range(k)]) for r in range(len(yl))]
        mean_wild = [mean([calw[j][r] for j in range(k)]) for r in range(len(yw))]
        # --- baseline: majority vote ---
        vote_lab = [mean([1.0 if Xl[r][j] >= 0.5 else 0.0 for j in range(k)]) for r in range(len(yl))]
        vote_wild = [mean([1.0 if calw[j][r] >= 0.5 else 0.0 for j in range(k)]) for r in range(len(yw))]

        # --- meta-learner: trained on lab ---
        t1 = time.time(); lr_m = fit_lr(Xl, yl); print(f"  LR train {time.time()-t1:.1f}s")
        t1 = time.time(); gbm_pred = fit_gbm(Xl, yl); print(f"  GBM train {time.time()-t1:.1f}s")
        lr_lab = predict_lr(lr_m, Xl); lr_wild = predict_lr(lr_m, Xw)
        gbm_lab = gbm_pred(Xl); gbm_wild = gbm_pred(Xw)

        def eval_method(name, lab_p, wild_p):
            thr = best_threshold(lab_p, yl)
            auc_lab = roc_auc(lab_p, yl)
            auc_w = roc_auc(wild_p, yw)
            acc, bacc, prec, tpr, f1 = metrics_at(wild_p, yw, thr)
            print(f"  {name:<20} labAUC={auc_lab:.4f} | wild AUC={auc_w:.4f} bACC={bacc:.4f} "
                  f"F1={f1:.4f} TPR={tpr:.4f} PREC={prec:.4f} (thr={thr:.2f})")
            return {"dataset": ds, "method": name, "auc_lab": round(auc_lab, 4),
                    "auc_wild": round(auc_w, 4), "bacc": round(bacc, 4), "f1": round(f1, 4),
                    "tpr": round(tpr, 4), "prec": round(prec, 4)}

        results.append(eval_method(f"best_single[{best}]", best_lab, best_wild))
        results.append(eval_method("mean_fusion", mean_lab, mean_wild))
        results.append(eval_method("majority_vote", vote_lab, vote_wild))
        results.append(eval_method("stack_LR", lr_lab, lr_wild))
        results.append(eval_method("stack_GBM", gbm_lab, gbm_wild))

    out = os.path.join(BASE, "ensemble_results.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)
    print(f"\n[done] results written to {out}")
    return results


if __name__ == "__main__":
    run()
