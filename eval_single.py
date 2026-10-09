# -*- coding: utf-8 -*-
"""
单机器学习分类器效果评估
三个数据集 (Ransomware / miner1 / miner2) x 两个划分 (lab / wild) x 9 种方法
每个方法文件含 confidence = P(malware) = P(label=1), 阈值 0.5
"""
import csv
import os
import math
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
THRESH = 0.5


# ---------- IO helpers ----------
def load_gt(path):
    """真值 -> {id: label(int)}，同时把 filename 和 hash 两种形式都映射进去。"""
    key2label = {}
    n = 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            label = int(row["label"])
            # 可能的标识列
            for col in ("filename", "hash"):
                v = row.get(col)
                if v:
                    v = v.strip()
                    if v in key2label and key2label[v] != label:
                        # 同一标识不同标签：保留首次，但记录冲突
                        pass
                    else:
                        key2label[v] = label
            n += 1
    return key2label, n


def load_result(path):
    """结果文件 -> {id: confidence}。confidence 列固定；id 列优先 filename 否则 hash。
    若同一 id 多次出现(如 fold)，取平均。返回 (dict, 总行数, 重复id信息)。"""
    conf_sum = defaultdict(float)
    conf_cnt = defaultdict(int)
    total = 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames
        # 选 id 列
        if "filename" in fields:
            idcol = "filename"
        elif "hash" in fields:
            idcol = "hash"
        else:
            idcol = fields[0]
        assert "confidence" in fields, f"{path} 缺 confidence 列: {fields}"
        for row in r:
            cid = row[idcol].strip()
            cstr = row["confidence"].strip()
            try:
                conf = float(cstr)
            except ValueError:
                continue
            conf_sum[cid] += conf
            conf_cnt[cid] += 1
            total += 1
    out = {k: conf_sum[k] / conf_cnt[k] for k in conf_sum}
    dup = {k: v for k, v in conf_cnt.items() if v > 1}
    return out, total, dup


# ---------- metrics ----------
def roc_auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    npos, nneg = len(pos), len(neg)
    if npos == 0 or nneg == 0:
        return float("nan")
    # 合并排序求平均秩(处理同分)
    allv = pos + neg
    order = sorted(range(len(allv)), key=lambda i: allv[i])
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and allv[order[j + 1]] == allv[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0  # 1-indexed 平均秩
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos_ranks = sum(ranks[idx] for idx in range(len(allv)) if idx < npos)
    auc = (sum_pos_ranks - npos * (npos + 1) / 2.0) / (npos * nneg)
    return auc


def evaluate(scores, labels, thresh=THRESH):
    tp = fp = tn = fn = 0
    for s, y in zip(scores, labels):
        pred = 1 if s >= thresh else 0
        if y == 1 and pred == 1:
            tp += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 0 and pred == 0:
            tn += 1
        else:
            fn += 1
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0  # TPR / 检测率
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    auc = roc_auc(scores, labels)
    return {
        "n": n, "n_pos": tp + fn, "n_neg": tn + fp,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "ACC": acc, "PREC": prec, "TPR": rec, "F1": f1, "FPR": fpr, "AUC": auc,
    }


# ---------- dataset config ----------
# 每个数据集: (lab_gt, wild_gt, [(method, lab_file, wild_file), ...])
DATASETS = {
    "Ransomware": {
        "lab_gt": "Ransomware/balanced_lab_dataset_1to2.csv",
        "wild_gt": "Ransomware/balanced_wild_dataset_1to2.csv",
        "lab_dir": "Ransomware/lab_results",
        "wild_dir": "Ransomware/real_world_results",
        "methods": [
            ("EHc", "EHc_lab_result.csv", "EHc_wild_result.csv"),
            ("EHm", "EHm_lab_result.csv", "EHm_wild_result.csv"),
            ("GI", "GI_lab_result.csv", "GI_wild_result.csv"),
            ("RB", "RB_lab_result.csv", "RB_wild_result.csv"),
            ("TIF", "TIF_lab_results.csv", "TIF_wild_result.csv"),
            ("cfg", "cfg_lab_result.csv", "cfg_wild_result.csv"),
            ("fcg", "fcg_lab_result.csv", "fcg_wild_result.csv"),
            ("opcode", "opcode_lab_dataset_test_samples_probabilities_lab_results_fold_all.csv",
                      "opcode_realworld_dataset_probabilities.csv"),
            ("pf", "pf_best_models_confidences.csv", "pf_fold_1_xgb_predictions.csv"),
        ],
    },
    "miner1": {
        "lab_gt": "miner1/train_data_gt.csv",
        "wild_gt": "miner1/miner_gt.csv",
        "lab_dir": "miner1/6000_samples_results",
        "wild_dir": "miner1/11g_results",
        "methods": [
            ("cfg", "cfg_6000_rewrite.csv", "cfg_11g_rewrite.csv"),
            ("ember_gbdt", "ember_gdbt_result_6000.csv", "ember_gdbt_11g_resut.csv"),
            ("fcg_gin", "fcg_gin_result_6000.csv", "fcg_gin_test11g_result_hejia_fix.csv"),
            ("fe_gbdt", "fe_gdbt_result_6000.csv", "fe_gbdt_test11g_resut.csv"),
            ("grayimg_cnn", "grayimg_cnn_result_6000.csv", "grayimg_cnn_test11g_result.csv"),
            ("histogram_cnn", "histogram_cnn_result_6000.csv", "histogram_cnn_test11g_result.csv"),
            ("histogram_mlp", "histogram_mlp_result_6000.csv", "histogram_mlp_test11g_result.csv"),
            ("malconv", "malconv_result_6000.csv", "malconv_6000_test11g_result.csv"),
            ("opcode_lstm", "opcode_lstm_result_6000.csv", "opcode_lstm_test11g_result.csv"),
        ],
    },
    "miner2": {
        "lab_gt": "miner2/lab_label.csv",
        "wild_gt": "miner2/wild_label.csv",
        "lab_dir": "miner2/lab",
        "wild_dir": "miner2/wild",
        "methods": [
            ("EHc", "EHc_lab.csv", "EHc_wild.csv"),
            ("EHm", "EHm_lab.csv", "EHm_wild.csv"),
            ("TIF", "TIF_lab.csv", "TIF_wild.csv"),
            ("cfg", "cfg_fold_lab_results.csv", "cfg_wild.csv"),
            ("fcg", "fcg_lab.csv", "fcg_wild.csv"),
            ("grayimage", "grayimage_lab.csv", "grayimage_wild.csv"),
            ("malconv", "malconv_lab.csv", "malconv_wild.csv"),
            ("opcode", "op_lab.csv", "op_wild.csv"),
            ("pf", "pf_lab.csv", "pf_wild.csv"),
        ],
    },
}


def join(gt, res):
    """返回 (labels, scores, 匹配数, 结果中未匹配数)"""
    labels, scores = [], []
    matched = 0
    unmatched = 0
    for cid, conf in res.items():
        if cid in gt:
            labels.append(gt[cid])
            scores.append(conf)
            matched += 1
        else:
            unmatched += 1
    return labels, scores, matched, unmatched


def main():
    rows = []  # 全部结果，用于写 CSV
    for dsname, cfg in DATASETS.items():
        print("=" * 92)
        print(f"数据集: {dsname}")
        print("=" * 92)
        lab_gt, lab_gt_n = load_gt(os.path.join(BASE, cfg["lab_gt"]))
        wild_gt, wild_gt_n = load_gt(os.path.join(BASE, cfg["wild_gt"]))
        print(f"  真值: lab={lab_gt_n} 样本 (唯一id {len(lab_gt)}), wild={wild_gt_n} 样本 (唯一id {len(wild_gt)})")
        for split, gt, gtn in (("lab", lab_gt, lab_gt_n), ("wild", wild_gt, wild_gt_n)):
            d = cfg["lab_dir"] if split == "lab" else cfg["wild_dir"]
            print(f"\n  --- 划分: {split} ({'实验室/训练域' if split=='lab' else '在野/测试域'}) ---")
            hdr = f"  {'method':<14}{'n':>7}{'POS':>6}{'NEG':>6}{'TP':>6}{'FP':>6}{'TN':>6}{'FN':>6}{'ACC':>8}{'PREC':>8}{'TPR':>8}{'F1':>8}{'FPR':>8}{'AUC':>8}"
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for m, labf, wildf in cfg["methods"]:
                fname = labf if split == "lab" else wildf
                path = os.path.join(BASE, d, fname)
                if not os.path.exists(path):
                    print(f"  {m:<14}  <缺失文件 {path}>")
                    continue
                res, total, dup = load_result(path)
                labels, scores, matched, unmatched = join(gt, res)
                if matched == 0:
                    print(f"  {m:<14}  <无匹配: 结果{total}行, 未匹配{unmatched}>")
                    continue
                mtr = evaluate(scores, labels)
                cov = matched / gtn if gtn else 0
                print(f"  {m:<14}{mtr['n']:>7}{mtr['n_pos']:>6}{mtr['n_neg']:>6}"
                      f"{mtr['TP']:>6}{mtr['FP']:>6}{mtr['TN']:>6}{mtr['FN']:>6}"
                      f"{mtr['ACC']:>8.4f}{mtr['PREC']:>8.4f}{mtr['TPR']:>8.4f}"
                      f"{mtr['F1']:>8.4f}{mtr['FPR']:>8.4f}{mtr['AUC']:>8.4f}"
                      f"   (cov={cov*100:.1f}%)" + (f"  dup={len(dup)}" if dup else ""))
                rows.append({
                    "dataset": dsname, "split": split, "method": m,
                    "n_gt": gtn, "n_result": total, "n_matched": matched,
                    "coverage": round(cov, 4),
                    "n_pos": mtr["n_pos"], "n_neg": mtr["n_neg"],
                    "TP": mtr["TP"], "FP": mtr["FP"], "TN": mtr["TN"], "FN": mtr["FN"],
                    "ACC": round(mtr["ACC"], 6), "PREC": round(mtr["PREC"], 6),
                    "TPR": round(mtr["TPR"], 6), "F1": round(mtr["F1"], 6),
                    "FPR": round(mtr["FPR"], 6), "AUC": round(mtr["AUC"], 6),
                })
        print()

    # 写汇总 CSV
    out = os.path.join(BASE, "single_classifier_metrics.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[完成] 详细结果已写入: {out}  (共 {len(rows)} 条)")


if __name__ == "__main__":
    main()
