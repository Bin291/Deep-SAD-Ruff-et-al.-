"""Collect Step-5 results into report tables (markdown + CSV).

Every cell: mean ± std [min, max] over the 10 runs, in %.
Families: "lớp độc hại (chuẩn)" = P/R/Acc/F1 of the malicious class;
"weighted-avg (chẩn đoán)" = per-class metrics weighted by support.

Usage: python src/make_tables.py -> results/tables.md, results/tables/*.csv
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("RUNS_DIR", ROOT / "results" / "runs"))
OUT = Path(os.environ.get("TABLES_DIR", ROOT / "results" / "tables"))
TABLES_MD = Path(os.environ.get("TABLES_MD", ROOT / "results" / "tables.md"))
PAPER = {"AE": {"precision": 90, "recall": 95, "accuracy": 95, "f1": 92},
         "VAE": {"precision": 92, "recall": 96, "accuracy": 96, "f1": 94}}
STD = ["precision", "recall", "accuracy", "f1"]
WAVG = ["w_precision", "w_recall", "accuracy", "w_f1"]


def fmt(v, pct=True, nd=1):
    v = np.asarray([x for x in v if x is not None], float)
    if len(v) == 0 or np.all(np.isnan(v)):
        return "—"
    k = 100 if pct else 1
    return f"{k*np.nanmean(v):.{nd}f} ± {k*np.nanstd(v):.{nd}f} [{k*np.nanmin(v):.{nd}f}, {k*np.nanmax(v):.{nd}f}]"


def short(v, pct=True, nd=1):
    v = np.asarray(v, float)
    k = 100 if pct else 1
    return "—" if np.all(np.isnan(v)) else f"{k*np.nanmean(v):.{nd}f} ± {k*np.nanstd(v):.{nd}f}"


def load_runs():
    out = {}
    for f in sorted(RUNS.glob("*.json")):
        d = json.loads(f.read_text())
        out[d["name"]] = d
    return out


def row_metrics(runs, getter, thr):
    """Metrics of one evaluation (dict with T1/T2 sub-dicts) across runs."""
    ev = [getter(r) for r in runs]
    r = {}
    for m in STD + ["w_precision", "w_recall", "w_f1"]:
        r[m] = [e[thr][m] for e in ev]
    r["roc_auc"] = [e["roc_auc"] for e in ev]
    r["pr_auc"] = [e["pr_auc"] for e in ev]
    r["positives"] = [e["positives"] for e in ev]
    return r


def confusion_rows(runs, getter):
    ev = [getter(r) for r in runs]
    return {m: [e[m] for e in ev] for m in STD + ["w_precision", "w_recall", "w_f1"]}


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def rows_table(R, B, rel, ls, feat, proto, long=True):
    """Model rows (R1/R2 x AE/VAE x T1/T2) + baseline rows for one P1/P1-bal/P2 config."""
    f = fmt if long else short
    rows = []
    key = f"r{rel}_{ls}_{feat}_{proto}"
    for model in ["AE", "VAE"]:
        for reading in ["R1", "R2"]:
            name = f"{key}_{model}_{reading}"
            if name not in R:
                continue
            runs = R[name]["runs"]
            for thr in ["T1", "T2"]:
                m = row_metrics(runs, lambda r: r["test"], thr)
                rows.append({"Mô hình": f"{model} {reading} {thr}",
                             **{f"{k.capitalize()} (chuẩn)": f(m[k]) for k in STD},
                             **{f"w{k[2:].capitalize()} (chẩn đoán)": f(m[k]) for k in ["w_precision", "w_recall", "w_f1"]},
                             "AUC": f(m["roc_auc"], pct=False, nd=3), "PR-AUC": f(m["pr_auc"], pct=False, nd=3)})
    if key in B:
        b = B[key]
        for label, get in [("Gắn cờ không ai", lambda x: x["flag_nobody"]), ("Gắn cờ tất cả", lambda x: x["flag_all"])]:
            m = confusion_rows(b, get)
            rows.append({"Mô hình": label, **{f"{k.capitalize()} (chuẩn)": f(m[k]) for k in STD},
                         **{f"w{k[2:].capitalize()} (chẩn đoán)": f(m[k]) for k in ["w_precision", "w_recall", "w_f1"]},
                         "AUC": "0.500" if "tất cả" in label else "—", "PR-AUC": "—"})
        for label, sc, thr in [("Bảng tra T2", "lookup", "T2"), ("Tần suất nghịch T2", "inv_freq", "T2"),
                               ("Trần oracle (in-sample test)", "oracle", "best")]:
            m = row_metrics(b, lambda x: x[sc], thr)
            rows.append({"Mô hình": label, **{f"{k.capitalize()} (chuẩn)": f(m[k]) for k in STD},
                         **{f"w{k[2:].capitalize()} (chẩn đoán)": f(m[k]) for k in ["w_precision", "w_recall", "w_f1"]},
                         "AUC": f(m["roc_auc"], pct=False, nd=3), "PR-AUC": f(m["pr_auc"], pct=False, nd=3)})
    return pd.DataFrame(rows)


def test_info(R, B, key):
    """Positives per test set (rows and users)."""
    src = B.get(key) or next((R[n]["runs"] for n in R if n.startswith(key + "_")), None)
    if not src:
        return ""
    pos = [x["info"]["test_pos"] for x in src]
    rows = [x["info"]["test_rows"] for x in src]
    users = [x["info"].get("test_pos_users", np.nan) for x in src]
    return (f"Tập test: {np.mean(rows):,.0f} dòng; dòng dương {np.mean(pos):,.0f} "
            f"[{np.min(pos):,}, {np.max(pos):,}] ({100*np.mean(pos)/np.mean(rows):.2f}%); "
            f"số user dương trong test {np.nanmean(users):.0f} [{np.nanmin(users):.0f}, {np.nanmax(users):.0f}].")


def p3_table(R, P3C, rel, ls, feat, unit, long=True):
    f = fmt if long else short
    rows = []
    key = f"r{rel}_{ls}_{feat}_P3"
    for model in ["AE", "VAE"]:
        for reading in ["R1", "R2"]:
            name = f"{key}_{model}_{reading}"
            if name not in R:
                continue
            for rule in ["max", "mean"]:
                for thr in ["T1", "T2"]:
                    m = row_metrics(R[name]["runs"], lambda r: r[f"{unit}_{rule}"], thr)
                    rows.append({"Mô hình": f"{model} {reading} {rule} {thr}",
                                 **{f"{k.capitalize()} (chuẩn)": f(m[k]) for k in STD},
                                 **{f"w{k[2:].capitalize()} (chẩn đoán)": f(m[k]) for k in ["w_precision", "w_recall", "w_f1"]},
                                 "AUC": f(m["roc_auc"], pct=False, nd=3), "PR-AUC": f(m["pr_auc"], pct=False, nd=3)})
    ck = f"r{rel}_{ls}_{feat}"
    if ck in P3C:
        b = P3C[ck]
        for label, get in [("Gắn cờ không ai", lambda x: x["oracle"][f"{unit}_flag_nobody"]),
                           ("Gắn cờ tất cả", lambda x: x["oracle"][f"{unit}_flag_all"])]:
            m = confusion_rows(b, get)
            rows.append({"Mô hình": label, **{f"{k.capitalize()} (chuẩn)": f(m[k]) for k in STD},
                         **{f"w{k[2:].capitalize()} (chẩn đoán)": f(m[k]) for k in ["w_precision", "w_recall", "w_f1"]},
                         "AUC": "—", "PR-AUC": "—"})
        for rule in ["max", "mean"]:
            for label, sc, thr in [(f"Bảng tra {rule} T2", "lookup", "T2"), (f"Tần suất nghịch {rule} T2", "inv_freq", "T2"),
                                   (f"Trần oracle {rule} (xấp xỉ)", "oracle", "best")]:
                m = row_metrics(b, lambda x: x[sc][f"{unit}_{rule}"], thr)
                rows.append({"Mô hình": label, **{f"{k.capitalize()} (chuẩn)": f(m[k]) for k in STD},
                             **{f"w{k[2:].capitalize()} (chẩn đoán)": f(m[k]) for k in ["w_precision", "w_recall", "w_f1"]},
                             "AUC": f(m["roc_auc"], pct=False, nd=3), "PR-AUC": f(m["pr_auc"], pct=False, nd=3)})
    return pd.DataFrame(rows)


def p3_info(P3C, rel, ls, feat):
    ck = f"r{rel}_{ls}_{feat}"
    if ck not in P3C:
        return ""
    i = [x["info"] for x in P3C[ck]]
    g = lambda k: (np.mean([x[k] for x in i]), np.min([x[k] for x in i]), np.max([x[k] for x in i]))
    u, pu, d, pd_ = g("test_users"), g("test_pos_users"), g("test_user_days"), g("test_pos_user_days")
    return (f"Mỗi fold test: {u[0]:.0f} user, trong đó {pu[0]:.0f} insider [{pu[1]:.0f}, {pu[2]:.0f}]; "
            f"{d[0]:,.0f} user-day, user-day dương {pd_[0]:.0f} [{pd_[1]:.0f}, {pd_[2]:.0f}].")


def paper_diff(R):
    rows = []
    for model in ["AE", "VAE"]:
        for reading in ["R1", "R2"]:
            name = f"r4.2_main_F38_P1_{model}_{reading}"
            if name not in R:
                continue
            for thr in ["T1", "T2"]:
                m = row_metrics(R[name]["runs"], lambda r: r["test"], thr)
                row = {"Mô hình": f"{model} {reading} {thr}"}
                for k, wk in zip(STD, WAVG):
                    p = PAPER[model][k]
                    row[f"{k.capitalize()}: bài"] = p
                    row[f"chuẩn − bài"] = None
                    row[f"{k.capitalize()} chuẩn − bài"] = f"{100*np.mean(m[k]) - p:+.1f}"
                    row[f"{k.capitalize()} w-avg − bài"] = f"{100*np.mean(m[wk]) - p:+.1f}"
                row.pop("chuẩn − bài")
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    R = load_runs()
    B = json.loads((ROOT / "results" / "baselines5.json").read_text())
    P3C = json.loads((ROOT / "results" / "p3_ceiling.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# Bảng kết quả Bước 5", "",
          "Mỗi ô: trung bình ± std [min, max] qua 10 lần chạy, đơn vị %. "
          "**(chuẩn)** = lớp độc hại là lớp dương; **(chẩn đoán)** = weighted-avg theo số mẫu mỗi lớp (như sklearn), chỉ để đối chiếu cách bài có thể đã tính. "
          "T1 = ngưỡng theo tỉ lệ dương của train; T2 = ngưỡng F1 tốt nhất trên validation. "
          "Baseline và trần tính trên đúng các tập test đó.", ""]
    # main: P1, P2 (F38, main) for both releases; P1-bal; P3
    for rel in ["4.2", "6.2"]:
        for proto in ["P1", "P2", "P1-bal"]:
            t = rows_table(R, B, rel, "main", "F38", proto)
            t.to_csv(OUT / f"r{rel}_main_F38_{proto}.csv", index=False)
            md += [f"## r{rel} · {proto} · insider main · F38 (nhãn mức user, đơn vị dòng)", "",
                   test_info(R, B, f"r{rel}_main_F38_{proto}"), "", md_table(t), ""]
        for unit, uname in [("ud", "user-day (nhãn cửa sổ)"), ("user", "user (insider)")]:
            t = p3_table(R, P3C, rel, "main", "F38", unit)
            t.to_csv(OUT / f"r{rel}_main_F38_P3_{unit}.csv", index=False)
            md += [f"## r{rel} · P3 · đơn vị {uname} · insider main · F38", "", p3_info(P3C, rel, "main", "F38"), "", md_table(t), ""]
    d = paper_diff(R)
    d.to_csv(OUT / "paper_diff_P1_r4.2.csv", index=False)
    md += ["## So với bài: P1 / r4.2 / main / F38 (hiệu = ours − bài, điểm %)", "", md_table(d), ""]
    # variants: compact F1 / AUC tables
    var_rows = []
    for rel, ls, feat in [("4.2", "V30a", "F38"), ("4.2", "V30b", "F38"), ("4.2", "main", "F50"), ("6.2", "main", "F50")]:
        for proto in ["P1", "P1-bal", "P2", "P3"]:
            for model in ["AE", "VAE"]:
                for reading in ["R1", "R2"]:
                    name = f"r{rel}_{ls}_{feat}_{proto}_{model}_{reading}"
                    if name not in R:
                        continue
                    get = (lambda r: r["test"]) if proto != "P3" else (lambda r: r["ud_max"])
                    m = row_metrics(R[name]["runs"], get, "T2")
                    var_rows.append({"Biến thể": f"r{rel} {ls} {feat}", "Giao thức": proto + (" (ud, max)" if proto == "P3" else ""),
                                     "Mô hình": f"{model} {reading}", "F1 chuẩn T2": fmt(m["f1"]), "wF1 T2": fmt(m["w_f1"]),
                                     "Acc T2": fmt(m["accuracy"]), "AUC": fmt(m["roc_auc"], pct=False, nd=3)})
    v = pd.DataFrame(var_rows)
    v.to_csv(OUT / "variants.csv", index=False)
    md += ["## Biến thể (T2; P3 ở đơn vị user-day, gộp max)", "", md_table(v), ""]
    for rel, ls, feat in [("4.2", "V30a", "F38"), ("4.2", "V30b", "F38"), ("4.2", "main", "F50"), ("6.2", "main", "F50")]:
        for proto in ["P1", "P1-bal", "P2"]:
            if f"r{rel}_{ls}_{feat}_{proto}" in B:
                t = rows_table(R, B, rel, ls, feat, proto, long=False)
                t.to_csv(OUT / f"r{rel}_{ls}_{feat}_{proto}.csv", index=False)
        for unit in ["ud", "user"]:
            t = p3_table(R, P3C, rel, ls, feat, unit, long=False)
            t.to_csv(OUT / f"r{rel}_{ls}_{feat}_P3_{unit}.csv", index=False)
    # health: finite + loss decrease
    health = []
    for n, d in R.items():
        for k, v in d["loss_curve_mean"].items():
            health.append({"config": n, "stage": k, "finite": d["finite"], "loss_first": round(v[0], 4), "loss_last": round(v[-1], 4),
                           "decreased": v[-1] < v[0], "train_group_s": d["seconds"]["train_group"]})
    h = pd.DataFrame(health)
    h.to_csv(OUT / "training_health.csv", index=False)
    md += ["## Sức khoẻ huấn luyện", "",
           f"{len(R)} cấu hình; non-finite: {int((~h['finite']).sum())}; giai đoạn loss không giảm: "
           f"{', '.join(h.loc[~h['decreased'], 'config'] + '/' + h.loc[~h['decreased'], 'stage']) or 'không có'}.", ""]
    TABLES_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"{len(R)} configs -> {TABLES_MD}")


if __name__ == "__main__":
    main()
