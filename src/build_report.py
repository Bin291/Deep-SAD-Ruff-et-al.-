"""Fill the generated tables into REPORT.md (markers <!--TABLE:name-->...<!--/TABLE-->).

Numbers come straight from results/runs, results/baselines5.json and
results/p3_ceiling.json; nothing is typed by hand.

Usage: python src/build_report.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from make_tables import PAPER, STD, WAVG, load_runs, md_table

ROOT = Path(__file__).resolve().parents[1]


def ms(v, pct=True, nd=1):
    v = np.asarray(v, float)
    k = 100 if pct else 1
    return f"{k*np.nanmean(v):.{nd}f} ± {k*np.nanstd(v):.{nd}f} [{k*np.nanmin(v):.{nd}f}, {k*np.nanmax(v):.{nd}f}]"


def m_only(v, pct=True, nd=1):
    v = np.asarray(v, float)
    return f"{(100 if pct else 1)*np.nanmean(v):.{nd}f}"


def main_table(R, B, P3C, protos):
    rows = []
    for rel in ["4.2", "6.2"]:
        for proto in protos:
            base_p = proto.split()[0]
            for model in ["AE", "VAE"]:
                for reading in ["R1", "R2"]:
                    name = f"r{rel}_main_F38_{base_p}_{model}_{reading}"
                    runs = R[name]["runs"]
                    if base_p == "P3":
                        ev = proto.split()[1]
                        get = lambda r: r[ev]
                    else:
                        get = lambda r: r["test"]
                    e = [get(r) for r in runs]
                    rows.append({"Bản": f"r{rel}", "Giao thức": proto, "Mô hình": f"{model} {reading}",
                                 "F1 chuẩn T1": ms([x["T1"]["f1"] for x in e]),
                                 "F1 chuẩn T2": ms([x["T2"]["f1"] for x in e]),
                                 "Acc T2": ms([x["T2"]["accuracy"] for x in e]),
                                 "wF1 T2 (chẩn đoán)": ms([x["T2"]["w_f1"] for x in e]),
                                 "AUC": ms([x["roc_auc"] for x in e], pct=False, nd=3)})
            # baselines for this (rel, proto)
            if base_p == "P3":
                ev = proto.split()[1]
                b = P3C[f"r{rel}_main_F38"]
                unit = ev.split("_")[0]
                nob = [x["oracle"][f"{unit}_flag_nobody"] for x in b]
                look = [x["lookup"][ev]["T2"] for x in b]
                orc = [x["oracle"][ev] for x in b]
            else:
                b = B[f"r{rel}_main_F38_{base_p}"]
                nob = [x["flag_nobody"] for x in b]
                look = [x["lookup"]["T2"] for x in b]
                orc = [x["oracle"] for x in b]
            rows.append({"Bản": f"r{rel}", "Giao thức": proto, "Mô hình": "*Gắn cờ không ai*",
                         "F1 chuẩn T1": "0.0", "F1 chuẩn T2": "0.0", "Acc T2": ms([x["accuracy"] for x in nob]),
                         "wF1 T2 (chẩn đoán)": ms([x["w_f1"] for x in nob]), "AUC": "—"})
            rows.append({"Bản": f"r{rel}", "Giao thức": proto, "Mô hình": "*Bảng tra (T2)*",
                         "F1 chuẩn T1": "—", "F1 chuẩn T2": ms([x["f1"] for x in look]),
                         "Acc T2": ms([x["accuracy"] for x in look]), "wF1 T2 (chẩn đoán)": ms([x["w_f1"] for x in look]),
                         "AUC": "—"})
            rows.append({"Bản": f"r{rel}", "Giao thức": proto, "Mô hình": "*Trần oracle*" + (" (xấp xỉ)" if base_p == "P3" else ""),
                         "F1 chuẩn T1": "—", "F1 chuẩn T2": ms([x["best"]["f1"] for x in orc]),
                         "Acc T2": ms([x["best"]["accuracy"] for x in orc]), "wF1 T2 (chẩn đoán)": ms([x["best"]["w_f1"] for x in orc]),
                         "AUC": ms([x["roc_auc"] for x in orc], pct=False, nd=3)})
    return md_table(pd.DataFrame(rows))


def paper_table(R, B):
    rows = []
    for model in ["AE", "VAE"]:
        p = PAPER[model]
        rows.append({"Hàng": f"**Bài — {model}**", "P": p["precision"], "R": p["recall"], "Acc": p["accuracy"], "F1": p["f1"],
                     "wP": "—", "wR": "—", "wF1": "—"})
        for reading in ["R1", "R2"]:
            for thr in ["T1", "T2"]:
                e = [r["test"][thr] for r in R[f"r4.2_main_F38_P1_{model}_{reading}"]["runs"]]
                g = lambda k: [x[k] for x in e]
                rows.append({"Hàng": f"{model} {reading} {thr}",
                             "P": f"{m_only(g('precision'))} ({100*np.mean(g('precision'))-p['precision']:+.1f})",
                             "R": f"{m_only(g('recall'))} ({100*np.mean(g('recall'))-p['recall']:+.1f})",
                             "Acc": f"{m_only(g('accuracy'))} ({100*np.mean(g('accuracy'))-p['accuracy']:+.1f})",
                             "F1": f"{m_only(g('f1'))} ({100*np.mean(g('f1'))-p['f1']:+.1f})",
                             "wP": f"{m_only(g('w_precision'))} ({100*np.mean(g('w_precision'))-p['precision']:+.1f})",
                             "wR": f"{m_only(g('w_recall'))} ({100*np.mean(g('w_recall'))-p['recall']:+.1f})",
                             "wF1": f"{m_only(g('w_f1'))} ({100*np.mean(g('w_f1'))-p['f1']:+.1f})"})
    b = B["r4.2_main_F38_P1"]
    nob = [x["flag_nobody"] for x in b]
    g = lambda k: [x[k] for x in nob]
    rows.append({"Hàng": "*Gắn cờ không ai (P1 test, ~6.3% dương)*", "P": "0.0", "R": "0.0", "Acc": m_only(g("accuracy")), "F1": "0.0",
                 "wP": m_only(g("w_precision")), "wR": m_only(g("w_recall")), "wF1": m_only(g("w_f1"))})
    b2 = B["r4.2_main_F38_P2"]
    nob2 = [x["flag_nobody"] for x in b2]
    g2 = lambda k: [x[k] for x in nob2]
    rows.append({"Hàng": "*Gắn cờ không ai (P2 test, 4.72% dương)*", "P": "0.0", "R": "0.0", "Acc": m_only(g2("accuracy")), "F1": "0.0",
                 "wP": m_only(g2("w_precision")), "wR": m_only(g2("w_recall")), "wF1": m_only(g2("w_f1"))})
    for prev, lab in [(0.05, "5%"), (0.04, "4%")]:
        q = 1 - prev
        rows.append({"Hàng": f"*Gắn cờ không ai, tỉ lệ dương {lab} (giải tích)*", "P": "0.0", "R": "0.0", "Acc": f"{100*q:.1f}", "F1": "0.0",
                     "wP": f"{100*q*q:.1f}", "wR": f"{100*q:.1f}", "wF1": f"{100*q*2*q/(1+q):.1f}"})
    return md_table(pd.DataFrame(rows))


def p1bal_table(R, B):
    rows = []
    for rel, ls in [("4.2", "main"), ("4.2", "V30a"), ("4.2", "V30b"), ("6.2", "main")]:
        key = f"r{rel}_{ls}_F38_P1-bal"
        b = B[key]
        prev = np.mean([x["info"]["test_pos"] / x["info"]["test_rows"] for x in b])
        for model in ["AE", "VAE"]:
            for reading in ["R1", "R2"]:
                e = [r["test"] for r in R[f"{key}_{model}_{reading}"]["runs"]]
                rows.append({"Cấu hình": f"r{rel} {ls} (dương {100*prev:.1f}%)", "Mô hình": f"{model} {reading}",
                             "F1 T1": ms([x["T1"]["f1"] for x in e]), "F1 T2": ms([x["T2"]["f1"] for x in e]),
                             "AUC": ms([x["roc_auc"] for x in e], pct=False, nd=3)})
        rows.append({"Cấu hình": f"r{rel} {ls}", "Mô hình": "*Gắn cờ tất cả*", "F1 T1": "—",
                     "F1 T2": ms([x["flag_all"]["f1"] for x in b]), "AUC": "0.500"})
        rows.append({"Cấu hình": f"r{rel} {ls}", "Mô hình": "*Trần oracle*", "F1 T1": "—",
                     "F1 T2": ms([x["oracle"]["best"]["f1"] for x in b]), "AUC": ms([x["oracle"]["roc_auc"] for x in b], pct=False, nd=3)})
    return md_table(pd.DataFrame(rows))


def variants_table(R, B, P3C):
    """Best model per variant x protocol (T2 F1 mean) vs lookup and oracle."""
    rows = []
    for rel, ls, feat in [("4.2", "main", "F38"), ("4.2", "V30a", "F38"), ("4.2", "V30b", "F38"),
                          ("4.2", "main", "F50"), ("6.2", "main", "F38"), ("6.2", "main", "F50")]:
        for proto in ["P1", "P1-bal", "P2", "P3"]:
            names = [f"r{rel}_{ls}_{feat}_{proto}_{m}_{r}" for m in ["AE", "VAE"] for r in ["R1", "R2"]]
            names = [n for n in names if n in R]
            if not names:
                continue
            get = (lambda r: r["ud_max"]) if proto == "P3" else (lambda r: r["test"])
            f1 = {n: [get(r)["T2"]["f1"] for r in R[n]["runs"]] for n in names}
            auc = {n: [get(r)["roc_auc"] for r in R[n]["runs"]] for n in names}
            best = max(names, key=lambda n: np.mean(f1[n]))
            if proto == "P3":
                b = P3C[f"r{rel}_{ls}_{feat}"]
                look = [x["lookup"]["ud_max"]["T2"]["f1"] for x in b]
                orc = [x["oracle"]["ud_max"]["best"]["f1"] for x in b]
                fa = [x["oracle"]["ud_flag_all"]["f1"] for x in b]
            else:
                b = B[f"r{rel}_{ls}_{feat}_{proto}"]
                look = [x["lookup"]["T2"]["f1"] for x in b]
                orc = [x["oracle"]["best"]["f1"] for x in b]
                fa = [x["flag_all"]["f1"] for x in b]
            rows.append({"Cấu hình": f"r{rel} {ls} {feat}", "Giao thức": proto + (" (ud, max)" if proto == "P3" else ""),
                         "Mô hình tốt nhất (F1 T2)": best.split(f"{proto}_")[1].replace("_", " "),
                         "F1 T2 của nó": ms(f1[best]), "AUC của nó": ms(auc[best], pct=False, nd=3),
                         "Gắn cờ tất cả F1": m_only(fa), "Bảng tra F1 T2": m_only(look), "Trần F1": m_only(orc)})
    return md_table(pd.DataFrame(rows))


def main():
    R = load_runs()
    B = json.loads((ROOT / "results" / "baselines5.json").read_text())
    P3C = json.loads((ROOT / "results" / "p3_ceiling.json").read_text())
    tables = {
        "main": main_table(R, B, P3C, ["P1", "P2", "P3 ud_max", "P3 ud_mean", "P3 user_max", "P3 user_mean"]),
        "paper": paper_table(R, B),
        "p1bal": p1bal_table(R, B),
        "variants": variants_table(R, B, P3C),
    }
    rep = ROOT / "REPORT.md"
    text = rep.read_text(encoding="utf-8")
    for k, t in tables.items():
        text, n = re.subn(rf"<!--TABLE:{k}-->.*?<!--/TABLE-->", f"<!--TABLE:{k}-->\n{t}\n<!--/TABLE-->", text, flags=re.S)
        assert n == 1, f"marker {k} not found"
    rep.write_text(text, encoding="utf-8")
    print("REPORT.md tables filled:", ", ".join(tables))


if __name__ == "__main__":
    main()
