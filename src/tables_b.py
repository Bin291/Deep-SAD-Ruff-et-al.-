"""B1 tables: descriptive tables required by B0 v2 + result tables from
results/phaseB/b1_r<rel>_<proto>.json.

Usage: python src/tables_b.py  -> results/phaseB_tables.md, results/phaseB/describe.json
"""
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import protocols_b as pb
import userday

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phaseB"
BUD = [str(b) for b in pb.BUDGETS]
BLAB = {"0.001": "0.1%", "0.005": "0.5%", "0.01": "1%", "0.05": "5%"}
NAMES = {
    "random": "Ngẫu nhiên (giải tích)",
    "volume*": "Khối lượng — chiều chọn trên train",
    "volume+": "Khối lượng +log(dòng)",
    "volume-": "Khối lượng −log(dòng)",
    "rare": "Ô hiếm (tỉ lệ dòng)",
    "rule": "Luật ngoài giờ + USB lần đầu ¹",
    "zmax": "z-score: max \\|z\\|",
    "zpos": "z-score: tổng z dương",
    "IF_GD": "Isolation Forest G+D",
    "IF_GDR": "Isolation Forest G+D+R",
    "HGB_GD": "*Tham chiếu có giám sát:* HGB G+D",
    "HGB_GDR": "*Tham chiếu có giám sát:* HGB G+D+R",
    "IF_GDI": "*Có thông tin kịch bản:* IF G+D+I",
    "HGB_GDI": "*Có thông tin kịch bản:* HGB G+D+I",
}
ORDER = list(NAMES)


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def ms(v, k=1.0, nd=3):
    v = np.asarray([x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))], float) * k
    if len(v) == 0:
        return "—"
    if len(v) == 1 or np.allclose(v, v[0]):
        return f"{v[0]:.{nd}f}" + (f" (×{len(v)})" if len(v) > 1 and k == 1.0 and False else "")
    return f"{v.mean():.{nd}f} ± {v.std():.{nd}f} [{v.min():.{nd}f}, {v.max():.{nd}f}]"


# ------------------------------------------------------------------ results
def per_baseline(runs):
    """{baseline: [result dicts over runs (and seeds)]}; adds volume* = sign chosen on fit."""
    out = {}
    for r in runs:
        for k, v in r["results"].items():
            out.setdefault(k.split("@")[0], []).append(v)
        out.setdefault("volume*", []).append(r["results"][r["info"]["volume_sign"]["chosen"]])
    return {k: out[k] for k in ORDER if k in out}


def chosen_note(runs):
    c = [r["info"]["volume_sign"]["chosen"] for r in runs]
    return f"{c.count('volume+')}/{len(c)} lần chọn +, {c.count('volume-')}/{len(c)} lần chọn −"


def main_table(B):
    rows = []
    for k, rs in B.items():
        t2 = [r["T2"]["f1"] if r.get("T2") else None for r in rs] if k != "random" else []
        rows.append({"Baseline": NAMES[k], "AUC": ms([r["roc_auc"] for r in rs]), "PR-AUC": ms([r["pr_auc"] for r in rs], nd=4),
                     "AUC (nhãn sự kiện)": ms([r.get("roc_auc_event") for r in rs]) if k != "random" else "0.500",
                     "PR-AUC (nhãn sự kiện)": ms([r.get("pr_auc_event") for r in rs], nd=4) if k != "random" else "—",
                     "F1 T2 (%)": ms(t2, 100, 1) if k != "random" else "—"})
    return md(pd.DataFrame(rows))


def budget_table(B, runs):
    ceil = {b: [r["results"]["random"]["budget"][b]["ceiling"] for r in runs] for b in BUD}
    rows = [{"Baseline": "**Trần recall min(1, b·N/P)**", **{f"R@{BLAB[b]}": ms(ceil[b], 100, 1) for b in BUD},
             **{f"P@{BLAB[b]}": "—" for b in BUD}}]
    for k, rs in B.items():
        rows.append({"Baseline": NAMES[k], **{f"R@{BLAB[b]}": ms([r["budget"][b]["recall"] for r in rs], 100, 1) for b in BUD},
                     **{f"P@{BLAB[b]}": ms([r["budget"][b]["precision"] for r in rs], 100, 2) for b in BUD}})
    return md(pd.DataFrame(rows))


def insider_table(B):
    rows = []
    for k, rs in B.items():
        if k == "random":
            continue
        row = {"Baseline": NAMES[k]}
        for b in BUD:
            row[f"Phát hiện @{BLAB[b]} (%)"] = ms([np.mean([i["detected"] for i in r["insider"][b]]) for r in rs], 100, 1)
        for b in ["0.01", "0.05"]:
            d = [i["days_to_alert"] for r in rs for i in r["insider"][b]]
            got = [x for x in d if x is not None]
            row[f"Ngày tới cảnh báo @{BLAB[b]} τ val: trung vị [đã cảnh báo/tổng]"] = (
                f"{np.median(got):.0f} [{len(got)}/{len(d)}]" if got else f"— [0/{len(d)}]")
        rows.append(row)
    return md(pd.DataFrame(rows))


def scenario_table(B, scen):
    rows = []
    for s in scen:
        s = str(s)
        for k, rs in B.items():
            if k == "random":
                continue
            rr = [r["scenario"].get(s) for r in rs if r["scenario"].get(s)]
            if not rr:
                continue
            row = {"Kịch bản": s, "Baseline": NAMES[k], "dương test": ms([x["pos"] for x in rr], nd=0),
                   "AUC": ms([x["roc_auc"] for x in rr]), "PR-AUC": ms([x["pr_auc"] for x in rr], nd=4)}
            row |= {f"R@{BLAB[b]} (%)": ms([x["budget"][b]["recall"] for x in rr], 100, 1) for b in BUD}
            if k == "volume*":
                row["Baseline"] += ""
            rows.append(row)
        rows.append({"Kịch bản": s, "Baseline": "**trần recall**", "dương test": "", "AUC": "", "PR-AUC": "",
                     **{f"R@{BLAB[b]} (%)": ms([x["scenario"][s]["budget"][b]["ceiling"] for x in B["rule"] if x["scenario"].get(s)], 100, 1)
                        for b in BUD}})
    return md(pd.DataFrame(rows))


def r62_insider_tables(runs, proto):
    """Per insider: best positive user-day rank (top x% of test) and days to alert @1%."""
    rank, days = {}, {}
    for r in runs:
        for k, v in r["results"].items():
            if k == "random":
                continue
            base = k.split("@")[0]
            for i in v["insider"]["0.01"]:
                col = f"{i['user']} (kb {i['scenario']})"
                rank.setdefault(base, {}).setdefault(col, []).append(i["best_rank_pct"])
                days.setdefault(base, {}).setdefault(col, []).append(i["days_to_alert"])
    cols = sorted({c for v in rank.values() for c in v}, key=lambda c: c.split("kb ")[1])

    def fmt_rank(v):
        return " / ".join(f"{100 * x:.2f}%" for x in v)

    def fmt_days(v):
        return " / ".join("–" if x is None else str(x) for x in v)

    tr = [{"Baseline": NAMES[b], **{c: fmt_rank(rank[b].get(c, [])) for c in cols}} for b in ORDER if b in rank]
    td = [{"Baseline": NAMES[b], **{c: fmt_days(days[b].get(c, [])) for c in cols}} for b in ORDER if b in days]
    note = ("mỗi ô: giá trị ở từng lần chạy chứa insider đó (PB-U: 2 lần lặp; PB-T: 1 lần cho baseline tất định, 5 seed cho IF/HGB)")
    return md(pd.DataFrame(tr)), md(pd.DataFrame(td)), note


def sizes_table(res):
    rows = []
    for (rel, proto), d in res.items():
        R = d["runs"]
        ins = [r["results"]["rule"]["insiders"] for r in R]
        rows.append({"Bản": f"r{rel}", "Giao thức": proto, "Số lần chia": len(R),
                     "user-day fit": ms([r["n_fit"] for r in R], nd=0), "dương fit": ms([r["fit_pos"] for r in R], nd=0),
                     "user-day val": ms([r["n_val"] for r in R], nd=0), "dương val": ms([r["val_pos"] for r in R], nd=0),
                     "user-day test": ms([r["n_test"] for r in R], nd=0), "dương test": ms([r["test_pos"] for r in R], nd=0),
                     "insider test": ms(ins, nd=0)})
    return md(pd.DataFrame(rows))


def sklearn_table(res):
    rows = []
    for (rel, proto), d in res.items():
        for r in d["runs"]:
            for k, m in r["info"].items():
                if k.startswith("HGB"):
                    rows.append({"Bản": f"r{rel}", "Giao thức": proto, "Mô hình": k.split("@")[0], "dương fit": m["fit_pos"],
                                 "early stopping bật": m["early_stopping_active"], "số vòng": m["n_iter"]})
    df = pd.DataFrame(rows)
    g = df.groupby(["Bản", "Giao thức", "Mô hình"]).agg(lambda s: ms(list(s), nd=0) if s.dtype != bool else f"{s.mean():.0%}")
    return md(g.reset_index())


# ------------------------------------------------------------------ describe
def describe(rel):
    f = pl.read_parquet(ROOT / "data" / "userday" / f"r{rel}.parquet")
    lab = pl.read_parquet(ROOT / "data" / "userday" / f"labels_r{rel}.parquet")
    ins = pb.insiders(rel)
    info = json.loads((OUT / f"feature_schema_r{rel}.json").read_text())
    start = f["day"].min()
    cold = lab.filter(pl.col("day") < start + timedelta(days=userday.COLD_DAYS))
    d = {"features": {"rows": info["rows"], "columns": info["columns"], "groups": info["groups"], "schema_sha256": info["schema_sha256"],
                      "first_day": str(start), "last_day": str(f["day"].max())},
         "cold_start": {"days": f"{start} → {start + timedelta(days=userday.COLD_DAYS - 1)}", "user_days": len(cold),
                        "pos_win": int(cold["y_win"].sum()), "pos_event": int(cold["y_event"].sum()),
                        "insiders_with_pos": int(cold.filter(pl.col("y_win") == 1)["user"].n_unique())}}
    j = f.select("user", "day", *[f"kw_{g}" for g in userday.KW]).join(lab.select("user", "day", "y_win", "y_user"), on=["user", "day"])
    d["keywords"] = {g: {"used_in_release": g in userday.KW_REL[rel], "http_rows": int(j[f"kw_{g}"].sum()),
                         "user_days_with_match": int((j[f"kw_{g}"] > 0).sum()),
                         "pos_user_days_with_match": int(((j[f"kw_{g}"] > 0) & (j["y_win"] == 1)).sum()),
                         "pos_user_days": int(j["y_win"].sum()),
                         "insiders_with_match_in_pos_days": int(j.filter((pl.col(f"kw_{g}") > 0) & (pl.col("y_win") == 1))["user"].n_unique())}
                     for g in userday.KW}
    sp = pb.pbt_split(lab, rel)
    phase = np.empty(len(lab), object)
    for k in ["fit", "val", "test"]:
        phase[sp[k]] = k
    L = lab.with_columns(pl.Series("phase", phase)).filter(pl.col("y_win") == 1)
    tab = []
    for s in sorted(ins["scenario"].unique()):
        row = {"scenario": int(s)}
        for k in ["fit", "val", "test"]:
            x = L.filter((pl.col("scenario") == s) & (pl.col("phase") == k))
            row[f"{k}_insiders"] = int(x["user"].n_unique())
            row[f"{k}_pos"] = len(x)
        a = set(L.filter((pl.col("scenario") == s) & (pl.col("phase") == "fit"))["user"])
        b = set(L.filter((pl.col("scenario") == s) & (pl.col("phase") == "test"))["user"])
        row["insiders_fit_and_test"] = len(a & b)
        tab.append(row)
    d["pbt_scenario_phase"] = tab
    u = (lab.group_by("user").agg(pl.len().alias("active_days"), pl.col("n_rows_counts").mean().alias("rows_per_day"),
                                  pl.col("day").max().alias("last_day"), pl.col("y_user").first())
            .join(pl.from_pandas(ins[["user", "scenario"]]), on="user", how="left"))
    last = u["last_day"].max()
    diag = []
    groups = [("người thường", pl.col("y_user") == 0), ("insider (tất cả)", pl.col("y_user") == 1)]
    groups += [(f"insider kịch bản {s}", pl.col("scenario") == s) for s in sorted(ins["scenario"].unique())]
    for name, cond in groups:
        x = u.filter(cond)
        diag.append({"group": name, "users": len(x),
                     "active_days_mean": float(x["active_days"].mean()), "active_days_median": float(x["active_days"].median()),
                     "rows_per_day_mean": float(x["rows_per_day"].mean()), "rows_per_day_median": float(x["rows_per_day"].median()),
                     "share_last_day_before_end_minus_30": float((x["last_day"] < last - timedelta(days=30)).mean()),
                     "last_day_median": str(x["last_day"].sort()[len(x) // 2])})
    d["diagnostic"] = diag
    d["release_last_day"] = str(last)
    return d


def describe_tables(D):
    out = []
    rows = [{"Bản": f"r{r}", "user-day": d["features"]["rows"], "cột": d["features"]["columns"],
             **{f"nhóm {k}": v for k, v in d["features"]["groups"].items() if k != "Z"},
             "sha256 schema (16 ký tự đầu)": d["features"]["schema_sha256"][:16], "ngày": f"{d['features']['first_day']} → {d['features']['last_day']}"}
            for r, d in D.items()]
    out.append(("Bảng đặc trưng", md(pd.DataFrame(rows))))
    rows = [{"Bản": f"r{r}", "30 ngày đầu": d["cold_start"]["days"], "user-day": d["cold_start"]["user_days"],
             "user-day dương (cửa sổ)": d["cold_start"]["pos_win"], "user-day dương (sự kiện)": d["cold_start"]["pos_event"],
             "insider có dương": d["cold_start"]["insiders_with_pos"]} for r, d in D.items()]
    out.append(("V2.1 — user-day dương trong 30 ngày khởi đầu", md(pd.DataFrame(rows))))
    rows = [{"Bản": f"r{r}", "Nhóm": g, "Dùng ở bản này": "có" if k["used_in_release"] else "không",
             "Lượt http khớp": f"{k['http_rows']:,}", "user-day có khớp": f"{k['user_days_with_match']:,}",
             "user-day dương có khớp / tổng dương": f"{k['pos_user_days_with_match']} / {k['pos_user_days']}",
             "insider có khớp trong ngày dương": k["insiders_with_match_in_pos_days"]}
            for r, d in D.items() for g, k in d["keywords"].items()]
    out.append(("V2.4 — nhóm I: số lượt khớp từ khoá (chỉ mô tả)", md(pd.DataFrame(rows))))
    rows = []
    for r, d in D.items():
        for t in d["pbt_scenario_phase"]:
            rows.append({"Bản": f"r{r}", "Kịch bản": t["scenario"],
                         "train: insider / dương": f"{t['fit_insiders']} / {t['fit_pos']}",
                         "val: insider / dương": f"{t['val_insiders']} / {t['val_pos']}",
                         "test: insider / dương": f"{t['test_insiders']} / {t['test_pos']}",
                         "insider có dương ở cả train và test": t["insiders_fit_and_test"]})
    out.append(("V2.5 — PB-T: kịch bản × giai đoạn (chỉ mô tả; mốc không đổi)", md(pd.DataFrame(rows))))
    rows = [{"Bản": f"r{r}", "Nhóm": x["group"], "Số user": x["users"],
             "Ngày hoạt động: TB / trung vị": f"{x['active_days_mean']:.1f} / {x['active_days_median']:.0f}",
             "Dòng / ngày hoạt động: TB / trung vị": f"{x['rows_per_day_mean']:.1f} / {x['rows_per_day_median']:.1f}",
             "Ngày hoạt động cuối (trung vị)": x["last_day_median"],
             f"Tỉ lệ dừng > 30 ngày trước ngày cuối của bản": f"{100 * x['share_last_day_before_end_minus_30']:.0f}%"}
            for r, d in D.items() for x in d["diagnostic"]]
    out.append(("V2.10 — chẩn đoán \"insider rời sớm\" (phân tích sau khi xem kết quả giai đoạn A)", md(pd.DataFrame(rows))))
    return out


def univariate(rel, top=12):
    """Post-hoc (after seeing B1 results): single-feature AUC / PR-AUC on FIT
    user-days only (train labels; no test data), PB-U run 0 and PB-T."""
    f = pl.read_parquet(ROOT / "data" / "userday" / f"r{rel}.parquet")
    lab = pl.read_parquet(ROOT / "data" / "userday" / f"labels_r{rel}.parquet")
    g = userday.groups(f, rel)
    cols = g["G"] + g["D"] + g["R"] + g["I"]
    X = userday.matrix(f, cols)
    y = lab["y_win"].to_numpy()
    rows = []
    for proto, sp in [("PB-U fold 0", pb.pbu_splits(lab, rel)[0]), ("PB-T", pb.pbt_split(lab, rel))]:
        fit = sp["fit"]
        res = []
        for j, c in enumerate(cols):
            r = pb.ranking(X[fit, j], y[fit])
            sign = "+"
            if r["roc_auc"] < 0.5:
                r, sign = pb.ranking(-X[fit, j], y[fit]), "−"
            res.append((c, sign, r["roc_auc"], r["pr_auc"]))
        res.sort(key=lambda t: -t[3])
        prev = y[fit].mean()
        for c, sign, a, p in res[:top]:
            grp = "I" if c.startswith("kw_") else "R" if c.startswith("r_") else "D" if c.startswith(("z_", "q_", "d_")) else "G"
            rows.append({"Bản": f"r{rel}", "Tập fit": proto, "Đặc trưng": f"{sign}{c}", "Nhóm": grp,
                         "AUC": f"{a:.3f}", "PR-AUC": f"{p:.4f}", "tỉ lệ dương fit": f"{prev:.2e}"})
    return md(pd.DataFrame(rows))


def summary_table(res, metric):
    """One row per baseline, one column per release x protocol (mean ± std over runs)."""
    cols = {f"r{rel} {proto}": per_baseline(d["runs"]) for (rel, proto), d in res.items()}
    rows = []
    for k in ORDER:
        row = {"Baseline": NAMES[k]}
        for c, B in cols.items():
            rs = B.get(k)
            if not rs:
                row[c] = "—"
            elif metric == "pr_auc":
                row[c] = ms([r["pr_auc"] for r in rs], nd=4)
            elif metric == "roc_auc":
                row[c] = ms([r["roc_auc"] for r in rs])
            elif metric == "r1":
                row[c] = ms([r["budget"]["0.01"]["recall"] for r in rs], 100, 1)
            elif metric == "det1":
                row[c] = "—" if k == "random" else ms([np.mean([i["detected"] for i in r["insider"]["0.01"]]) for r in rs], 100, 0)
        rows.append(row)
    if metric == "r1":
        rows.insert(0, {"Baseline": "**Trần recall @1%**", **{c: ms([r["results"]["random"]["budget"]["0.01"]["ceiling"] for r in d["runs"]], 100, 1)
                                                            for c, d in zip(cols, res.values())}})
    return md(pd.DataFrame(rows))


def fill_report(markers):
    rep = ROOT / "REPORT_B.md"
    if not rep.exists():
        return
    import re
    text = rep.read_text(encoding="utf-8")
    for k, t in markers.items():
        text, n = re.subn(rf"<!--TABLE:{k}-->.*?<!--/TABLE-->", lambda _: f"<!--TABLE:{k}-->\n{t}\n<!--/TABLE-->", text, flags=re.S)
        if n != 1:
            print("marker missing:", k)
    rep.write_text(text, encoding="utf-8")


def main():
    D = {rel: describe(rel) for rel in ["4.2", "6.2"] if (ROOT / "data" / "userday" / f"labels_r{rel}.parquet").exists()}
    (OUT / "describe.json").write_text(json.dumps(D, indent=1, default=str))
    res = {}
    for rel in ["4.2", "6.2"]:
        for proto in ["PBU", "PBT"]:
            p = OUT / f"b1_r{rel}_{proto}.json"
            if p.exists():
                res[(rel, proto)] = json.loads(p.read_text())
    parts = ["# Giai đoạn B — B1: bảng kết quả\n",
             "Sinh tự động bởi `src/tables_b.py` từ `results/phaseB/*.json`. Mỗi số: trung bình ± std [min, max] qua các lần chạy "
             "(PB-U: 10 fold; PB-T: 1 lần chia — baseline tất định 1 giá trị, IF/HGB 5 seed). "
             "¹ Luật được thiết kế khi đã biết hình dạng kịch bản (V2.7). Dòng in nghiêng là tham chiếu, không phải baseline.\n"]
    for title, t in describe_tables(D):
        parts += [f"## {title}\n", t, ""]
    parts += ["## Kích thước tập\n", sizes_table(res), ""]
    for (rel, proto), d in res.items():
        B = per_baseline(d["runs"])
        parts += [f"## r{rel} — {proto}\n", f"Khối lượng: {chosen_note(d['runs'])}.\n",
                  "### Xếp hạng (AUC, PR-AUC) và F1 ở ngưỡng T2\n", main_table(B), "",
                  "### Ngân sách cảnh báo: recall@b, precision@b (%)\n", budget_table(B, d["runs"]), "",
                  "### Mức insider\n", insider_table(B), ""]
        if rel == "4.2":
            parts += ["### Theo kịch bản (V2.9; nhãn kịch bản chỉ để báo cáo)\n", scenario_table(B, [1, 2, 3]), ""]
        else:
            tr, td, note = r62_insider_tables(d["runs"], proto)
            parts += [f"### Từng insider: thứ hạng của user-day dương tốt nhất (top x% của test) — {note}\n", tr, "",
                      "### Từng insider: số ngày từ đầu cửa sổ tới cảnh báo đầu tiên @1% (τ từ val; – = không cảnh báo)\n", td, ""]
    parts += ["## HGB: early stopping thực tế\n", sklearn_table(res), ""]
    parts += ["## Phân tích sau (sau khi thấy kết quả B1): đặc trưng đơn lẻ mạnh nhất trên tập FIT\n",
              "Chỉ dùng user-day fit và nhãn train (không đọc test), xếp theo PR-AUC; dấu − = điểm càng thấp càng nghi. "
              "Dùng để giải thích khoảng cách HGB ↔ baseline không giám sát, không dùng để chọn gì.\n"]
    for rel in ["4.2", "6.2"]:
        if (ROOT / "data" / "userday" / f"r{rel}.parquet").exists():
            parts += [f"### r{rel}\n", univariate(rel), ""]
    summ = {m: summary_table(res, m) for m in ["roc_auc", "pr_auc", "r1", "det1"]}
    parts[2:2] = ["## Tóm tắt\n", "### AUC\n", summ["roc_auc"], "", "### PR-AUC\n", summ["pr_auc"], "",
                  "### recall@1% (%)\n", summ["r1"], "", "### insider phát hiện @1% (%)\n", summ["det1"], ""]
    (ROOT / "results" / "phaseB_tables.md").write_text("\n".join(parts), encoding="utf-8")
    print("written results/phaseB_tables.md")
    dt = describe_tables(D)
    fill_report({"b1auc": summ["roc_auc"], "b1pr": summ["pr_auc"], "b1r1": summ["r1"], "b1det": summ["det1"],
                  "sizes": sizes_table(res), **{f"desc{i}": t for i, (_, t) in enumerate(dt)}})


if __name__ == "__main__":
    main()
