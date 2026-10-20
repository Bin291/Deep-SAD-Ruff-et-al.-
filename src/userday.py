"""Phase B (B1): user-day features, spec docs/phaseB_spec.md (B0 + B0 v2).

Two stages:
  1. extract(src_dir, out_dir): one streaming pass per log file (two for email
     and http) -> intermediate tables keyed by (user, day[, pc | domain]).
     Nothing here depends on labels or on any split.
  2. assemble(out_dir): per-user-day table, then history features that only
     use days < d (G4 primary PC, G5 first USB, G8 new domains, group D).

Groups (column prefixes): g1_ .. g8_ (G, both releases), r_ (R, r6.2 only),
kw_ (I, "scenario knowledge" reference only), z_/q_/d_ (D). Labels live in a
separate file (labels(rel)) so the feature table never sees them.

Usage: python src/userday.py 4.2 [--force]
  -> data/userday/r<rel>/*.parquet (intermediates),
     data/userday/r<rel>.parquet (features), data/userday/labels_r<rel>.parquet,
     results/phaseB/feature_schema_r<rel>.json
"""
import hashlib
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = {  # local copies, sha256 identical to HF (results/hash_log.jsonl)
    "4.2": Path(r"D:\Code\paper\data\cert_r4.2"),
    "6.2": Path(r"D:\Code\paper\data\kaggle_cert_insider_threat\extracted"),
}
DATE_FMT = "%m/%d/%Y %H:%M:%S"
ORG_DOMAIN = "@dtaa.com"
COLD_DAYS = 30          # V2.1
MIN_HIST = 10           # §3.3, V2.2
RATIO_WIN = 30          # §3.3
EXT_GROUPS = {"docs": ["doc", "pdf", "txt"], "exe": ["exe", "dll"], "zip": ["zip", "rar"], "img": ["jpg", "png"]}
KW = {  # §3.4, fixed in B0; per-release use in KW_REL (V2.4)
    "wikileaks": ["wikileaks"],
    "job": ["job", "career", "resume", "hiring", "recruit"],
    "keylog": ["keylog", "spy"],
    "dropbox": ["dropbox", "upload", "storage", "drive"],
}
KW_REL = {"4.2": ["wikileaks", "job", "keylog"], "6.2": ["wikileaks", "job", "keylog", "dropbox"]}
ACTS = ["logon", "logoff", "connect", "disconnect", "file", "email", "http"]
OFF_ACTS = ["logon", "connect", "file", "email", "http"]
D_BASE = ["g5_connect", "g6_file", "g7_send", "g7_ext_rcpt", "g7_with_att", "g8_http", "g8_dom_new_user",
          "g4_logon_otherpc"] + [f"g2_off_{a}" for a in OFF_ACTS] + ["g3_session_h"]


def header(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.readline().strip().split(",")


def scan(path, cols):
    return with_time(pl.scan_csv(path, infer_schema=False).select(cols))


def with_time(lf):
    ts = pl.col("date").str.strptime(pl.Datetime("us"), DATE_FMT, strict=True)
    hr = ts.dt.hour()
    return lf.with_columns(
        ts.dt.date().alias("day"),
        (hr // 6).cast(pl.Int8).alias("k"),
        ((ts.dt.weekday() >= 6) | (hr < 8) | (hr >= 18)).alias("off"),
        (hr + ts.dt.minute() / 60 + ts.dt.second() / 3600).alias("h"),
    )


def kcounts(name, cond=None):
    c = pl.lit(True) if cond is None else cond
    return [(c & (pl.col("k") == i)).sum().cast(pl.UInt32).alias(f"g1_{name}_k{i}") for i in range(4)]


def times():
    return [pl.col("h").min().alias("first_h"), pl.col("h").max().alias("last_h")]


def domain_expr(url):
    """host part of the URL, lower case; '' when the URL has no scheme://host."""
    return url.str.extract(r"^[A-Za-z][A-Za-z0-9+.-]*://([^/:?#]+)", 1).str.to_lowercase().fill_null("")


def addr_list(col_exprs):
    """';'-joined address fields -> list of lower-case non-empty addresses."""
    allr = pl.concat_str(col_exprs, separator=";", ignore_nulls=True)
    return allr.str.to_lowercase().str.split(";").list.eval(
        pl.element().str.strip_chars().filter(pl.element().str.strip_chars() != ""))


def rcpt_list():
    return addr_list([pl.col("to"), pl.col("cc"), pl.col("bcc")])


def n_external(lst):
    return lst.list.eval(pl.element().filter(~pl.element().str.ends_with(ORG_DOMAIN))).list.len()


def n_attachments(has_activity):
    a = pl.col("attachments")
    if has_activity:  # r6.2: "path(size);path(size)"
        return pl.when(a.is_null() | (a.str.strip_chars() == "")).then(0).otherwise(a.str.count_matches(";") + 1)
    return a.cast(pl.Int64, strict=True).fill_null(0)  # r4.2: integer count


def extract(src_dir, out_dir, log=print):
    """Stage 1. Writes logon/device/file/email/email_rcpt/http/http_dom parquet."""
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    q = {}

    lf = scan(src_dir / "logon.csv", ["date", "user", "pc", "activity"])
    lf = lf.with_columns(pl.col("activity").replace_strict({"Logon": "logon", "Logoff": "logoff"}).alias("a"))
    q["logon"] = lf.group_by("user", "day", "pc").agg(
        *kcounts("logon", pl.col("a") == "logon"), *kcounts("logoff", pl.col("a") == "logoff"),
        (pl.col("a") == "logon").sum().cast(pl.UInt32).alias("n_logon"),
        ((pl.col("a") == "logon") & pl.col("off")).sum().cast(pl.UInt32).alias("g2_off_logon"),
        pl.col("h").filter(pl.col("a") == "logon").min().alias("first_logon_h"),
        pl.col("h").filter(pl.col("a") == "logoff").max().alias("last_logoff_h"),
        *times())

    lf = scan(src_dir / "device.csv", ["date", "user", "pc", "activity"])
    lf = lf.with_columns(pl.col("activity").replace_strict({"Connect": "connect", "Disconnect": "disconnect"}).alias("a"))
    q["device"] = lf.group_by("user", "day", "pc").agg(
        *kcounts("connect", pl.col("a") == "connect"), *kcounts("disconnect", pl.col("a") == "disconnect"),
        ((pl.col("a") == "connect") & pl.col("off")).sum().cast(pl.UInt32).alias("g2_off_connect"), *times())

    rich = "activity" in header(src_dir / "file.csv")
    lf = scan(src_dir / "file.csv", ["date", "user", "pc", "filename"] + (["activity", "to_removable_media", "from_removable_media"] if rich else []))
    lf = lf.with_columns(pl.col("filename").str.extract(r"\.([A-Za-z0-9]+)$", 1).str.to_lowercase().alias("ext"))
    aggs = [*kcounts("file"), pl.col("off").sum().cast(pl.UInt32).alias("g2_off_file"), *times()]
    known = []
    for g, exts in EXT_GROUPS.items():
        aggs.append(pl.col("ext").is_in(exts).fill_null(False).sum().cast(pl.UInt32).alias(f"g6_ext_{g}"))
        known += exts
    aggs.append((~pl.col("ext").is_in(known).fill_null(False)).sum().cast(pl.UInt32).alias("g6_ext_other"))
    if rich:
        op = pl.col("activity").replace_strict({"File Open": "open", "File Write": "write", "File Copy": "copy", "File Delete": "delete"})
        lf = lf.with_columns(op.alias("op"))
        aggs += [(pl.col("op") == o).sum().cast(pl.UInt32).alias(f"r_file_{o}") for o in ["open", "write", "copy", "delete"]]
        aggs += [pl.col(c).replace_strict({"True": True, "False": False}, return_dtype=pl.Boolean).sum().cast(pl.UInt32).alias(n)
                 for c, n in [("to_removable_media", "r_file_to_rem"), ("from_removable_media", "r_file_from_rem")]]
        decoy = pl.scan_csv(src_dir / "decoy_file.csv", infer_schema=False).select(
            pl.col("decoy_filename").alias("filename"), "pc").unique().with_columns(pl.lit(True).alias("is_decoy"))
        lf = lf.join(decoy, on=["filename", "pc"], how="left")
        aggs.append(pl.col("is_decoy").fill_null(False).sum().cast(pl.UInt32).alias("r_decoy"))
    q["file"] = lf.group_by("user", "day", "pc").agg(aggs)

    rich = "activity" in header(src_dir / "email.csv")
    lf = scan(src_dir / "email.csv", ["date", "user", "pc", "to", "cc", "bcc", "size", "attachments"] + (["activity"] if rich else []))
    if rich:  # any other value is an error
        lf = lf.with_columns(pl.col("activity").replace_strict({"Send": "Send", "View": "View"}))
    send = (pl.col("activity") == "Send") if rich else pl.lit(True)
    lf = lf.with_columns(send.alias("send"), n_external(rcpt_list()).alias("n_ext"),
                         n_external(addr_list([pl.col("bcc")])).alias("n_bcc_ext"),
                         n_attachments(rich).alias("n_att"), pl.col("size").cast(pl.Int64, strict=True).alias("sz"))
    s = pl.col("send")
    aggs = [*kcounts("email"), pl.col("off").sum().cast(pl.UInt32).alias("g2_off_email"), *times(),
            s.sum().cast(pl.UInt32).alias("g7_send"),
            pl.col("n_ext").filter(s).sum().cast(pl.UInt32).alias("g7_ext_rcpt"),
            pl.col("n_bcc_ext").filter(s).sum().cast(pl.UInt32).alias("g7_bcc_ext"),
            (s & (pl.col("n_att") > 0)).sum().cast(pl.UInt32).alias("g7_with_att"),
            pl.col("n_att").filter(s).sum().cast(pl.UInt32).alias("g7_att"),
            pl.col("sz").filter(s).sum().cast(pl.Float64).alias("size_sum")]
    if rich:
        aggs.append((pl.col("activity") == "View").sum().cast(pl.UInt32).alias("r_email_view"))
    q["email"] = lf.group_by("user", "day", "pc").agg(aggs)
    q["email_rcpt"] = (scan(src_dir / "email.csv", ["date", "user", "to", "cc", "bcc"] + (["activity"] if rich else []))
                       .filter(send).select("user", "day", rcpt_list().alias("r")).explode("r")
                       .filter(pl.col("r").is_not_null())
                       .group_by("user", "day").agg(pl.col("r").n_unique().cast(pl.UInt32).alias("g7_rcpt_distinct")))

    for name, query in q.items():
        t0 = time.time()
        df = query.collect(engine="streaming")
        df.write_parquet(out_dir / f"{name}.parquet")
        log(f"  extract {name}: {len(df):,} rows, {time.time() - t0:.1f}s")
    t0 = time.time()
    h, d = http_tables(src_dir / "http.csv", log)
    h.write_parquet(out_dir / "http.parquet")
    d.write_parquet(out_dir / "http_dom.parquet")
    log(f"  extract http: {len(h):,} rows; http_dom: {len(d):,} rows, {time.time() - t0:.1f}s")


def http_flags(rich):
    """Row-level 0/1 columns of the http file (summed per group later)."""
    lu = pl.col("url").str.to_lowercase()
    flags = [(pl.col("k") == i).cast(pl.UInt32).alias(f"g1_http_k{i}") for i in range(4)]
    flags += [pl.col("off").cast(pl.UInt32).alias("g2_off_http")]
    flags += [lu.str.contains("|".join(words)).fill_null(False).cast(pl.UInt32).alias(f"kw_{g}") for g, words in KW.items()]
    if rich:
        a = pl.col("activity").replace_strict({"WWW Visit": "visit", "WWW Download": "download", "WWW Upload": "upload"})
        flags += [(a == "download").cast(pl.UInt32).alias("r_http_download"), (a == "upload").cast(pl.UInt32).alias("r_http_upload")]
    return flags


def http_tables(path, log=print, block=1 << 25, combine_every=128):
    """http per (user, day, pc) and per (user, day, domain), read in bounded
    blocks with pyarrow and aggregated with polars. The polars streaming CSV
    reader buffered the long url column of r6.2 (90 GB) without bound and ran
    out of RAM, so this file is read block by block instead (no value spans
    lines: rows == lines - 1, results/counts_summary_r*.json). pyarrow reads
    ahead in parallel, RSS ~ 70 x block size: 32 MB blocks -> ~2.4 GB."""
    import pyarrow as pa
    import pyarrow.csv as pacsv
    rich = "activity" in header(path)
    cols = ["date", "user", "pc", "url"] + (["activity"] if rich else [])
    flags = http_flags(rich)
    sums = [f.meta.output_name() for f in flags]
    keys_h, keys_d = ["user", "day", "pc"], ["user", "day", "domain"]
    agg_h = [pl.col(c).sum().cast(pl.UInt32) for c in sums] + [pl.col("first_h").min(), pl.col("last_h").max()]
    agg_d = [pl.col("n").sum().cast(pl.UInt32)]
    reader = pacsv.open_csv(path, read_options=pacsv.ReadOptions(block_size=block),
                            convert_options=pacsv.ConvertOptions(include_columns=cols, column_types={c: pa.string() for c in cols}))
    acc_h, acc_d, ph, pd_, n_rows = [], [], [], [], 0
    for i, batch in enumerate(reader):
        lf = with_time(pl.from_arrow(batch).lazy())
        n_rows += batch.num_rows
        ph.append(lf.with_columns(flags).group_by(keys_h).agg([pl.col(c).sum().cast(pl.UInt32) for c in sums] + times()).collect())
        pd_.append(lf.select("user", "day", domain_expr(pl.col("url")).alias("domain")).group_by(keys_d)
                   .agg(pl.len().cast(pl.UInt32).alias("n")).collect())
        if len(ph) >= combine_every:
            acc_h = [pl.concat(acc_h + ph).group_by(keys_h).agg(agg_h)]
            acc_d = [pl.concat(acc_d + pd_).group_by(keys_d).agg(agg_d)]
            ph, pd_ = [], []
            log(f"    http block {i + 1}: {n_rows:,} rows, partial {len(acc_h[0]):,} / {len(acc_d[0]):,}")
    h = pl.concat(acc_h + ph).group_by(keys_h).agg(agg_h)
    d = pl.concat(acc_d + pd_).group_by(keys_d).agg(agg_d)
    log(f"    http rows read: {n_rows:,}")
    return h, d


# ------------------------------------------------------------------ stage 2
def primary_pc_other(logon):
    """G4: logons on day d on PCs other than the user's primary PC, where the
    primary PC = most logons over days < d (ties: first PC to reach the max).
    0 when the user has no logon before d (V2.2)."""
    lg = logon.filter(pl.col("n_logon") > 0).select("user", "day", "pc", "n_logon").sort("user", "day", "pc")
    users, days, pcs, ns = (lg[c].to_list() for c in ["user", "day", "pc", "n_logon"])
    out_u, out_d, out_v = [], [], []
    i, n = 0, len(users)
    cnt, best, best_n, cur_user = {}, None, 0, None
    while i < n:
        u, d = users[i], days[i]
        if u != cur_user:
            cnt, best, best_n, cur_user = {}, None, 0, u
        j = i
        while j < n and users[j] == u and days[j] == d:
            j += 1
        other = 0 if best is None else sum(ns[k] for k in range(i, j) if pcs[k] != best)
        out_u.append(u)
        out_d.append(d)
        out_v.append(other)
        for k in range(i, j):
            c = cnt.get(pcs[k], 0) + ns[k]
            cnt[pcs[k]] = c
            if c > best_n:
                best, best_n = pcs[k], c
        i = j
    return pl.DataFrame({"user": out_u, "day": out_d, "g4_logon_otherpc": out_v},
                        schema={"user": pl.String, "day": pl.Date, "g4_logon_otherpc": pl.UInt32})


def domain_features(dom, start):
    """G8 n_domain, new-to-user, new-to-org (V2.1 cold start, V2.3 definition)."""
    d = dom.filter(pl.col("domain") != "").with_columns(
        pl.col("day").min().over("user", "domain").alias("fu"), pl.col("day").min().over("domain").alias("fo"))
    cold = pl.col("day") < pl.lit(start + timedelta(days=COLD_DAYS))
    return (d.group_by("user", "day").agg(pl.len().cast(pl.UInt32).alias("g8_domains"),
                                         (pl.col("day") == pl.col("fu")).sum().cast(pl.UInt32).alias("g8_dom_new_user"),
                                         (pl.col("day") == pl.col("fo")).sum().cast(pl.UInt32).alias("g8_dom_new_org"))
             .with_columns(pl.when(cold).then(0).otherwise(pl.col(c)).cast(pl.UInt32).alias(c)
                           for c in ["g8_dom_new_user", "g8_dom_new_org"]))


def history_features(df):
    """Group D (§3.3) + G5 first-USB flag (V2.2). df sorted by (user, day)."""
    out = [pl.int_range(pl.len()).over("user").alias("_prior")]  # active days before d
    for c in D_BASE:
        x = pl.col(c).cast(pl.Float64)
        out += [(x.cum_sum().over("user") - x).alias(f"_s_{c}"),
                ((x * x).cum_sum().over("user") - x * x).alias(f"_s2_{c}"),
                x.shift(1).rolling_mean(RATIO_WIN, min_samples=1).over("user").alias(f"_rm_{c}")]
    conn = pl.col("g5_connect").cast(pl.Int64)
    out.append((conn.cum_sum().over("user") - conn).alias("_prior_conn"))
    df = df.with_columns(out)
    short = pl.col("_prior") < MIN_HIST
    new = [short.cast(pl.Int8).alias("d_lich_su_ngan"),
           ((pl.col("g5_connect") > 0) & (pl.col("_prior_conn") == 0) & ~short).cast(pl.Int8).alias("g5_usb_first")]
    n = pl.col("_prior").cast(pl.Float64)
    for c in D_BASE:
        mu = pl.col(f"_s_{c}") / n
        sd = (pl.col(f"_s2_{c}") / n - mu * mu).clip(lower_bound=0).sqrt()
        x = pl.col(c).cast(pl.Float64)
        new += [pl.when(short).then(0.0).otherwise((x - mu) / (sd + 1)).alias(f"z_{c}"),
                pl.when(short).then(1.0).otherwise(x / (pl.col(f"_rm_{c}") + 1)).alias(f"q_{c}")]
    df = df.with_columns(new)
    return df.drop([c for c in df.columns if c.startswith("_")])


def assemble(out_dir):
    out_dir = Path(out_dir)
    t = {n: pl.read_parquet(out_dir / f"{n}.parquet") for n in ["logon", "device", "file", "email", "email_rcpt", "http", "http_dom"]}
    per_pc = ["logon", "device", "file", "email", "http"]
    pcs = pl.concat([t[n].select("user", "day", "pc") for n in per_pc]).unique()
    df = pcs.group_by("user", "day").agg(pl.len().cast(pl.UInt32).alias("g4_pcs"))
    firsts, lasts = [], []
    for n in per_pc:
        x = t[n]
        num = [c for c in x.columns if c not in ("user", "day", "pc") and (x[c].dtype.is_integer() or c == "size_sum")]
        agg = [pl.col(c).sum().alias(c) for c in num]
        agg += [pl.col("first_h").min().alias(f"first_h_{n}"), pl.col("last_h").max().alias(f"last_h_{n}")]
        if n == "logon":
            agg += [pl.col("first_logon_h").min(), pl.col("last_logoff_h").max()]
        df = df.join(x.group_by("user", "day").agg(agg), on=["user", "day"], how="left")
        firsts.append(f"first_h_{n}")
        lasts.append(f"last_h_{n}")
    df = df.join(t["email_rcpt"], on=["user", "day"], how="left")
    start = df["day"].min()
    df = df.join(domain_features(t["http_dom"], start), on=["user", "day"], how="left")
    df = df.join(primary_pc_other(t["logon"]), on=["user", "day"], how="left")
    ints = [c for c in df.columns if df[c].dtype.is_integer()]
    df = df.with_columns(pl.col(c).fill_null(0) for c in ints + ["size_sum"])

    def g1(a):
        return pl.sum_horizontal([pl.col(f"g1_{a}_k{i}").cast(pl.Int64) for i in range(4)])

    df = df.with_columns(
        pl.col("first_logon_h").fill_null(pl.min_horizontal(firsts)).alias("g3_first_logon_h"),
        pl.col("last_logoff_h").fill_null(pl.max_horizontal(lasts)).alias("g3_last_logoff_h"),
        (pl.col("day").dt.weekday() >= 6).cast(pl.Int8).alias("g2_weekend"),
        g1("connect").alias("g5_connect"), g1("file").alias("g6_file"), g1("http").alias("g8_http"),
        pl.when(pl.col("g7_send") > 0).then(pl.col("size_sum") / pl.col("g7_send")).otherwise(0.0).alias("g7_size_mean"),
        (pl.col("day") < pl.lit(start + timedelta(days=COLD_DAYS))).cast(pl.Int8).alias("g8_khoi_dau"),
        pl.sum_horizontal([g1(a) for a in ACTS]).alias("n_rows"),
    ).with_columns((pl.col("g3_last_logoff_h") - pl.col("g3_first_logon_h")).alias("g3_session_h"))
    df = df.drop(firsts + lasts + ["first_logon_h", "last_logoff_h", "size_sum", "n_logon"]).sort("user", "day")
    df = history_features(df)
    return df.select(["user", "day", "n_rows"] + sorted(c for c in df.columns if c not in ("user", "day", "n_rows")))


# ------------------------------------------------------------------ groups
def groups(df, rel):
    cols = df.columns
    D = [c for c in cols if c.startswith(("z_", "q_", "d_"))]
    return {"G": [c for c in cols if c[:3] in {f"g{i}_" for i in range(1, 9)}], "D": D,
            "R": [c for c in cols if c.startswith("r_")], "I": [f"kw_{g}" for g in KW_REL[rel]],
            "Z": [c for c in D if c.startswith("z_")]}


NON_COUNT = {"g3_first_logon_h", "g3_last_logoff_h", "g3_session_h", "g2_weekend", "g5_usb_first", "g8_khoi_dau", "g7_size_mean"}


def is_count(c):
    return not (c.startswith(("z_", "q_", "d_")) or c in NON_COUNT)


def matrix(df, cols):
    """Model input: log1p on count columns (and the mean size), raw for hours/flags/D. float32."""
    def col(c):
        v = df[c].to_numpy().astype(np.float64)
        return np.log1p(v) if is_count(c) or c == "g7_size_mean" else v
    return np.column_stack([col(c) for c in cols]).astype(np.float32)


def schema_hash(df):
    return hashlib.sha256(";".join(f"{c}:{df[c].dtype}" for c in df.columns).encode()).hexdigest()


# ------------------------------------------------------------------ labels
def labels(rel, users=None):
    """User-day labels from the Phase-A count tables (window + event, set `main`).
    Kept apart from the features."""
    ins = pd.read_csv(ROOT / "data" / "labels" / f"insiders_r{rel}.csv")
    users = sorted(ins["user"]) if users is None else users
    lf = pl.concat([pl.scan_parquet(ROOT / "data" / "counts" / f"r{rel}" / f"{t}.parquet") for t in ["logon", "device", "file", "email", "http"]])
    ud = lf.group_by("user", "day").agg(pl.col("n").sum().alias("n_rows_counts"), pl.col("n_win").sum(), pl.col("n_event").sum()).collect()
    is_ins = pl.col("user").is_in(pl.Series(users).implode())
    scen = pl.from_pandas(ins[["user", "scenario"]]).with_columns(pl.col("scenario").cast(pl.Int8))
    return (ud.with_columns(is_ins.cast(pl.Int8).alias("y_user"),
                            (is_ins & (pl.col("n_win") > 0)).cast(pl.Int8).alias("y_win"),
                            (is_ins & (pl.col("n_event") > 0)).cast(pl.Int8).alias("y_event"))
              .join(scen, on="user", how="left").sort("user", "day"))


def main(rel, force=False):
    out_dir = ROOT / "data" / "userday" / f"r{rel}"
    logp = ROOT / "logs" / f"phaseB_userday_r{rel}.log"

    def log(m):
        print(m, flush=True)
        with open(logp, "a", encoding="utf-8") as f:
            f.write(m + "\n")

    t0 = time.time()
    if force or not (out_dir / "http_dom.parquet").exists():
        log(f"r{rel}: extract from {SRC[rel]}")
        extract(SRC[rel], out_dir, log)
    df = assemble(out_dir)
    df.write_parquet(ROOT / "data" / "userday" / f"r{rel}.parquet")
    labels(rel).write_parquet(ROOT / "data" / "userday" / f"labels_r{rel}.parquet")
    g = groups(df, rel)
    info = {"release": rel, "rows": len(df), "columns": len(df.columns), "schema_sha256": schema_hash(df),
            "schema": {c: str(df[c].dtype) for c in df.columns}, "groups": {k: len(v) for k, v in g.items()},
            "group_columns": g, "seconds": round(time.time() - t0, 1)}
    (ROOT / "results" / "phaseB").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "phaseB" / f"feature_schema_r{rel}.json").write_text(json.dumps(info, indent=1))
    log(f"r{rel}: {len(df):,} user-days, {len(df.columns)} columns, groups {info['groups']}, "
        f"schema {info['schema_sha256'][:16]}, {info['seconds']}s")


if __name__ == "__main__":
    main(sys.argv[1], "--force" in sys.argv)
