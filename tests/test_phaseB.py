"""Phase B mandatory tests (docs/phaseB_spec.md §4.6 + V2.8):
(a) features of days <= d do not change when data of days > d is scrambled
(b) train/val/test users (PB-U) and date ranges (PB-T) are disjoint
(c) G1 totals equal the Phase-A count tables (real data; skipped if not built)
(d) domain / e-mail parsers on synthetic examples
(e) raw features do not depend on the split or the label set; only the scaler does
"""
import inspect
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import userday  # noqa: E402

USERS = ["AAA0001", "BBB0002", "CCC0003", "DDD0004"]
PCS = ["PC-0001", "PC-0002", "PC-0003", "PC-0004", "PC-0099"]
START = datetime(2010, 1, 2)
N_DAYS = 70


def _ts(day, hour, rng):
    t = START + timedelta(days=int(day), hours=float(hour), seconds=int(rng.integers(0, 3600)))
    return t.strftime("%m/%d/%Y %H:%M:%S")


def make_logs(path, rich, seed=0, scramble_after=None, scramble_seed=1):
    """Synthetic CERT-format logs. rich=True -> r6.2 columns. When
    scramble_after=d, events of days > d are redrawn with another seed."""
    path.mkdir(parents=True, exist_ok=True)
    rows = {k: [] for k in ["logon", "device", "file", "email", "http"]}
    for day in range(N_DAYS):
        rng = np.random.default_rng((seed, day) if scramble_after is None or day <= scramble_after else (scramble_seed, day))
        for ui, u in enumerate(USERS):
            if rng.random() < 0.15:
                continue
            pc = PCS[ui] if rng.random() < 0.85 else PCS[4]
            h0 = rng.integers(5, 10)
            rows["logon"].append(f"{{L{day}{u}a}},{_ts(day, h0, rng)},{u},{pc},Logon")
            if rng.random() < 0.9:
                rows["logon"].append(f"{{L{day}{u}b}},{_ts(day, h0 + 8, rng)},{u},{pc},Logoff")
            if rng.random() < 0.3:
                hh = rng.integers(0, 23)
                extra = ",R:\\;R:\\x" if rich else ""
                rows["device"].append(f"{{D{day}{u}a}},{_ts(day, hh, rng)},{u},{pc}{extra},Connect")
                rows["device"].append(f"{{D{day}{u}b}},{_ts(day, hh, rng)},{u},{pc}{extra},Disconnect")
            for k in range(rng.integers(0, 4)):
                ext = rng.choice(["doc", "pdf", "exe", "zip", "jpg", "txt"])
                fn = f"C:\\X\\F{k}{u}.{ext}" if rich else f"F{k}{u}.{ext}"
                t = _ts(day, rng.integers(0, 24), rng)
                if rich:
                    op = rng.choice(["File Open", "File Write", "File Copy", "File Delete"])
                    rows["file"].append(f"{{F{day}{u}{k}}},{t},{u},{pc},{fn},{op},{rng.random() < 0.3},{rng.random() < 0.2},x y")
                else:
                    rows["file"].append(f"{{F{day}{u}{k}}},{t},{u},{pc},{fn},x y")
            for k in range(rng.integers(0, 4)):
                to = "a.b@dtaa.com;" + ("x@gmail.com" if rng.random() < 0.3 else "c.d@dtaa.com")
                bcc = "z@yahoo.com" if rng.random() < 0.1 else ""
                t = _ts(day, rng.integers(0, 24), rng)
                if rich:
                    act = rng.choice(["Send", "View"])
                    att = "C:\\a.doc(12);C:\\b.pdf(3)" if rng.random() < 0.3 else ""
                    rows["email"].append(f"{{E{day}{u}{k}}},{t},{u},{pc},{to},,{bcc},{u}@dtaa.com,{act},{rng.integers(100, 9999)},{att},hi")
                else:
                    rows["email"].append(f"{{E{day}{u}{k}}},{t},{u},{pc},{to},,{bcc},{u}@dtaa.com,{rng.integers(100, 9999)},{rng.integers(0, 3)},hi")
            for k in range(rng.integers(1, 8)):
                dom = rng.choice(["msn.com", "aa.com", f"site{rng.integers(0, 40)}.org", "wikileaks.org", "jobhunt.com"])
                t = _ts(day, rng.integers(0, 24), rng)
                if rich:
                    act = rng.choice(["WWW Visit", "WWW Download", "WWW Upload"], p=[0.9, 0.08, 0.02])
                    rows["http"].append(f"{{H{day}{u}{k}}},{t},{u},{pc},http://{dom}/Some_Page/x{k}.htm,{act},words")
                else:
                    rows["http"].append(f"{{H{day}{u}{k}}},{t},{u},{pc},http://{dom}/Some_Page/x{k}.htm,words")
    heads = {
        "logon": "id,date,user,pc,activity",
        "device": "id,date,user,pc,file_tree,activity" if rich else "id,date,user,pc,activity",
        "file": "id,date,user,pc,filename,activity,to_removable_media,from_removable_media,content" if rich else "id,date,user,pc,filename,content",
        "email": "id,date,user,pc,to,cc,bcc,from,activity,size,attachments,content" if rich else "id,date,user,pc,to,cc,bcc,from,size,attachments,content",
        "http": "id,date,user,pc,url,activity,content" if rich else "id,date,user,pc,url,content",
    }
    for k, v in rows.items():
        (path / f"{k}.csv").write_text(heads[k] + "\n" + "\n".join(v) + "\n")
    if rich:
        (path / "decoy_file.csv").write_text('"decoy_filename","pc"\n"C:\\X\\F0AAA0001.doc","PC-0001"\n')
    return path


def build(tmp_path, name, **kw):
    src = make_logs(tmp_path / f"src_{name}", **kw)
    out = tmp_path / f"out_{name}"
    userday.extract(src, out, log=lambda m: None)
    return userday.assemble(out)


@pytest.mark.parametrize("rich", [False, True])
def test_a_no_future_leak(tmp_path, rich):
    cut = 40
    base = build(tmp_path, "base", rich=rich)
    scr = build(tmp_path, "scr", rich=rich, scramble_after=cut)
    d_cut = (START + timedelta(days=cut)).date()
    a = base.filter(pl.col("day") <= d_cut)
    b = scr.filter(pl.col("day") <= d_cut)
    assert a.shape == b.shape and len(a) > 100
    assert a.equals(b), "features of days <= d changed when days > d were scrambled"
    # the scramble really changed something after the cut
    assert not base.filter(pl.col("day") > d_cut).equals(scr.filter(pl.col("day") > d_cut))
    # history features exist and are non-trivial
    assert (a["d_lich_su_ngan"] == 0).sum() > 0 and a["z_g8_http"].abs().sum() > 0
    if rich:
        assert int(base["r_decoy"].sum()) >= 1 and int(base["r_file_to_rem"].sum()) > 0


def test_d_parsers():
    urls = pl.Series(["http://WikiLeaks.org/a/b.htm", "https://x.y.com:8080/q?z=1", "http://msn.com", "ftp://h/p", "no-scheme/path", None])
    got = pl.DataFrame({"u": urls}).select(userday.domain_expr(pl.col("u"))).to_series().to_list()
    assert got == ["wikileaks.org", "x.y.com", "msn.com", "h", "", ""]
    df = pl.DataFrame({"to": ["A.B@dtaa.com;X@Gmail.com", "a@dtaa.com", None],
                       "cc": ["c@lockheedmartin.com", None, ""], "bcc": ["", "Y@yahoo.com;q@DTAA.com", None]})
    out = df.select(userday.n_external(userday.rcpt_list()).alias("ext"),
                    userday.rcpt_list().list.len().alias("n"),
                    userday.n_external(userday.addr_list([pl.col("bcc")])).alias("bcc_ext"))
    assert out["ext"].to_list() == [2, 1, 0]
    assert out["n"].to_list() == [3, 3, 0]
    assert out["bcc_ext"].to_list() == [0, 1, 0]
    att = pl.DataFrame({"attachments": ["C:\\a.doc(1);C:\\b.pdf(2)", "C:\\a.doc(1)", None, ""]})
    assert att.select(userday.n_attachments(True)).to_series().to_list() == [2, 1, 0, 0]
    att = pl.DataFrame({"attachments": ["0", "3", None]})
    assert att.select(userday.n_attachments(False)).to_series().to_list() == [0, 3, 0]


def test_d_features_on_synthetic(tmp_path):
    """Hand-checked semantics on a tiny log: G4, G5 first USB (V2.2), G8 new domains (V2.1, V2.3), D."""
    src = tmp_path / "tiny"
    src.mkdir()
    lg, dv, ht = ["id,date,user,pc,activity"], ["id,date,user,pc,activity"], ["id,date,user,pc,url,content"]
    for d in range(45):
        day = (date(2010, 1, 2) + timedelta(days=d)).strftime("%m/%d/%Y")
        for u in ["U1", "U2"]:
            pc = "PC-2" if (u == "U1" and d == 41) else f"PC-{u[-1]}"
            lg.append(f"{{l{d}{u}}},{day} 09:00:00,{u},{pc},Logon")
            lg.append(f"{{o{d}{u}}},{day} 17:00:00,{u},{pc},Logoff")
            ht.append(f"{{h{d}{u}}},{day} 10:00:00,{u},{pc},http://common.com/a,w")
        if d in (5, 35):  # first USB: U1 at d=5 (<10 days history -> no flag); U2 at d=35 -> flag
            dv.append(f"{{c{d}}},{day} 20:00:00,{'U1' if d == 5 else 'U2'},PC-1,Connect")
        if d == 40:  # new domain for both users the same day (V2.3), after cold start
            ht.append(f"{{n1}},{day} 11:00:00,U1,PC-1,http://NEW.com/x,w")
            ht.append(f"{{n2}},{day} 12:00:00,U2,PC-2,http://new.com/y,w")
        if d == 3:  # domain first seen during cold start -> feature forced to 0
            ht.append(f"{{e1}},{day} 11:00:00,U1,PC-1,http://early.com/x,w")
    (src / "logon.csv").write_text("\n".join(lg) + "\n")
    (src / "device.csv").write_text("\n".join(dv) + "\n")
    (src / "http.csv").write_text("\n".join(ht) + "\n")
    (src / "file.csv").write_text("id,date,user,pc,filename,content\n{f},01/02/2010 09:30:00,U1,PC-1,a.DOC,x\n")
    (src / "email.csv").write_text("id,date,user,pc,to,cc,bcc,from,size,attachments,content\n"
                                   "{e},01/02/2010 09:40:00,U1,PC-1,a@dtaa.com;b@x.com,,c@y.com,U1@dtaa.com,100,2,x\n")
    userday.extract(src, tmp_path / "o", log=lambda m: None)
    df = userday.assemble(tmp_path / "o")

    def at(u, d, c):
        return df.filter((pl.col("user") == u) & (pl.col("day") == date(2010, 1, 2) + timedelta(days=d)))[c].item()

    assert at("U1", 0, "g4_logon_otherpc") == 0          # no history
    assert at("U1", 41, "g4_logon_otherpc") == 1         # logon on PC-2, primary PC-1
    assert at("U1", 41, "g4_pcs") == 1                   # every U1 row that day is on PC-2
    assert at("U1", 5, "g5_usb_first") == 0              # only 5 active days before
    assert at("U2", 35, "g5_usb_first") == 1
    assert at("U1", 0, "g8_dom_new_user") == 0 and at("U1", 0, "g8_khoi_dau") == 1   # cold start
    assert at("U1", 3, "g8_dom_new_org") == 0
    assert at("U1", 40, "g8_dom_new_org") == 1 and at("U2", 40, "g8_dom_new_org") == 1  # same day: both new
    assert at("U1", 40, "g8_dom_new_user") == 1 and at("U1", 39, "g8_dom_new_user") == 0
    assert at("U1", 40, "g8_khoi_dau") == 0
    assert at("U1", 0, "g6_ext_docs") == 1 and at("U1", 0, "g7_ext_rcpt") == 2 and at("U1", 0, "g7_bcc_ext") == 1
    assert at("U1", 0, "g7_att") == 2 and at("U1", 0, "g7_rcpt_distinct") == 3
    assert at("U1", 5, "g2_off_connect") == 1
    assert abs(at("U1", 0, "g3_session_h") - 8.0) < 1e-9
    # D: only the past; day 10 is the first with >= 10 days of history
    assert at("U1", 9, "d_lich_su_ngan") == 1 and at("U1", 10, "d_lich_su_ngan") == 0
    assert at("U1", 9, "z_g8_http") == 0.0 and at("U1", 9, "q_g8_http") == 1.0
    # U1 http: 1/day except d=3 (2). At d=10: past = 10 days, mean 1.1, sd 0.3
    assert abs(at("U1", 10, "z_g8_http") - (1 - 1.1) / (0.3 + 1)) < 1e-9
    assert abs(at("U1", 10, "q_g8_http") - 1 / (1.1 + 1)) < 1e-9


def test_e_features_independent_of_split_and_labels(tmp_path):
    import protocols_b as pb
    src = make_logs(tmp_path / "src", rich=False)
    out = tmp_path / "out"
    userday.extract(src, out, log=lambda m: None)
    f1 = userday.assemble(out)
    # the feature builder takes no label / split argument at all
    for fn in (userday.extract, userday.assemble):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"labels", "users", "insiders", "split", "fold", "train"}, params
    assert f1.equals(userday.assemble(out))
    # two different splits: the raw matrix is identical, only the scaler differs
    cols = userday.groups(f1, "4.2")["G"]
    X = userday.matrix(f1, cols)
    users = f1["user"].to_numpy()
    sA = pb.Scaler().fit(X[np.isin(users, USERS[:2])])
    sB = pb.Scaler().fit(X[np.isin(users, USERS[2:])])
    assert np.array_equal(X, userday.matrix(f1, cols))
    assert not np.allclose(sA.mean, sB.mean)


def test_b_splits_disjoint():
    import protocols_b as pb
    for rel in ["4.2", "6.2"]:
        p = ROOT / "data" / "userday" / f"labels_r{rel}.parquet"
        if not p.exists():
            pytest.skip("labels not built")
        lab = pl.read_parquet(p)
        users = lab["user"].to_numpy()
        runs = pb.pbu_splits(lab, rel)
        assert len(runs) == 10
        for sp in runs:
            fu, vu, tu = (set(users[sp[k]]) for k in ["fit", "val", "test"])
            assert not (fu & vu) and not (fu & tu) and not (vu & tu)
            assert len(sp["fit"]) + len(sp["val"]) + len(sp["test"]) == len(lab)
        sp = pb.pbt_split(lab, rel)
        days = lab["day"].to_numpy()
        tr, va, te = (days[sp[k]] for k in ["fit", "val", "test"])
        assert tr.max() < va.min() and va.max() < te.min()
        assert len(tr) + len(va) + len(te) == len(lab)


@pytest.mark.parametrize("rel", ["4.2", "6.2"])
def test_c_g1_matches_phaseA_counts(rel):
    p = ROOT / "data" / "userday" / f"r{rel}.parquet"
    if not p.exists():
        pytest.skip("features not built")
    f = pl.read_parquet(p)
    counts = pl.concat([pl.scan_parquet(ROOT / "data" / "counts" / f"r{rel}" / f"{t}.parquet")
                        for t in ["logon", "device", "file", "email", "http"]]).collect()
    names = {1: "logon", 2: "logoff", 3: "connect", 4: "disconnect", 5: "email", 6: "file", 7: "http"}
    c = (counts.with_columns(pl.col("act").replace_strict(names, return_dtype=pl.String).alias("a"),
                             ((pl.col("hour").cast(pl.Int32) - 1) // 6).alias("k"))
               .group_by("user", "day", "a", "k").agg(pl.col("n").sum()))
    piv = c.with_columns(pl.format("g1_{}_k{}", "a", "k").alias("col")).pivot(on="col", index=["user", "day"], values="n")
    g1 = [x for x in f.columns if x.startswith("g1_")]
    assert len(g1) == 28
    j = f.select(["user", "day"] + g1).join(piv, on=["user", "day"], how="full", coalesce=True, suffix="_A")
    assert len(j) == len(f) == len(piv), "user-day sets differ"
    for col in g1:
        a = j[col].fill_null(0).cast(pl.Int64)
        b = j[f"{col}_A"].fill_null(0).cast(pl.Int64) if f"{col}_A" in j.columns else pl.Series([0] * len(j))
        assert (a == b).all(), col
    assert int(f["n_rows"].sum()) == int(counts["n"].sum())
