#!/usr/bin/env python3
"""
H3 — AE / VAE kiểu Pantelidis et al. (2021) chạy trên ma trận user-day 90 chiều của bạn.

Mục đích: cho người tạo source (partner) tự so với Hybrid TAE + Deep SVDD/SAD trên CÙNG dữ liệu,
CÙNG cách chia thời gian, CÙNG bộ chỉ số (AUC-PR, ROC-AUC, rec@k, recall theo kịch bản).

Cái gì GIỐNG bài:   AE 100-50-50-50-100 (bottleneck 50), VAE lớp ẩn 100-50, latent 2,
                    học trên dữ liệu bình thường, điểm = lỗi tái tạo.
Cái gì KHÁC bài (bắt buộc, ghi vào luận văn):
  1. Đầu vào là vector user-day 90 chiều (liên tục) chứ KHÔNG phải one-hot (activity, thứ, giờ) 38 chiều.
  2. Đầu ra tuyến tính + MSE (bài dùng sigmoid/BCE cho đầu vào one-hot).
  3. Adam lr 1e-3, batch 256, dừng sớm theo mất mát trên phần giữ riêng (bài: 30 epoch AE, 1000 epoch VAE, Nadam).
  4. PyTorch thay Keras.

Không chỉnh gì theo test: epoch tốt nhất chọn bằng mất mát tái tạo trên N ngày cuối kỳ train
(chỉ dòng bình thường); test chỉ được chấm MỘT lần cho mỗi (mô hình, seed) sau khi huấn luyện xong.

Cách dùng nhanh (file local):
  python h3_pantelidis_userday.py --features features_userday.parquet --labels labels_userday.parquet --inspect
  python h3_pantelidis_userday.py --features features_userday.parquet --labels labels_userday.parquet --out out_h3

Lấy từ Hugging Face (HF_TOKEN trong biến môi trường, không ghi vào file; file tải về được kiểm sha256 == lfs.oid):
  python h3_pantelidis_userday.py --hf-repo USER/REPO --hf-list                  # xem tên file trong repo
  python h3_pantelidis_userday.py --hf-repo USER/REPO --hf-revision <commit> \\
      --hf-features path/features_userday.parquet --hf-labels path/labels_userday.parquet --out out_h3
"""
import argparse, json, math, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

LABEL_CANDIDATES = ["label", "y", "is_malicious", "malicious", "is_insider", "insider", "target"]
SPLIT_CANDIDATES = ["split", "subset", "set", "partition"]
SCEN_CANDIDATES = ["scenario", "scen", "scenario_id", "kich_ban"]
DAY_CANDIDATES = ["day", "date", "ngay"]
USER_CANDIDATES = ["user", "user_id", "userid", "uid"]


# ----------------------------------------------------------------------------- I/O
def read_table(path):
    if path.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path)


# ----------------------------------------------------------------------------- Hugging Face
# Lấy file từ HF bằng curl (transport mà máy bạn đã dùng được; Python/OpenSSL từng bị reset TLS tới HF).
# - Token CHỈ đọc từ biến môi trường HF_TOKEN, gửi qua stdin (-H @-), không nằm trên dòng lệnh, log hay file.
# - Mọi file tải về phải có sha256 == lfs.oid trên HF (file không-LFS: git blob sha1 == oid), sai thì xoá và dừng.
import hashlib, shutil, subprocess, time as _time

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _hf_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _curl(extra, retries=6, capture=True):
    """Chạy curl với retry + backoff mũ; token (nếu có) đi qua stdin."""
    if shutil.which("curl") is None and shutil.which("curl.exe") is None:
        sys.exit("[LỖI] Không tìm thấy curl. Windows 10+ có sẵn curl.exe; hoặc dùng --features/--labels với file local.")
    tok = _hf_token()
    cmd = ["curl", "-sS", "-L", "--fail"] + (["-H", "@-"] if tok else []) + extra
    stdin = f"Authorization: Bearer {tok}\n" if tok else None
    last = None
    for i in range(retries):
        r = subprocess.run(cmd, input=stdin, capture_output=capture, text=True)
        if r.returncode == 0:
            return r.stdout
        last = (r.returncode, (r.stderr or "").strip()[:300])
        if r.returncode == 33:          # 416: file .part đã đủ byte
            return r.stdout
        if "401" in (r.stderr or "") or "403" in (r.stderr or ""):
            sys.exit("[LỖI] HF trả 401/403: repo gated hoặc token sai. Đặt HF_TOKEN (đã chấp nhận điều khoản dataset), "
                     "ví dụ PowerShell: $env:HF_TOKEN = '<token>'  (không ghi token vào file).")
        wait = min(60, 2 ** i)
        print(f"[HF] curl lỗi {last}; thử lại sau {wait}s ({i + 1}/{retries})")
        _time.sleep(wait)
    sys.exit(f"[LỖI] curl thất bại sau {retries} lần: {last}")


def hf_resolve_revision(repo, rev):
    out = _curl([f"{HF_ENDPOINT}/api/datasets/{repo}/revision/{rev}"])
    return json.loads(out)["sha"]


def hf_list(repo, rev, prefix=""):
    out = _curl([f"{HF_ENDPOINT}/api/datasets/{repo}/tree/{rev}?recursive=true"])
    items = [e for e in json.loads(out) if e.get("type") == "file" and e["path"].startswith(prefix)]
    for e in items:
        print(f"{e.get('size', 0):>14,}  {e['path']}")
    print(f"({len(items)} file; danh sách có thể bị cắt nếu repo rất lớn)")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_blob_sha1(path):
    size = os.path.getsize(path)
    h = hashlib.sha1(); h.update(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_fetch(repo, path_in_repo, rev, cache_dir):
    """Tải path_in_repo từ dataset `repo`@rev vào cache_dir, kiểm hash; trả về đường dẫn local."""
    sha_rev = hf_resolve_revision(repo, rev)
    body = json.dumps({"paths": [path_in_repo], "expand": False})
    info = json.loads(_curl(["-X", "POST", "-H", "Content-Type: application/json", "-d", body,
                             f"{HF_ENDPOINT}/api/datasets/{repo}/paths-info/{sha_rev}"]))
    if not info:
        sys.exit(f"[LỖI] Không có '{path_in_repo}' trong {repo}@{sha_rev[:8]}. Dùng --hf-list để xem file có trong repo.")
    e = info[0]
    expect, kind = (e["lfs"]["oid"], "sha256") if e.get("lfs") else (e["oid"], "git-blob-sha1")
    dest = os.path.join(cache_dir, repo.replace("/", "__"), sha_rev, *path_in_repo.split("/"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    check = _sha256 if kind == "sha256" else _git_blob_sha1
    if os.path.exists(dest) and check(dest) == expect:
        print(f"[HF] dùng bản cache đã kiểm: {path_in_repo} ({kind} khớp)")
        return dest
    part = dest + ".part"
    print(f"[HF] tải {repo}@{sha_rev[:8]}:{path_in_repo} ({e.get('size', 0):,} byte) ...")
    _curl(["-C", "-", "-o", part, f"{HF_ENDPOINT}/datasets/{repo}/resolve/{sha_rev}/{path_in_repo}"], capture=False)
    got = check(part)
    if got != expect:
        os.remove(part)
        sys.exit(f"[LỖI] {kind} không khớp HF cho {path_in_repo}: nhận {got[:12]}…, mong đợi {expect[:12]}…. Đã xoá file tải dở.")
    os.replace(part, dest)
    print(f"[HF] {kind} khớp lfs.oid/oid: {expect[:12]}…")
    return dest


def pick(cols, explicit, candidates, what, required=True):
    if explicit:
        if explicit not in cols:
            sys.exit(f"[LỖI] Cột {what} '{explicit}' không có. Các cột hiện có: {list(cols)[:40]}")
        return explicit
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c in low:
            return low[c]
    if required:
        sys.exit(f"[LỖI] Không tự nhận ra cột {what}. Chạy lại với --inspect để xem cột, rồi truyền tên bằng cờ tương ứng.")
    return None


def inspect(feat, lab):
    print("== features ==", feat.shape)
    print(feat.dtypes.value_counts().to_string())
    print("5 cột đầu:", list(feat.columns[:5]), "| 5 cột cuối:", list(feat.columns[-5:]))
    print("index:", feat.index.names, feat.index[:3].tolist())
    print("\n== labels ==", lab.shape)
    print(lab.head(5).to_string())
    for c in lab.columns:
        if lab[c].nunique() <= 12:
            print(f"- {c}: {lab[c].value_counts(dropna=False).to_dict()}")


def build_frame(args):
    feat = read_table(args.features)
    lab = read_table(args.labels)
    if args.inspect:
        inspect(feat, lab)
        sys.exit(0)

    ycol = pick(lab.columns, args.label_col, LABEL_CANDIDATES, "nhãn")
    scol = pick(lab.columns, args.split_col, SPLIT_CANDIDATES, "split", required=False)
    ncol = pick(lab.columns, args.scenario_col, SCEN_CANDIDATES, "kịch bản", required=False)
    dcol_l = pick(lab.columns, args.day_col, DAY_CANDIDATES, "ngày (labels)", required=False)
    ucol_l = pick(lab.columns, args.user_col, USER_CANDIDATES, "user (labels)", required=False)
    dcol_f = pick(feat.columns, args.day_col, DAY_CANDIDATES, "ngày (features)", required=False)
    ucol_f = pick(feat.columns, args.user_col, USER_CANDIDATES, "user (features)", required=False)

    # ghép features với labels
    if ucol_f and dcol_f and ucol_l and dcol_l:
        f = feat.copy(); l = lab.copy()
        f[dcol_f] = pd.to_datetime(f[dcol_f]); l[dcol_l] = pd.to_datetime(l[dcol_l])
        df = f.merge(l, left_on=[ucol_f, dcol_f], right_on=[ucol_l, dcol_l], how="inner", suffixes=("", "_lab"))
        if len(df) != len(feat):
            print(f"[CẢNH BÁO] ghép (user, day) còn {len(df)}/{len(feat)} dòng của features")
        day = df[dcol_f]; user = df[ucol_f]
    elif len(feat) == len(lab):
        print("[GHI CHÚ] không có cặp (user, day) ở cả hai file; ghép theo THỨ TỰ DÒNG (cùng số dòng). Hãy kiểm tra lại.")
        df = pd.concat([feat.reset_index(drop=True), lab.reset_index(drop=True)], axis=1, keys=["f", "l"])
        df.columns = [c[1] if c[0] == "l" else c[1] for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        day = pd.to_datetime(df[dcol_l or dcol_f]) if (dcol_l or dcol_f) else None
        user = df[ucol_l or ucol_f] if (ucol_l or ucol_f) else None
    else:
        sys.exit("[LỖI] features và labels khác số dòng và không có khoá (user, day) chung để ghép.")

    non_feat = {ycol, scol, ncol, dcol_l, dcol_f, ucol_l, ucol_f} - {None}
    non_feat |= {c for c in df.columns if c.endswith("_lab")}
    fcols = [c for c in feat.columns if c not in non_feat and pd.api.types.is_numeric_dtype(feat[c])]
    if args.n_features and len(fcols) != args.n_features:
        sys.exit(f"[LỖI] Tìm được {len(fcols)} cột đặc trưng, mong đợi {args.n_features}. Kiểm tra --inspect.")
    print(f"[OK] {len(df)} user-day, {len(fcols)} đặc trưng, nhãn='{ycol}', split='{scol}', kịch bản='{ncol}'")

    X = df[fcols].to_numpy(dtype=np.float64)
    y = df[ycol].to_numpy().astype(int)
    scen = df[ncol].to_numpy() if ncol else None
    # split
    if scol:
        sv = df[scol].astype(str).str.lower().to_numpy()
        is_train = np.char.find(sv.astype(str), "train") >= 0
        is_test = np.char.find(sv.astype(str), "test") >= 0
        dropped = len(df) - is_train.sum() - is_test.sum()
        print(f"[split] train={is_train.sum()} test={is_test.sum()} bỏ={dropped} (giá trị: {pd.Series(sv).value_counts().to_dict()})")
    else:
        if day is None:
            sys.exit("[LỖI] Không có cột split lẫn cột ngày để cắt theo --cutoff.")
        is_train = (day < pd.Timestamp(args.cutoff)).to_numpy()
        is_test = ~is_train
        print(f"[split] cắt theo ngày {args.cutoff}: train={is_train.sum()} test={is_test.sum()} "
              "(CHƯA bỏ các insider nằm hai phía; dùng cột split của bạn nếu có)")
    return X, y, scen, is_train, is_test, (day.to_numpy() if day is not None else None), fcols


# ----------------------------------------------------------------------------- tiền xử lý
def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def preprocess(X, y, is_train, args):
    Xp = X if args.no_log1p else signed_log1p(X)
    normal_train = is_train & (y == 0)
    mu = Xp[normal_train].mean(0)
    sd = Xp[normal_train].std(0)
    sd[sd < 1e-8] = 1.0
    Z = (Xp - mu) / sd
    Z = np.clip(Z, -args.clip, args.clip) if args.clip else Z
    return Z.astype(np.float32)


# ----------------------------------------------------------------------------- mô hình
class AE(nn.Module):
    """d -> 100 -> 50 -> 50 -> 50 -> 100 -> d (bottleneck = lớp 50 ở giữa)."""
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 100), nn.Tanh(), nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 50), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(50, 50), nn.ReLU(), nn.Linear(50, 100), nn.ReLU(), nn.Linear(100, d))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.dec(self.enc(x))

    def loss(self, x):
        return ((self(x) - x) ** 2).mean()

    @torch.no_grad()
    def score(self, x):
        return ((self(x) - x) ** 2).mean(1)


class VAE(nn.Module):
    """d -> 100 -> 50 -> (mu, logvar) latent 2 -> 50 -> 100 -> d."""
    def __init__(self, d, latent=2, beta=1.0):
        super().__init__()
        self.beta = beta
        self.enc = nn.Sequential(nn.Linear(d, 100), nn.Tanh(), nn.Linear(100, 50), nn.ReLU())
        self.mu = nn.Linear(50, latent); self.lv = nn.Linear(50, latent)
        self.dec = nn.Sequential(nn.Linear(latent, 50), nn.ReLU(), nn.Linear(50, 100), nn.ReLU(), nn.Linear(100, d))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.enc(x); mu, lv = self.mu(h), self.lv(h).clamp(-10, 10)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.dec(z), mu, lv

    def loss(self, x):
        xr, mu, lv = self(x)
        rec = ((xr - x) ** 2).sum(1).mean()          # tổng theo chiều, trung bình theo mẫu
        kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean()
        return (rec + self.beta * kl) / x.shape[1]    # chia d để cùng thang với AE

    @torch.no_grad()
    def score(self, x):
        h = self.enc(x); xr = self.dec(self.mu(h))   # giải mã tại μ, bỏ nhiễu
        return ((xr - x) ** 2).mean(1)


def fit(model, Xtr, Xva, device, args, seed):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    Xtr_t = torch.from_numpy(Xtr).to(device); Xva_t = torch.from_numpy(Xva).to(device)
    best, best_state, bad, hist = math.inf, None, 0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=g).to(device)
        tot, n = 0.0, 0
        for i in range(0, len(perm), args.batch):
            xb = Xtr_t[perm[i:i + args.batch]]
            opt.zero_grad(); loss = model.loss(xb); loss.backward(); opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        model.eval()
        with torch.no_grad():
            va = model.score(Xva_t).mean().item()      # lỗi tái tạo trên phần giữ riêng (chỉ dòng bình thường)
        hist.append((ep, tot / n, va))
        if va < best - 1e-6:
            best, bad = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return hist, best


# ----------------------------------------------------------------------------- chỉ số
def recall_at_budget(score, y, n_days, k):
    m = min(len(score), int(k * n_days))
    idx = np.argsort(-score, kind="stable")[:m]
    return float(y[idx].sum() / max(1, y.sum()))


def evaluate(score, y, scen, n_days, thr):
    out = {
        "auc_pr": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
    }
    for k in (5, 10, 25, 50):
        out[f"rec@{k}"] = recall_at_budget(score, y, n_days, k)
    pred = score > thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    out.update({"thr95_precision": p, "thr95_recall": r, "thr95_f1": (2 * p * r / (p + r)) if p + r else 0.0,
                "thr95_alerts": int(pred.sum())})
    if scen is not None:
        m = min(len(score), int(25 * n_days))
        top = np.zeros(len(score), bool); top[np.argsort(-score, kind="stable")[:m]] = True
        for s in sorted(pd.unique(scen[y == 1])):
            mask = (scen == s) & (y == 1)
            out[f"recall25_scen{s}"] = float(top[mask].sum() / max(1, mask.sum()))
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features"); ap.add_argument("--labels")
    ap.add_argument("--hf-repo", help="vd. nguynn/cert-insider-threat-datasets; cần HF_TOKEN nếu repo gated")
    ap.add_argument("--hf-features", help="đường dẫn file đặc trưng TRONG repo HF")
    ap.add_argument("--hf-labels", help="đường dẫn file nhãn TRONG repo HF")
    ap.add_argument("--hf-revision", default="main", help="nên ghim bằng mã commit để tái lập")
    ap.add_argument("--hf-cache", default="hf_cache"); ap.add_argument("--hf-list", nargs="?", const="", metavar="PREFIX",
                    help="liệt kê file trong repo (tuỳ chọn: lọc theo tiền tố) rồi thoát")
    ap.add_argument("--out", default="out_h3")
    ap.add_argument("--inspect", action="store_true", help="chỉ in cột/giá trị của hai file rồi thoát")
    ap.add_argument("--models", default="ae,vae"); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout-days", type=int, default=30, help="số ngày cuối kỳ train giữ riêng để chọn epoch")
    ap.add_argument("--cutoff", default="2010-10-01", help="chỉ dùng khi không có cột split")
    ap.add_argument("--label-col"); ap.add_argument("--split-col"); ap.add_argument("--scenario-col")
    ap.add_argument("--day-col"); ap.add_argument("--user-col")
    ap.add_argument("--n-features", type=int, default=90)
    ap.add_argument("--no-log1p", action="store_true", help="bỏ signed-log1p nếu ma trận của bạn đã được log/chuẩn hoá")
    ap.add_argument("--clip", type=float, default=0.0, help="cắt |z| tối đa (0 = không cắt)")
    ap.add_argument("--max-rows", type=int, default=0, help="chỉ để chạy thử nhanh: giới hạn số dòng train")
    args = ap.parse_args()

    if args.hf_repo and args.hf_list is not None:
        hf_list(args.hf_repo, hf_resolve_revision(args.hf_repo, args.hf_revision), args.hf_list)
        sys.exit(0)
    if args.hf_repo:
        if not (args.hf_features and args.hf_labels):
            sys.exit("[LỖI] Với --hf-repo cần cả --hf-features và --hf-labels (dùng --hf-list để xem tên file).")
        args.features = hf_fetch(args.hf_repo, args.hf_features, args.hf_revision, args.hf_cache)
        args.labels = hf_fetch(args.hf_repo, args.hf_labels, args.hf_revision, args.hf_cache)
    if not (args.features and args.labels):
        sys.exit("[LỖI] Cần --features và --labels (file local), hoặc --hf-repo + --hf-features + --hf-labels.")

    os.makedirs(args.out, exist_ok=True)
    X, y, scen, is_train, is_test, day, fcols = build_frame(args)
    Z = preprocess(X, y, is_train, args)

    # phần giữ riêng: N ngày cuối của kỳ train
    if day is None:
        sys.exit("[LỖI] Cần cột ngày để tách phần giữ riêng cuối kỳ train.")
    day = pd.to_datetime(day)
    last = day[is_train].max()
    hold = is_train & np.asarray(day > last - pd.Timedelta(days=args.holdout_days))
    fit_mask = is_train & ~hold & (y == 0)
    val_mask = hold & (y == 0)
    print(f"[giữ riêng] {args.holdout_days} ngày cuối train ({hold.sum()} dòng); fit={fit_mask.sum()} val bình thường={val_mask.sum()}")
    print(f"[test] {is_test.sum()} dòng, {int(y[is_test].sum())} dương, {pd.Series(day[is_test]).nunique()} ngày "
          f"(mức ngẫu nhiên AUC-PR ≈ {y[is_test].mean():.4f})")
    Xfit, Xval = Z[fit_mask], Z[val_mask]
    if args.max_rows:
        Xfit = Xfit[: args.max_rows]
    n_test_days = int(pd.Series(day[is_test]).nunique())
    Xte, yte = Z[is_test], y[is_test]
    scen_te = scen[is_test] if scen is not None else None
    train_normal_all = Z[is_train & (y == 0)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[thiết bị] {device}")
    rows = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        for seed in range(args.seeds):
            torch.manual_seed(seed); np.random.seed(seed)
            d = Z.shape[1]
            model = (AE(d) if name == "ae" else VAE(d)).to(device)
            t0 = time.time()
            hist, best_val = fit(model, Xfit, Xval, device, args, seed)
            model.eval()
            # ngưỡng 95% trên train bình thường (cùng quy tắc R² của bạn) — không dùng test
            thr = float(np.quantile(model.score(torch.from_numpy(train_normal_all).to(device)).cpu().numpy(), 0.95))
            # test được đọc MỘT lần ở đây, sau khi mô hình đã chốt
            s_te = model.score(torch.from_numpy(Xte).to(device)).cpu().numpy()
            res = evaluate(s_te, yte, scen_te, n_test_days, thr)
            res.update({"model": name, "seed": seed, "epochs_run": len(hist), "best_epoch": int(np.argmin([h[2] for h in hist]) + 1),
                        "best_val_recon": best_val, "train_sec": round(time.time() - t0, 1)})
            rows.append(res)
            print(f"[{name} seed {seed}] AUC-PR={res['auc_pr']:.4f} ROC={res['roc_auc']:.4f} rec@25={res['rec@25']:.4f} "
                  f"(epoch tốt nhất {res['best_epoch']}/{len(hist)}, {res['train_sec']}s)")
            pd.DataFrame(rows).to_csv(os.path.join(args.out, "h3_per_seed.csv"), index=False)

    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in ("model", "seed")]
    summ = df.groupby("model")[num].agg(["mean", "std"])
    summ.to_csv(os.path.join(args.out, "h3_summary.csv"))
    meta = {"args": vars(args), "n_features": len(fcols), "n_test_days": n_test_days,
            "n_test": int(is_test.sum()), "n_test_pos": int(yte.sum()), "random_auc_pr": float(yte.mean())}
    json.dump(meta, open(os.path.join(args.out, "h3_meta.json"), "w"), indent=2, default=str)
    key = ["auc_pr", "roc_auc", "rec@5", "rec@10", "rec@25", "rec@50"]
    print("\n== Trung bình ± std theo seed ==")
    for name, g in df.groupby("model"):
        print(name, " | ".join(f"{k} {g[k].mean():.4f}±{g[k].std(ddof=1) if len(g) > 1 else 0:.4f}" for k in key))
    print(f"mức ngẫu nhiên AUC-PR = {yte.mean():.4f}. Kết quả ở thư mục: {args.out}")


if __name__ == "__main__":
    main()