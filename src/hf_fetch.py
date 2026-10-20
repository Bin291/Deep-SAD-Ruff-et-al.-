"""Download files from the HF dataset repo and verify sha256 against lfs.oid.

Transport is curl.exe (Windows Schannel): from this machine, Python/OpenSSL TLS
handshakes to huggingface.co are reset most of the time, while curl succeeds on
~60% of connections. curl runs with retry + backoff + resume (-C -).

Usage:
  python src/hf_fetch.py manifest                      # write results/hf_manifest.json
  python src/hf_fetch.py get PATTERN [PATTERN ...]     # download + verify (fnmatch patterns)
  python src/hf_fetch.py verify-local LOCAL_DIR PREFIX PATTERN [...]
        # hash LOCAL_DIR/<rel> against HF PREFIX/<rel>; reusable if they match
  python src/hf_fetch.py size PATTERN [...]            # total bytes of matching files

HF_TOKEN is read from the environment and passed to curl on stdin, so it never
appears on a command line, in a log or in a file. Every action is appended to
results/hash_log.jsonl.
"""
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

REPO = "nguynn/cert-insider-threat-datasets"
API = f"https://huggingface.co/api/datasets/{REPO}"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("CERT_DATA_DIR", r"D:\Code\paper\data\hf_full"))
MANIFEST = ROOT / "results" / "hf_manifest.json"
HASH_LOG = ROOT / "results" / "hash_log.jsonl"
MAX_ATTEMPTS = 30


def _auth_header():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        sys.exit("HF_TOKEN is not set")
    return f"Authorization: Bearer {tok}\n"


def curl(args, out=None):
    """Run curl with the auth header fed on stdin. Returns (exit_code, stdout_bytes)."""
    cmd = ["curl.exe", "-sS", "-L", "--fail", "-H", "@-", "--connect-timeout", "20"] + args
    if out:
        cmd += ["-o", str(out)]
    p = subprocess.run(cmd, input=_auth_header().encode(), capture_output=True)
    return p.returncode, p.stdout, p.stderr.decode(errors="replace").strip()


def curl_retry(args, out=None, what=""):
    """Retry with exponential backoff. Returns (stdout, attempts_used)."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        rc, so, se = curl(args, out)
        if rc == 0:
            return so, attempt
        if rc == 22:  # HTTP error (401/403/404): retrying won't help
            sys.exit(f"HTTP error for {what}: {se}")
        wait = min(120, 2 ** min(attempt, 7))
        print(f"[retry {attempt}/{MAX_ATTEMPTS}] {what}: curl exit {rc} ({se[:80]}); sleep {wait}s", flush=True)
        time.sleep(wait)
    sys.exit(f"giving up on {what} after {MAX_ATTEMPTS} attempts")


def sha256_file(path, bufsize=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path):
    # Non-LFS files are identified by their git blob sha1 instead of sha256.
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def build_manifest():
    so, _ = curl_retry([f"{API}/revision/main"], what="revision")
    rev = json.loads(so)["sha"]
    files, url = {}, f"{API}/tree/{rev}?recursive=true&expand=false"
    while url:  # tree API is paginated via the Link header
        hdr = ROOT / "results" / ".tree_headers"
        so, _ = curl_retry(["-D", str(hdr), url], what="tree")
        for e in json.loads(so):
            if e["type"] != "file":
                continue
            lfs = e.get("lfs")
            files[e["path"]] = {"size": e["size"], "sha256": lfs["oid"] if lfs else None, "blob_id": e["oid"]}
        url = None
        for line in hdr.read_text(errors="replace").splitlines():
            if line.lower().startswith("link:") and 'rel="next"' in line:
                url = line.split("<", 1)[1].split(">", 1)[0]
        hdr.unlink()
    out = {"repo": REPO, "revision": rev, "files": files}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(out, indent=1))
    return out


def load_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return build_manifest()


def check(path, meta):
    if meta["sha256"]:
        got = sha256_file(path)
        return got, got == meta["sha256"]
    got = git_blob_sha1(path)
    return got, got == meta["blob_id"]


def log(rec):
    HASH_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(HASH_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def select(manifest, patterns):
    names = sorted(manifest["files"])
    return [n for n in names if any(fnmatch.fnmatch(n, p) for p in patterns)]


def get(patterns):
    m = load_manifest()
    rev = m["revision"]
    targets = select(m, patterns)
    if not targets:
        sys.exit(f"no files match {patterns}")
    bad = []
    for name in targets:
        meta = m["files"][name]
        dest = DATA_DIR / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size == meta["size"]:
            got, ok = check(dest, meta)
            if ok:
                log({"file": name, "source": "HF (already downloaded)", "size": meta["size"], "hash": got, "match": True, "attempts": 0})
                continue
            dest.unlink()  # complete-size but wrong content: inside our own data dir, refetch
        part = dest.with_name(dest.name + ".part")
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{rev}/{urllib.parse.quote(name)}"
        t0 = time.time()
        # -C - resumes from the bytes already in .part after a reset
        _, attempts = curl_retry(["-C", "-", url], out=part, what=name)
        os.replace(part, dest)
        got, ok = check(dest, meta)
        log({"file": name, "source": "HF", "size": meta["size"], "hash": got, "match": ok,
             "attempts": attempts, "seconds": round(time.time() - t0, 1)})
        if not ok:
            bad.append(name)
    if bad:
        sys.exit(f"HASH MISMATCH - stop and report: {bad}")


def verify_local(local_dir, prefix, patterns):
    m = load_manifest()
    for name in select(m, [f"{prefix}/{p}" for p in patterns]):
        meta = m["files"][name]
        local = Path(local_dir) / name[len(prefix) + 1:]
        rec = {"file": name, "source": "local", "local": str(local)}
        if not local.exists():
            log({**rec, "match": False, "note": "missing"})
        elif local.stat().st_size != meta["size"]:
            log({**rec, "match": False, "note": f"size {local.stat().st_size} != {meta['size']}"})
        else:
            t0 = time.time()
            got, ok = check(local, meta)
            log({**rec, "size": meta["size"], "hash": got, "match": ok, "seconds": round(time.time() - t0, 1)})


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["help"]
    if cmd == "manifest":
        m = build_manifest()
        print("revision", m["revision"], "files", len(m["files"]))
    elif cmd == "get":
        get(args)
    elif cmd == "verify-local":
        verify_local(args[0], args[1], args[2:])
    elif cmd == "size":
        m = load_manifest()
        sel = select(m, args)
        print(len(sel), "files", sum(m["files"][n]["size"] for n in sel), "bytes")
    else:
        sys.exit(__doc__)
