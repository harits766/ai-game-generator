"""Penyimpanan permanen log request in-app di branch `data` repo GitHub (gratis).

Streamlit Cloud menghapus file lokal setiap app restart/tidur. Supaya log request
(yang mulai dari NOL dan terus bertambah) tidak hilang:
- setiap request ditulis ke file lokal (cepat), lalu
- disinkronkan ke GitHub (branch `data`, file requests_runtime.csv) di thread latar,
- saat app baru nyala, file lokal diisi ulang dari GitHub.
GitHub Actions mingguan membaca file yang sama untuk retrain & update ground truth.

Butuh secret GITHUB_TOKEN (fine-grained token, izin Contents: Read and write untuk repo ini).
Tanpa token, log hanya lokal (tetap jalan, tapi hilang saat restart).
"""
from __future__ import annotations

import base64
import threading
import time

import requests

from .config import RUNTIME_DIR, RUNTIME_REQUESTS
from .llm import _secret

API = "https://api.github.com"
_lock = threading.Lock()
_status = {"backend": "lokal", "last_sync": None, "last_error": None, "pulled": False}
_dirty = threading.Event()
_worker: threading.Thread | None = None


def _cfg():
    return (_secret("GITHUB_TOKEN"), _secret("GITHUB_REPO", "harits766/ai-game-generator"),
            _secret("DATA_BRANCH", "data"))


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


def status() -> dict:
    token, repo, branch = _cfg()
    return {**_status, "backend": f"GitHub {repo}@{branch}" if token else "lokal (hilang saat restart)"}


def _get_remote(token, repo, branch) -> tuple[str | None, str | None]:
    r = requests.get(f"{API}/repos/{repo}/contents/requests_runtime.csv", params={"ref": branch},
                     headers=_headers(token), timeout=15)
    if r.status_code == 404:
        return None, None
    r.raise_for_status()
    j = r.json()
    return base64.b64decode(j["content"]).decode(), j["sha"]


def _ensure_branch(token, repo, branch):
    h = _headers(token)
    if requests.get(f"{API}/repos/{repo}/branches/{branch}", headers=h, timeout=15).status_code == 200:
        return
    default = requests.get(f"{API}/repos/{repo}", headers=h, timeout=15).json()["default_branch"]
    sha = requests.get(f"{API}/repos/{repo}/git/ref/heads/{default}", headers=h, timeout=15).json()["object"]["sha"]
    requests.post(f"{API}/repos/{repo}/git/refs", headers=h, json={"ref": f"refs/heads/{branch}", "sha": sha},
                  timeout=15)


def pull_once() -> None:
    """Isi file lokal dari GitHub saat app baru nyala (sekali per proses)."""
    if _status["pulled"]:
        return
    _status["pulled"] = True
    token, repo, branch = _cfg()
    if not token:
        return
    try:
        text, _ = _get_remote(token, repo, branch)
        if text and not RUNTIME_REQUESTS.exists():
            RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
            RUNTIME_REQUESTS.write_text(text)
    except Exception as e:
        _status["last_error"] = f"pull: {type(e).__name__}"


def _merge(remote: str | None, local: str) -> str:
    """Gabung baris unik (berdasarkan request_id di kolom pertama)."""
    lines_l = local.strip().splitlines()
    if not remote:
        return local if local.endswith("\n") else local + "\n"
    lines_r = remote.strip().splitlines()
    header = lines_r[0]
    seen = {ln.split(",", 1)[0] for ln in lines_r[1:]}
    extra = [ln for ln in lines_l[1:] if ln.split(",", 1)[0] not in seen]
    return "\n".join([header] + lines_r[1:] + extra) + "\n"


def _push() -> None:
    token, repo, branch = _cfg()
    if not token or not RUNTIME_REQUESTS.exists():
        return
    with _lock:
        local = RUNTIME_REQUESTS.read_text()
    for _ in range(3):
        try:
            _ensure_branch(token, repo, branch)
            remote, sha = _get_remote(token, repo, branch)
            merged = _merge(remote, local)
            if remote == merged:
                break
            body = {"message": "data: log request in-app", "branch": branch,
                    "content": base64.b64encode(merged.encode()).decode()}
            if sha:
                body["sha"] = sha
            r = requests.put(f"{API}/repos/{repo}/contents/requests_runtime.csv", headers=_headers(token),
                             json=body, timeout=20)
            if r.status_code in (409, 422):      # sha basi (ada penulis lain) -> ulangi
                time.sleep(1)
                continue
            r.raise_for_status()
            with _lock:
                RUNTIME_REQUESTS.write_text(merged)
            _status["last_sync"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _status["last_error"] = None
            break
        except Exception as e:
            _status["last_error"] = f"push: {type(e).__name__} {str(e)[:80]}"
            time.sleep(2)


def _loop():
    while True:
        _dirty.wait()
        time.sleep(5)          # kumpulkan beberapa request dulu (debounce)
        _dirty.clear()
        _push()


def schedule_sync() -> None:
    global _worker
    if not _cfg()[0]:
        return
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=_loop, daemon=True, name="gh-sync")
        _worker.start()
    _dirty.set()
