"""Konfigurasi global dan ATURAN KUNCI (batas komputasi & desain game).

Semua batas di sini dipakai oleh validator (core/rules.py), cache, room store,
dan agent. Ubah di satu tempat ini saja.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RUNTIME_DIR = DATA_DIR / "runtime"
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

GROUND_TRUTH_MD = DOCS_DIR / "GROUND_TRUTH.md"
GROUND_TRUTH_HISTORY = DOCS_DIR / "ground_truth_history.json"
LIBRARY_STATE = RUNTIME_DIR / "library_state.json"
RUNTIME_REQUESTS = RUNTIME_DIR / "requests_runtime.csv"
MODEL_PATH = MODELS_DIR / "popularity_model.joblib"
MODEL_REPORT = MODELS_DIR / "model_report.json"

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
TOP_N = 10                      # jumlah game aktif di library
MULTIVERSE_SLOTS = 1            # container eksperimen untuk game baru/custom
MULTIVERSE_GRACE_DAYS = 7       # umur minimal sebelum game multiverse dinilai
MULTIVERSE_MIN_REQUESTS = 5     # di bawah ini setelah grace period -> dihapus
TOP10_STABLE_OVERLAP = 8        # overlap >= 8/10 dengan versi sebelumnya = STABIL

# ---------------------------------------------------------------------------
# ATURAN KUNCI untuk game custom (dan game hasil generate AI)
# ---------------------------------------------------------------------------
KEY_RULES = {
    "min_rules": 4,
    "max_rules": 7,
    "required_roles": ("host", "peserta"),
    "min_players": 2,
    "max_players": 30,          # batas aman 1 instance Streamlit Cloud gratis
    "min_rounds": 3,
    "max_rounds": 15,
    "min_seconds_per_round": 10,
    "max_seconds_per_round": 120,
    "max_session_minutes": 60,
    "max_rule_chars": 160,
    "max_name_chars": 40,
}

# Batas komputasi
MAX_UPLOAD_MB = 5
MAX_DOC_CHARS = 20_000          # dokumen dipotong ke panjang ini sebelum diproses
MAX_LLM_CALLS_PER_ROOM = 2      # konten game dibuat max 2 panggilan LLM
MAX_AGENT_STEPS = 6             # langkah tool-calling maksimum per chat
MAX_ACTIVE_ROOMS = 20
ROOM_IDLE_TTL_MIN = 120         # room dihapus kalau idle > 2 jam
CACHE_TTL_SEC = 30 * 60         # konten game di-cache 30 menit
CACHE_MAXSIZE = 64

SUPPORTED_DOC_TYPES = ("pdf", "docx", "txt", "md")

EVENT_TYPES = [
    "kajian", "kelas/sekolah", "kantor", "keluarga",
    "komunitas", "sosialisasi desa", "seminar",
]
MODES = ["offline", "online", "hybrid"]
LOCATIONS = [
    "DKI Jakarta", "Jawa Barat", "Jawa Tengah", "DI Yogyakarta", "Jawa Timur",
    "Banten", "Bali", "Sumatera Utara", "Sumatera Barat", "Sulawesi Selatan",
    "Kalimantan Timur", "Nusa Tenggara Barat",
]
DURATIONS = [10, 15, 20, 30, 45, 60]
