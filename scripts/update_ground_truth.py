"""Regenerate docs/GROUND_TRUTH.md dan cek apakah Top-10 masih sama.

Jalankan: python scripts/update_ground_truth.py
Exit code 0 = integritas lulus (verdict STABIL/BERUBAH/BASELINE hanya informasi).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.ground_truth import update_ground_truth  # noqa: E402

if __name__ == "__main__":
    snap = update_ground_truth()
    c = snap["compare"]
    print(f"Verdict: {c['verdict']} | overlap {c['overlap']}/10 | masuk {c['masuk']} | keluar {c['keluar']}")
    print(f"Backtest P@10: {snap['backtest']['precision_at_10']:.2f} | integritas: {snap['integrity_ok']}")
    sys.exit(0 if snap["integrity_ok"] else 1)
