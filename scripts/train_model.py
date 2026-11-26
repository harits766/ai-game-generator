"""Eksperimen >= 3 model untuk memprediksi minat publik minggu depan (data REAL).

Data  : Wikipedia pageviews mingguan + Google Trends Indonesia per jenis game.
Target: perubahan log(1 + views) minggu depan (delta). Prediksi level = level sekarang + delta.
        Memprediksi delta lebih jujur untuk data web yang mirip random walk: baseline naive = delta 0.
Split : berbasis waktu, 20 minggu terakhir = test (tanpa shuffle).
Metrik: MAE (log, ~ error relatif), MAPE (views), akurasi arah (naik/turun),
        Precision@10 ranking momentum (dipakai untuk Top-10 library).

Jalankan: python scripts/train_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.catalog import GAME_BY_ID  # noqa: E402
from core.config import MODEL_PATH, MODEL_REPORT, PROCESSED_DIR, TOP_N  # noqa: E402
from core.trend_model import FEATURES, TARGET, build_panel  # noqa: E402

TEST_WEEKS = 20
SEED = 42


def candidate_models() -> dict:
    return {
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Huber Regression (robust)": make_pipeline(StandardScaler(), HuberRegressor(alpha=1.0, max_iter=2000)),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.5,
                                               n_jobs=-1, random_state=SEED),
        "Gradient Boosting (HistGB)": HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=200, learning_rate=0.03, max_depth=3, l2_regularization=1.0,
            random_state=SEED),
        "MLP (Neural Net)": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=2000, early_stopping=True, random_state=SEED)),
    }


def momentum_p10(df: pd.DataFrame, pred_col: str) -> float:
    """Top-10 game (katalog) dengan momentum tertinggi: prediksi vs aktual."""
    d = df[df["game_id"].isin(GAME_BY_ID)]
    scores = []
    for _, wk in d.groupby("week"):
        if len(wk) < TOP_N + 1:
            continue
        act = set(wk.assign(m=wk[TARGET] - wk["roll12"]).nlargest(TOP_N, "m")["game_id"])
        pr = set(wk.assign(m=wk[pred_col] - wk["roll12"]).nlargest(TOP_N, "m")["game_id"])
        scores.append(len(act & pr) / TOP_N)
    return float(np.mean(scores)) if scores else float("nan")


def evaluate(name: str, test: pd.DataFrame, p: np.ndarray) -> dict:
    y = test[TARGET].values
    views, pviews = np.expm1(y), np.expm1(p)
    mape = float(np.mean(np.abs(pviews - views) / np.maximum(views, 1)))
    direction = float(np.mean(np.sign(p - test["lv_t"].values) == np.sign(y - test["lv_t"].values)))
    return {"model": name, "MAE_log": round(float(mean_absolute_error(y, p)), 4),
            "MAPE_views": round(mape, 4), "R2_log": round(float(r2_score(y, p)), 4),
            "Akurasi_arah": round(direction, 4),
            "Precision@10_momentum": round(momentum_p10(test.assign(_p=p), "_p"), 4)}


def main() -> dict:
    panel = build_panel()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PROCESSED_DIR / "trend_panel.csv", index=False)
    data = panel.dropna(subset=[TARGET])
    weeks = sorted(data["week"].unique())
    cut = weeks[-TEST_WEEKS]
    train, test = data[data["week"] < cut], data[data["week"] >= cut]

    results = [evaluate("Baseline (naive: = minggu ini)", test, test["lv_t"].values),
               evaluate("Baseline (rata-rata 4 minggu)", test, test["roll4"].values)]
    fitted = {}
    for name, m in candidate_models().items():
        m.fit(train[FEATURES], train[TARGET] - train["lv_t"])
        p = test["lv_t"].values + m.predict(test[FEATURES])
        results.append(evaluate(name, test, p))
        fitted[name] = m
        r = results[-1]
        print(f"{name:28s} MAE_log={r['MAE_log']:.4f} arah={r['Akurasi_arah']:.3f} P@10={r['Precision@10_momentum']:.3f}")
    best = min((r for r in results if r["model"] in fitted), key=lambda r: r["MAE_log"])["model"]
    final = candidate_models()[best].fit(data[FEATURES], data[TARGET] - data["lv_t"])

    if hasattr(fitted[best], "feature_importances_"):
        imp = dict(zip(FEATURES, map(float, fitted[best].feature_importances_)))
    else:
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(fitted[best], test[FEATURES], test[TARGET] - test["lv_t"], n_repeats=5,
                                    random_state=SEED)
        imp = dict(zip(FEATURES, map(float, pi.importances_mean)))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "features": FEATURES, "name": best, "target": "delta"}, MODEL_PATH, compress=3)
    report = {
        "task": "Prediksi perubahan log(1+Wikipedia views) minggu depan per jenis game (data real)",
        "best_model": best, "selection_metric": f"MAE_log (test {TEST_WEEKS} minggu terakhir)",
        "data_weeks": [str(weeks[0])[:10], str(weeks[-1])[:10]], "test_from": str(cut)[:10],
        "n_series": int(data["game_id"].nunique()), "n_train": int(len(train)), "n_test": int(len(test)),
        "google_trends_coverage": round(float(data["has_gt"].mean()), 3),
        "results": results,
        "feature_importance": dict(sorted(imp.items(), key=lambda kv: -kv[1])),
    }
    MODEL_REPORT.write_text(json.dumps(report, indent=2))
    print("BEST:", best)
    return report


if __name__ == "__main__":
    main()
