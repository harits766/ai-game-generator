"""Eksperimen >= 3 model untuk memprediksi permintaan game minggu depan, simpan yang terbaik.

Split berbasis waktu (bukan acak) agar tidak bocor: 26 minggu terakhir = test.
Metrik: MAE, RMSE, R2, dan Precision@10 (seberapa tepat Top-10 hasil prediksi
dibanding Top-10 aktual tiap minggu) karena itu yang dipakai aplikasi.

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
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import MODEL_PATH, MODEL_REPORT, PROCESSED_DIR, TOP_N  # noqa: E402
from core.features import FEATURES, TARGET, build_panel, clean_requests, load_raw  # noqa: E402

TEST_WEEKS = 26
SEED = 42


def candidate_models() -> dict:
    return {
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Poisson GLM": make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.1, max_iter=2000)),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                               n_jobs=-1, random_state=SEED),
        "Gradient Boosting (HistGB)": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, l2_regularization=0.1, random_state=SEED),
        "MLP (Neural Net)": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=1500, early_stopping=True,
            random_state=SEED)),
    }


def precision_at_k(df: pd.DataFrame, pred_col: str, k: int = TOP_N) -> float:
    scores = []
    for _, wk in df.groupby("week"):
        actual = set(wk.nlargest(k, TARGET)["game_id"])
        pred = set(wk.nlargest(k, pred_col)["game_id"])
        scores.append(len(actual & pred) / k)
    return float(np.mean(scores))


def evaluate(y, p, df, name) -> dict:
    d = df.assign(_p=p)
    return {
        "model": name,
        "MAE": round(float(mean_absolute_error(y, p)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 3),
        "R2": round(float(r2_score(y, p)), 4),
        "Precision@10": round(precision_at_k(d, "_p"), 4),
    }


def main() -> dict:
    reqs, sig = load_raw()
    panel = build_panel(clean_requests(reqs), sig)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PROCESSED_DIR / "weekly_panel.csv", index=False)

    data = panel[panel["active"] & panel[TARGET].notna()].copy()
    weeks = sorted(data["week"].unique())
    cut = weeks[-TEST_WEEKS]
    train, test = data[data["week"] < cut], data[data["week"] >= cut]
    Xtr, ytr, Xte, yte = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]

    results = [evaluate(yte, test["req_t"].values, test, "Baseline (naive: = minggu ini)")]
    fitted = {}
    for name, model in candidate_models().items():
        model.fit(Xtr, ytr)
        p = np.clip(model.predict(Xte), 0, None)
        results.append(evaluate(yte, p, test, name))
        fitted[name] = model
        print(f"{name:32s} MAE={results[-1]['MAE']:.3f}  P@10={results[-1]['Precision@10']:.3f}")

    ranked = sorted([r for r in results if r["model"] in fitted], key=lambda r: r["MAE"])
    best = ranked[0]["model"]
    final = candidate_models()[best]
    final.fit(data[FEATURES], data[TARGET])  # refit di seluruh data untuk deploy

    importance = {}
    if hasattr(fitted[best], "feature_importances_"):
        importance = dict(zip(FEATURES, map(float, fitted[best].feature_importances_)))
    else:
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(fitted[best], Xte, yte, n_repeats=5, random_state=SEED)
        importance = dict(zip(FEATURES, map(float, pi.importances_mean)))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "features": FEATURES, "name": best}, MODEL_PATH, compress=3)
    report = {
        "best_model": best,
        "selection_metric": "MAE (test, 26 minggu terakhir)",
        "train_weeks": [str(weeks[0])[:10], str(cut)[:10]],
        "n_train": int(len(train)), "n_test": int(len(test)),
        "results": results,
        "feature_importance": dict(sorted(importance.items(), key=lambda kv: -kv[1])),
    }
    MODEL_REPORT.write_text(json.dumps(report, indent=2))
    print("BEST:", best)
    return report


if __name__ == "__main__":
    main()
