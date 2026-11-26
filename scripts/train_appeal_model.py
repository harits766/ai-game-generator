"""Eksperimen >=3 model klasifikasi daya tarik game party (data REAL BoardGameGeek).

Target: populer = users_rated di 25% teratas (game terbit 2000-2021, agar game lama/baru tidak bias).
Evaluasi: Stratified 5-fold CV (ROC-AUC, F1) + holdout 20%.
Jalankan: python scripts/train_appeal_model.py
"""
import json
import sys
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.appeal import APPEAL_PATH, featurize, load_bgg, vocab  # noqa: E402
from core.config import MODELS_DIR  # noqa: E402

SEED = 42


def models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5,
                                                                                 class_weight="balanced")),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                                n_jobs=-1, random_state=SEED),
        "Gradient Boosting (HistGB)": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                                     class_weight="balanced", random_state=SEED),
        "MLP (Neural Net)": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                                                          max_iter=800, early_stopping=True,
                                                                          random_state=SEED)),
    }


def main():
    df = load_bgg()
    df = df[df["yearpublished"].between(2000, 2021)].reset_index(drop=True)
    thr = df["users_rated"].quantile(0.75)
    y = (df["users_rated"] >= thr).astype(int)
    cats, mechs = vocab(df)
    X = featurize(df, cats, mechs)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results, fitted = [], {}
    base = {"model": "Baseline (selalu 'tidak populer')", "CV_ROC_AUC": 0.5, "CV_F1": 0.0,
            "Test_ROC_AUC": 0.5, "Test_F1": 0.0}
    results.append(base)
    for name, m in models().items():
        s = cross_validate(m, Xtr, ytr, cv=cv, scoring=["roc_auc", "f1"], n_jobs=1)
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        results.append({"model": name, "CV_ROC_AUC": round(float(s["test_roc_auc"].mean()), 4),
                        "CV_F1": round(float(s["test_f1"].mean()), 4),
                        "Test_ROC_AUC": round(float(roc_auc_score(yte, p)), 4),
                        "Test_F1": round(float(f1_score(yte, (p >= 0.5).astype(int))), 4)})
        fitted[name] = m
        print(f"{name:28s} CV AUC={results[-1]['CV_ROC_AUC']:.3f} test AUC={results[-1]['Test_ROC_AUC']:.3f}")
    best = max(results[1:], key=lambda r: r["CV_ROC_AUC"])["model"]
    final = models()[best].fit(X, y)
    imp = {}
    if hasattr(final, "feature_importances_"):
        imp = dict(zip(X.columns, map(float, final.feature_importances_)))
    else:
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(fitted[best], Xte, yte, scoring="roc_auc", n_repeats=5, random_state=SEED)
        imp = dict(zip(X.columns, map(float, pi.importances_mean)))
    joblib.dump({"model": final, "features": list(X.columns), "cats": cats, "mechs": mechs, "name": best,
                 "threshold_users_rated": float(thr)}, APPEAL_PATH, compress=3)
    report = {"best_model": best, "selection_metric": "ROC-AUC (5-fold CV)", "n_games": int(len(df)),
              "positive_rate": round(float(y.mean()), 3), "threshold_users_rated": float(thr),
              "results": results,
              "feature_importance": dict(sorted(imp.items(), key=lambda kv: -kv[1])[:25])}
    (MODELS_DIR / "appeal_report.json").write_text(json.dumps(report, indent=2))
    print("BEST:", best)
    return report


if __name__ == "__main__":
    main()
