"""Probe 5: Ceiling analysis — how much signal is in this data?

1. Train on full data, test on full data (overfitting ceiling)
2. Add time_to_event as a feature (information ceiling — cheating)
3. Random label permutation (noise floor)
4. Bayes-optimal estimate via calibration analysis
"""
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "96"
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deployment_analysis import (
    load_dataset, resolve_preferred_dataset, add_secondary_horizon_labels,
    build_feature_sets, stratified_group_split, fit_feature_space,
    TARGET_COL, TIME_COL, SUBJECT_ID_COL, safe_auroc, safe_auprc,
    MISSINGNESS_DROP_THRESHOLD, columns_above_missingness,
    candidate_feature_columns, COHORT_TIME_COLS,
)

from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer

OUT = Path(__file__).resolve().parent.parent.parent / "local_outputs" / "probes"
OUT.mkdir(parents=True, exist_ok=True)

frame = add_secondary_horizon_labels(load_dataset(resolve_preferred_dataset()))

train, test = stratified_group_split(
    frame, group_col=SUBJECT_ID_COL, label_col=TARGET_COL,
    test_size=0.20, random_state=42,
)

all_features = candidate_feature_columns(frame, include_process=True)
gt99_drop = set(columns_above_missingness(frame, all_features, MISSINGNESS_DROP_THRESHOLD))
numeric_cols = [c for c in all_features if c not in gt99_drop and c not in COHORT_TIME_COLS and pd.api.types.is_numeric_dtype(frame[c])]

imp = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imp.fit_transform(train[numeric_cols]), columns=numeric_cols, index=train.index)
X_test = pd.DataFrame(imp.transform(test[numeric_cols]), columns=numeric_cols, index=test.index)
y_train = train[TARGET_COL].astype(int)
y_test = test[TARGET_COL].astype(int)

results = {}

def quick_lgbm(X_tr, y_tr, X_te, y_te, label):
    clf = LGBMClassifier(
        objective="binary", n_estimators=350, learning_rate=0.05,
        num_leaves=31, class_weight="balanced", n_jobs=96,
        random_state=42, verbose=-1,
    )
    clf.fit(X_tr.values, y_tr.values)
    probs = clf.predict_proba(X_te.values)[:, 1]
    auroc = safe_auroc(y_te, probs)
    auprc = safe_auprc(y_te, probs)
    print(f"{label}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
    return {"auroc": auroc, "auprc": auprc}

results["honest_grouped"] = quick_lgbm(X_train, y_train, X_test, y_test, "Honest grouped split")

results["overfit_train_on_train"] = quick_lgbm(X_train, y_train, X_train, y_train, "Overfit (train=test)")

X_train_cheat = X_train.copy()
X_test_cheat = X_test.copy()
X_train_cheat["time_to_event_hrs"] = train[TIME_COL].values
X_test_cheat["time_to_event_hrs"] = test[TIME_COL].values
results["with_time_to_event"] = quick_lgbm(X_train_cheat, y_train, X_test_cheat, y_test, "Cheating (time_to_event)")

rng = np.random.default_rng(42)
y_train_shuffled = pd.Series(rng.permutation(y_train.values), index=y_train.index)
results["random_labels"] = quick_lgbm(X_train, y_train_shuffled, X_test, y_test, "Random labels (noise floor)")

# Estimate Bayes-optimal by looking at how much overlap exists
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train.values, y_train.values)
lda_probs = lda.predict_proba(X_test.values)[:, 1]
lda_auroc = safe_auroc(y_test, lda_probs)
results["lda_discriminant"] = {"auroc": lda_auroc}
print(f"LDA (linear Bayes proxy): AUROC={lda_auroc:.4f}")

# Feature correlation with outcome
correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
top20 = correlations.head(20)
results["top_20_correlations"] = {k: round(v, 4) for k, v in top20.items()}
print(f"\nTop 5 features by |correlation| with outcome:")
for feat, corr in top20.head(5).items():
    print(f"  {feat}: {corr:.4f}")

with open(OUT / "probe_ceiling_analysis.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {OUT / 'probe_ceiling_analysis.json'}")
