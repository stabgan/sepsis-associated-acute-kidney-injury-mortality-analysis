"""Consolidated evidence-gathering probe: runs all experiments serially.
Avoids CPU thrashing by running one at a time on all 96 cores.
"""
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict

CPU = os.cpu_count() or 96
os.environ["OMP_NUM_THREADS"] = str(CPU)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU)
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deployment_analysis import (
    load_dataset, resolve_preferred_dataset, add_secondary_horizon_labels,
    build_feature_sets, stratified_group_split, fit_feature_space,
    candidate_feature_columns, columns_above_missingness,
    TARGET_COL, TIME_COL, SUBJECT_ID_COL, safe_auroc, safe_auprc,
    binary_metrics, threshold_search,
    MISSINGNESS_DROP_THRESHOLD, COHORT_TIME_COLS,
)

OUT = Path(__file__).resolve().parent.parent.parent / "local_outputs" / "probes"
OUT.mkdir(parents=True, exist_ok=True)

t_global = time.time()
ALL = OrderedDict()

print("=" * 70)
print("LOADING DATA")
print("=" * 70)
frame = add_secondary_horizon_labels(load_dataset(resolve_preferred_dataset()))
print(f"  Loaded {len(frame)} rows, {frame.shape[1]} columns")

all_features = candidate_feature_columns(frame, include_process=True)
gt99_drop = set(columns_above_missingness(frame, all_features, MISSINGNESS_DROP_THRESHOLD))
numeric_cols = [c for c in all_features if c not in gt99_drop
                and c not in COHORT_TIME_COLS and pd.api.types.is_numeric_dtype(frame[c])]
print(f"  {len(numeric_cols)} numeric features after filtering")

feature_sets = build_feature_sets(frame)

train_full, test = stratified_group_split(
    frame, group_col=SUBJECT_ID_COL, label_col=TARGET_COL,
    test_size=0.20, random_state=42,
)
train, val = stratified_group_split(
    train_full, group_col=SUBJECT_ID_COL, label_col=TARGET_COL,
    test_size=0.20, random_state=42,
)

from sklearn.impute import SimpleImputer
imp_full = SimpleImputer(strategy="median")
X_train_full = pd.DataFrame(imp_full.fit_transform(train_full[numeric_cols]), columns=numeric_cols, index=train_full.index)
X_test = pd.DataFrame(imp_full.transform(test[numeric_cols]), columns=numeric_cols, index=test.index)
imp_tv = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imp_tv.fit_transform(train[numeric_cols]), columns=numeric_cols, index=train.index)
X_val = pd.DataFrame(imp_tv.transform(val[numeric_cols]), columns=numeric_cols, index=val.index)

y_train_full = train_full[TARGET_COL].astype(int)
y_test = test[TARGET_COL].astype(int)
y_train = train[TARGET_COL].astype(int)
y_val = val[TARGET_COL].astype(int)
time_train_full = train_full[TIME_COL].values
time_test = test[TIME_COL].values

print(f"  Train/val/test: {len(train)}/{len(val)}/{len(test)}")
print(f"  Event rates: train={y_train.mean():.3f} val={y_val.mean():.3f} test={y_test.mean():.3f}")

def quick_eval(probs, y_true, label=""):
    auroc = safe_auroc(y_true, probs)
    auprc = safe_auprc(y_true, probs)
    ppv50 = threshold_search(y_true, probs, precision_target=0.50)
    ppv60 = threshold_search(y_true, probs, precision_target=0.60)
    ppv70 = threshold_search(y_true, probs, precision_target=0.70)
    r = {
        "auroc": round(auroc, 4), "auprc": round(auprc, 4),
        "recall@ppv50": round(ppv50.get("recall", float("nan")), 4),
        "recall@ppv60": round(ppv60.get("recall", float("nan")), 4),
        "recall@ppv70": round(ppv70.get("recall", float("nan")), 4),
    }
    if label:
        print(f"  {label}: AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}  "
              f"recall@50={r['recall@ppv50']:.4f}  recall@60={r['recall@ppv60']:.4f}  recall@70={r['recall@ppv70']:.4f}")
    return r


# ======================================================================
# PROBE 1: CEILING ANALYSIS
# ======================================================================
print("\n" + "=" * 70)
print("PROBE 1: CEILING ANALYSIS")
print("=" * 70)
from lightgbm import LGBMClassifier

def mk_lgbm(**kw):
    defaults = dict(objective="binary", n_estimators=350, learning_rate=0.05,
                    num_leaves=31, class_weight="balanced", n_jobs=CPU,
                    random_state=42, verbose=-1)
    defaults.update(kw)
    return LGBMClassifier(**defaults)

t0 = time.time()
clf = mk_lgbm()
clf.fit(X_train_full.values, y_train_full.values)
p_test = clf.predict_proba(X_test.values)[:, 1]
ALL["ceiling_honest"] = quick_eval(p_test, y_test, "Honest grouped")

clf.fit(X_train_full.values, y_train_full.values)
p_train = clf.predict_proba(X_train_full.values)[:, 1]
ALL["ceiling_overfit"] = quick_eval(p_train, y_train_full, "Overfit (train=test)")

X_train_cheat = X_train_full.copy()
X_test_cheat = X_test.copy()
X_train_cheat["time_to_event_hrs"] = time_train_full
X_test_cheat["time_to_event_hrs"] = time_test
clf2 = mk_lgbm()
clf2.fit(X_train_cheat.values, y_train_full.values)
p_cheat = clf2.predict_proba(X_test_cheat.values)[:, 1]
ALL["ceiling_cheat_time"] = quick_eval(p_cheat, y_test, "Cheating (time_to_event)")

rng = np.random.default_rng(42)
y_shuffled = pd.Series(rng.permutation(y_train_full.values), index=y_train_full.index)
clf3 = mk_lgbm()
clf3.fit(X_train_full.values, y_shuffled.values)
p_noise = clf3.predict_proba(X_test.values)[:, 1]
ALL["ceiling_noise_floor"] = quick_eval(p_noise, y_test, "Noise floor (shuffled labels)")

corrs = X_train_full.corrwith(y_train_full).abs().sort_values(ascending=False)
ALL["top20_correlations"] = {k: round(v, 4) for k, v in corrs.head(20).items()}
print(f"  Top 5 features: {list(corrs.head(5).items())}")
print(f"  Ceiling analysis done in {time.time()-t0:.1f}s")


# ======================================================================
# PROBE 2: FEATURE ENGINEERING
# ======================================================================
print("\n" + "=" * 70)
print("PROBE 2: FEATURE ENGINEERING")
print("=" * 70)
t0 = time.time()

def add_missingness(X_tr, X_te, orig_tr, orig_te, cols):
    for c in cols:
        X_tr[f"{c}__miss"] = orig_tr[c].isna().astype(int).values
        X_te[f"{c}__miss"] = orig_te[c].isna().astype(int).values
    X_tr["total_missing"] = orig_tr[cols].isna().sum(axis=1).values
    X_te["total_missing"] = orig_te[cols].isna().sum(axis=1).values
    X_tr["missing_frac"] = X_tr["total_missing"] / len(cols)
    X_te["missing_frac"] = X_te["total_missing"] / len(cols)
    return X_tr, X_te

def add_clinical_ratios(X_tr, X_te):
    pairs = [
        ("bun_median", "creatinine_max", "bun_cr_ratio"),
        ("lactate_max", "bicarbonate_min", "lactate_bicarb_ratio"),
        ("sofa_total_24hr", "apache_iii_score", "sofa_apache_ratio"),
        ("urine_output_total_24hr", "fluid_balance_24hr_ml", "uo_fluid_ratio"),
        ("plateletcount_min", "wbc_max_10e9l", "plt_wbc_ratio"),
        ("ast_max", "alt_median", "ast_alt_ratio"),
        ("inr_max", "plateletcount_min", "inr_plt_ratio"),
        ("heartrate_max", "sysbp_min", "shock_index_proxy"),
        ("creatinine_max", "creatinine_first", "cr_trajectory"),
        ("lactate_max", "lactate_first", "lactate_trajectory"),
    ]
    for num, den, name in pairs:
        if num in X_tr.columns and den in X_tr.columns:
            X_tr[name] = X_tr[num] / (X_tr[den].abs() + 1e-6)
            X_te[name] = X_te[num] / (X_te[den].abs() + 1e-6)
    return X_tr, X_te

def add_delta_velocity(X_tr, X_te):
    bases = ["heartrate", "sysbp", "meanbp", "resprate", "spo2", "creatinine", "lactate", "temperature"]
    for b in bases:
        first, last, med = f"{b}_first", f"{b}_last", f"{b}_median"
        if first in X_tr.columns and last in X_tr.columns:
            X_tr[f"{b}_abs_delta"] = (X_tr[last] - X_tr[first]).abs()
            X_te[f"{b}_abs_delta"] = (X_te[last] - X_te[first]).abs()
        if med in X_tr.columns and first in X_tr.columns:
            X_tr[f"{b}_deviation"] = (X_tr[med] - X_tr[first]).abs()
            X_te[f"{b}_deviation"] = (X_te[med] - X_te[first]).abs()
    return X_tr, X_te

def add_organ_composites(X_tr, X_te):
    sofa_cols = [c for c in X_tr.columns if c.startswith("sofa_") and c.endswith("_24hr") and c != "sofa_total_24hr"]
    if sofa_cols:
        X_tr["sofa_max_component"] = X_tr[sofa_cols].max(axis=1)
        X_te["sofa_max_component"] = X_te[sofa_cols].max(axis=1)
        X_tr["sofa_nonzero_count"] = (X_tr[sofa_cols] > 0).sum(axis=1)
        X_te["sofa_nonzero_count"] = (X_te[sofa_cols] > 0).sum(axis=1)
    charlson_cols = [c for c in X_tr.columns if c.startswith("charlson_")]
    if charlson_cols:
        X_tr["charlson_total"] = X_tr[charlson_cols].sum(axis=1)
        X_te["charlson_total"] = X_te[charlson_cols].sum(axis=1)
    return X_tr, X_te

X_tr_b, X_te_b = X_train_full.copy(), X_test.copy()
ALL["feat_baseline"] = quick_eval(
    mk_lgbm().fit(X_tr_b.values, y_train_full.values).predict_proba(X_te_b.values)[:, 1],
    y_test, "Baseline")

X_tr_m, X_te_m = X_tr_b.copy(), X_te_b.copy()
X_tr_m, X_te_m = add_missingness(X_tr_m, X_te_m, train_full, test, numeric_cols)
ALL["feat_missingness"] = quick_eval(
    mk_lgbm().fit(X_tr_m.values, y_train_full.values).predict_proba(X_te_m.values)[:, 1],
    y_test, "+ Missingness")

X_tr_r, X_te_r = X_tr_m.copy(), X_te_m.copy()
X_tr_r, X_te_r = add_clinical_ratios(X_tr_r, X_te_r)
ALL["feat_ratios"] = quick_eval(
    mk_lgbm().fit(X_tr_r.values, y_train_full.values).predict_proba(X_te_r.values)[:, 1],
    y_test, "+ Ratios")

X_tr_d, X_te_d = X_tr_r.copy(), X_te_r.copy()
X_tr_d, X_te_d = add_delta_velocity(X_tr_d, X_te_d)
ALL["feat_deltas"] = quick_eval(
    mk_lgbm().fit(X_tr_d.values, y_train_full.values).predict_proba(X_te_d.values)[:, 1],
    y_test, "+ Deltas")

X_tr_o, X_te_o = X_tr_d.copy(), X_te_d.copy()
X_tr_o, X_te_o = add_organ_composites(X_tr_o, X_te_o)
ALL["feat_all_engineered"] = quick_eval(
    mk_lgbm().fit(X_tr_o.values, y_train_full.values).predict_proba(X_te_o.values)[:, 1],
    y_test, "+ Organ composites (FULL)")
n_engineered = X_tr_o.shape[1]
print(f"  Feature engineering done in {time.time()-t0:.1f}s  ({n_engineered} features)")


# ======================================================================
# PROBE 3: STACKED ENSEMBLE
# ======================================================================
print("\n" + "=" * 70)
print("PROBE 3: STACKED ENSEMBLE")
print("=" * 70)
t0 = time.time()
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier

estimators = [
    ("lgbm", LGBMClassifier(
        objective="binary", n_estimators=350, learning_rate=0.05,
        num_leaves=31, class_weight="balanced", n_jobs=CPU,
        random_state=42, verbose=-1,
    )),
    ("xgb", XGBClassifier(
        objective="binary:logistic", n_estimators=350, learning_rate=0.05,
        max_depth=4, scale_pos_weight=2.7, tree_method="hist",
        n_jobs=CPU, random_state=42,
    )),
    ("lr", LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=42)),
]

stacker = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000, solver="lbfgs"),
    cv=3, stack_method="predict_proba", n_jobs=1,
)
stacker.fit(X_train_full.values, y_train_full.values)
p_stack = stacker.predict_proba(X_test.values)[:, 1]
ALL["ensemble_stack_3"] = quick_eval(p_stack, y_test, "Stack (LGB+XGB+LR)")

for name, est in estimators:
    est.fit(X_train_full.values, y_train_full.values)
    p_i = est.predict_proba(X_test.values)[:, 1]
    ALL[f"individual_{name}"] = quick_eval(p_i, y_test, f"Individual {name}")

p_avg = np.mean([
    est.predict_proba(X_test.values)[:, 1]
    for _, est in estimators
], axis=0)
ALL["ensemble_avg_3"] = quick_eval(p_avg, y_test, "Simple average (LGB+XGB+LR)")

stacker_feat = StackingClassifier(
    estimators=[
        ("lgbm", LGBMClassifier(objective="binary", n_estimators=350, learning_rate=0.05,
                                num_leaves=31, class_weight="balanced", n_jobs=CPU,
                                random_state=42, verbose=-1)),
        ("xgb", XGBClassifier(objective="binary:logistic", n_estimators=350, learning_rate=0.05,
                              max_depth=4, scale_pos_weight=2.7, tree_method="hist",
                              n_jobs=CPU, random_state=42)),
        ("lr", LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=42)),
    ],
    final_estimator=LogisticRegression(max_iter=1000, solver="lbfgs"),
    cv=3, stack_method="predict_proba", n_jobs=1,
)
stacker_feat.fit(X_tr_o.values, y_train_full.values)
p_stack_feat = stacker_feat.predict_proba(X_te_o.values)[:, 1]
ALL["ensemble_stack_engineered"] = quick_eval(p_stack_feat, y_test, "Stack + engineered features")
print(f"  Ensemble done in {time.time()-t0:.1f}s")


# ======================================================================
# PROBE 4: CONFORMAL PREDICTION
# ======================================================================
print("\n" + "=" * 70)
print("PROBE 4: CONFORMAL PREDICTION")
print("=" * 70)
t0 = time.time()
X_tr_c = pd.DataFrame(imp_tv.fit_transform(train[numeric_cols]), columns=numeric_cols, index=train.index)
X_val_c = pd.DataFrame(imp_tv.transform(val[numeric_cols]), columns=numeric_cols, index=val.index)
X_te_c = pd.DataFrame(imp_full.transform(test[numeric_cols]), columns=numeric_cols, index=test.index)

clf_conf = mk_lgbm()
clf_conf.fit(X_tr_c.values, y_train.values)
cal_probs = clf_conf.predict_proba(X_val_c.values)[:, 1]
test_probs = clf_conf.predict_proba(X_te_c.values)[:, 1]

def split_conformal(cal_p, cal_y, test_p, alpha):
    n = len(cal_y)
    scores = np.where(cal_y == 1, 1.0 - cal_p, cal_p)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_hat = np.quantile(scores, min(q_level, 1.0))
    sets = []
    for p in test_p:
        s = set()
        if p <= q_hat:
            s.add(0)
        if (1 - p) <= q_hat:
            s.add(1)
        sets.append(s)
    return sets, q_hat

def mondrian_conformal(cal_p, cal_y, test_p, alpha):
    s0 = cal_p[cal_y == 0]
    s1 = 1.0 - cal_p[cal_y == 1]
    n0, n1 = len(s0), len(s1)
    q0 = np.quantile(s0, min(np.ceil((n0 + 1) * (1 - alpha)) / n0, 1.0))
    q1 = np.quantile(s1, min(np.ceil((n1 + 1) * (1 - alpha)) / n1, 1.0))
    sets = []
    for p in test_p:
        s = set()
        if p <= q0:
            s.add(0)
        if (1 - p) <= q1:
            s.add(1)
        sets.append(s)
    return sets, q0, q1

def eval_sets(pred_sets, y_true, label):
    n = len(y_true)
    coverage = np.mean([y_true.iloc[i] in s for i, s in enumerate(pred_sets)])
    certain_0 = sum(1 for s in pred_sets if s == {0})
    certain_1 = sum(1 for s in pred_sets if s == {1})
    uncertain = sum(1 for s in pred_sets if len(s) == 2)
    empty = sum(1 for s in pred_sets if len(s) == 0)
    alert_idx = [i for i, s in enumerate(pred_sets) if s == {1}]
    clear_idx = [i for i, s in enumerate(pred_sets) if s == {0}]
    alert_ppv = float(y_true.iloc[alert_idx].mean()) if alert_idx else float("nan")
    clear_npv = float((1 - y_true.iloc[clear_idx]).mean()) if clear_idx else float("nan")
    r = {
        "coverage": round(coverage, 4),
        "certain_survive": certain_0, "certain_death": certain_1,
        "uncertain": uncertain, "empty": empty,
        "alert_ppv": round(alert_ppv, 4) if not np.isnan(alert_ppv) else None,
        "clear_npv": round(clear_npv, 4) if not np.isnan(clear_npv) else None,
        "certain_frac": round((certain_0 + certain_1) / n, 4),
    }
    print(f"  {label}: cov={r['coverage']:.3f}  certain={certain_0+certain_1}/{n} ({r['certain_frac']:.1%})  "
          f"alert_PPV={alert_ppv:.3f}  clear_NPV={clear_npv:.3f}  uncertain={uncertain}")
    return r

for alpha in [0.05, 0.10, 0.15, 0.20]:
    sets_s, _ = split_conformal(cal_probs, y_val.values, test_probs, alpha)
    ALL[f"conformal_split_a{alpha}"] = eval_sets(sets_s, y_test, f"Split α={alpha}")

for alpha in [0.05, 0.10, 0.15]:
    sets_m, _, _ = mondrian_conformal(cal_probs, y_val.values, test_probs, alpha)
    ALL[f"conformal_mondrian_a{alpha}"] = eval_sets(sets_m, y_test, f"Mondrian α={alpha}")

print(f"  Conformal done in {time.time()-t0:.1f}s")


# ======================================================================
# PROBE 5: SURVIVAL ANALYSIS
# ======================================================================
print("\n" + "=" * 70)
print("PROBE 5: SURVIVAL ANALYSIS")
print("=" * 70)
t0 = time.time()
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored

    y_surv_train = np.array(
        [(bool(e), float(t)) for e, t in zip(y_train_full.values, time_train_full)],
        dtype=[("event", bool), ("time", float)],
    )
    y_surv_test = np.array(
        [(bool(e), float(t)) for e, t in zip(y_test.values, time_test)],
        dtype=[("event", bool), ("time", float)],
    )

    cox = CoxPHSurvivalAnalysis(alpha=0.1, n_iter=300)
    cox.fit(X_train_full.values, y_surv_train)
    risk_cox = cox.predict(X_test.values)
    ci_cox = concordance_index_censored(y_surv_test["event"], y_surv_test["time"], risk_cox)
    auroc_cox = safe_auroc(y_test, risk_cox)
    ALL["survival_cox"] = {"c_index": round(ci_cox[0], 4), "auroc": round(auroc_cox, 4)}
    print(f"  CoxPH: C-index={ci_cox[0]:.4f}  AUROC={auroc_cox:.4f}")

    rsf = RandomSurvivalForest(
        n_estimators=200, max_depth=6, min_samples_leaf=10,
        n_jobs=CPU, random_state=42,
    )
    rsf.fit(X_train_full.values, y_surv_train)
    risk_rsf = rsf.predict(X_test.values)
    ci_rsf = concordance_index_censored(y_surv_test["event"], y_surv_test["time"], risk_rsf)
    auroc_rsf = safe_auroc(y_test, risk_rsf)
    ALL["survival_rsf"] = {"c_index": round(ci_rsf[0], 4), "auroc": round(auroc_rsf, 4)}
    print(f"  RSF: C-index={ci_rsf[0]:.4f}  AUROC={auroc_rsf:.4f}")

    surv_fn = rsf.predict_survival_function(X_test.values)
    for h in [24, 48, 72, 168]:
        mort_h = np.array([1.0 - fn(h) if h <= fn.x[-1] else 1.0 for fn in surv_fn])
        y_actual_h = ((y_test.values == 1) & (time_test <= h)).astype(int)
        auroc_h = safe_auroc(y_actual_h, mort_h)
        ALL[f"survival_rsf_{h}h"] = {"auroc": round(auroc_h, 4), "n_events": int(y_actual_h.sum())}
        print(f"  RSF @ {h}h: AUROC={auroc_h:.4f}  events={y_actual_h.sum()}")

except Exception as e:
    ALL["survival_error"] = str(e)
    print(f"  Survival analysis error: {e}")
print(f"  Survival done in {time.time()-t0:.1f}s")


# ======================================================================
# PROBE 6: OPTUNA HPO (compact — 50 trials)
# ======================================================================
print("\n" + "=" * 70)
print("PROBE 6: OPTUNA HPO (50 trials)")
print("=" * 70)
t0 = time.time()
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

X_tr_h = pd.DataFrame(imp_tv.fit_transform(train[numeric_cols]), columns=numeric_cols, index=train.index)
X_val_h = pd.DataFrame(imp_tv.transform(val[numeric_cols]), columns=numeric_cols, index=val.index)

def objective(trial):
    params = {
        "objective": "binary", "verbose": -1, "random_state": 42,
        "n_jobs": CPU,
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "class_weight": "balanced",
    }
    clf = LGBMClassifier(**params)
    clf.fit(X_tr_h.values, y_train.values)
    probs = clf.predict_proba(X_val_h.values)[:, 1]
    return safe_auroc(y_val, probs)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, n_jobs=1, show_progress_bar=False)

best = study.best_trial
print(f"  Best val AUROC: {best.value:.4f}")
print(f"  Best params: {best.params}")

best_params = {**best.params, "objective": "binary", "class_weight": "balanced",
               "n_jobs": CPU, "random_state": 42, "verbose": -1}
clf_opt = LGBMClassifier(**best_params)
clf_opt.fit(X_train_full.values, y_train_full.values)
p_opt = clf_opt.predict_proba(X_test.values)[:, 1]
ALL["hpo_lgbm"] = quick_eval(p_opt, y_test, "HPO LightGBM (test)")
ALL["hpo_lgbm"]["best_val_auroc"] = round(best.value, 4)
ALL["hpo_lgbm"]["best_params"] = best.params

from sklearn.calibration import CalibratedClassifierCV
cal_opt = CalibratedClassifierCV(LGBMClassifier(**best_params), method="isotonic", cv=3)
cal_opt.fit(X_train_full.values, y_train_full.values)
p_cal = cal_opt.predict_proba(X_test.values)[:, 1]
ALL["hpo_lgbm_calibrated"] = quick_eval(p_cal, y_test, "HPO LightGBM + isotonic cal")
print(f"  HPO done in {time.time()-t0:.1f}s")


# ======================================================================
# PROBE 7: HPO + FEATURE ENGINEERING COMBO
# ======================================================================
print("\n" + "=" * 70)
print("PROBE 7: HPO + ENGINEERED FEATURES")
print("=" * 70)
t0 = time.time()
clf_combo = LGBMClassifier(**best_params)
clf_combo.fit(X_tr_o.values, y_train_full.values)
p_combo = clf_combo.predict_proba(X_te_o.values)[:, 1]
ALL["combo_hpo_engineered"] = quick_eval(p_combo, y_test, "HPO + engineered features")

cal_combo = CalibratedClassifierCV(LGBMClassifier(**best_params), method="isotonic", cv=3)
cal_combo.fit(X_tr_o.values, y_train_full.values)
p_cal_combo = cal_combo.predict_proba(X_te_o.values)[:, 1]
ALL["combo_hpo_eng_cal"] = quick_eval(p_cal_combo, y_test, "HPO + eng + isotonic cal")
print(f"  Combo done in {time.time()-t0:.1f}s")


# ======================================================================
# PROBE 8: MULTI-SEED STABILITY
# ======================================================================
print("\n" + "=" * 70)
print("PROBE 8: MULTI-SEED STABILITY (5 seeds)")
print("=" * 70)
t0 = time.time()
seed_results = []
for seed in [42, 77, 123, 202, 404]:
    tr_s, te_s = stratified_group_split(
        frame, group_col=SUBJECT_ID_COL, label_col=TARGET_COL,
        test_size=0.20, random_state=seed,
    )
    imp_s = SimpleImputer(strategy="median")
    X_tr_s = imp_s.fit_transform(tr_s[numeric_cols])
    X_te_s = imp_s.transform(te_s[numeric_cols])
    y_tr_s = tr_s[TARGET_COL].astype(int).values
    y_te_s = te_s[TARGET_COL].astype(int).values

    clf_s = LGBMClassifier(**best_params)
    clf_s.fit(X_tr_s, y_tr_s)
    p_s = clf_s.predict_proba(X_te_s)[:, 1]
    auroc_s = safe_auroc(y_te_s, p_s)
    auprc_s = safe_auprc(y_te_s, p_s)
    ppv50_s = threshold_search(y_te_s, p_s, precision_target=0.50)
    seed_results.append({
        "seed": seed, "auroc": round(auroc_s, 4), "auprc": round(auprc_s, 4),
        "recall@ppv50": round(ppv50_s.get("recall", float("nan")), 4),
    })
    print(f"  Seed {seed}: AUROC={auroc_s:.4f}  AUPRC={auprc_s:.4f}  recall@PPV50={ppv50_s.get('recall', float('nan')):.4f}")

aurocs = [r["auroc"] for r in seed_results]
ALL["stability_seeds"] = {
    "results": seed_results,
    "mean_auroc": round(np.mean(aurocs), 4),
    "std_auroc": round(np.std(aurocs), 4),
    "min_auroc": round(np.min(aurocs), 4),
    "max_auroc": round(np.max(aurocs), 4),
}
print(f"  Mean AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
print(f"  Stability done in {time.time()-t0:.1f}s")


# ======================================================================
# FINAL SUMMARY
# ======================================================================
total_time = time.time() - t_global
print("\n" + "=" * 70)
print(f"ALL PROBES COMPLETE in {total_time:.1f}s ({total_time/60:.1f} min)")
print("=" * 70)

with open(OUT / "probe_all_results.json", "w") as f:
    json.dump(ALL, f, indent=2, default=str)
print(f"Saved to {OUT / 'probe_all_results.json'}")

print("\n--- EXECUTIVE SUMMARY ---")
keys_of_interest = [
    ("ceiling_honest", "Honest baseline"),
    ("ceiling_overfit", "Overfit ceiling"),
    ("ceiling_cheat_time", "Cheat (time_to_event)"),
    ("ceiling_noise_floor", "Noise floor"),
    ("feat_baseline", "Feat: baseline"),
    ("feat_missingness", "Feat: +missingness"),
    ("feat_ratios", "Feat: +ratios"),
    ("feat_all_engineered", "Feat: full engineered"),
    ("individual_lgbm", "Individual LightGBM"),
    ("individual_xgb", "Individual XGBoost"),
    ("ensemble_stack_3", "Stack (LGB+XGB+LR)"),
    ("ensemble_avg_3", "Avg (LGB+XGB+LR)"),
    ("ensemble_stack_engineered", "Stack + engineered"),
    ("hpo_lgbm", "HPO LightGBM"),
    ("hpo_lgbm_calibrated", "HPO LGB + calibration"),
    ("combo_hpo_engineered", "HPO + engineered"),
    ("combo_hpo_eng_cal", "HPO + eng + cal"),
]
print(f"{'Method':<30s} {'AUROC':>7s} {'AUPRC':>7s} {'R@P50':>7s} {'R@P60':>7s} {'R@P70':>7s}")
print("-" * 70)
for key, name in keys_of_interest:
    if key in ALL and "auroc" in ALL[key]:
        r = ALL[key]
        print(f"{name:<30s} {r['auroc']:>7.4f} {r.get('auprc', float('nan')):>7.4f} "
              f"{r.get('recall@ppv50', float('nan')):>7.4f} {r.get('recall@ppv60', float('nan')):>7.4f} "
              f"{r.get('recall@ppv70', float('nan')):>7.4f}")
