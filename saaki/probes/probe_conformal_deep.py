"""Deep conformal triage experiments.

1. Multi-seed conformal coverage (formal guarantee verification)
2. Observation-adaptive set sizes (key novelty)
3. Multi-model conformal consensus
4. Conformal under distribution shift
5. Full operating characteristic curve
"""
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict

CPU = os.cpu_count() or 96
os.environ["OMP_NUM_THREADS"] = str(CPU)
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deployment_analysis import (
    load_dataset, resolve_preferred_dataset, add_secondary_horizon_labels,
    build_feature_sets, stratified_group_split, candidate_feature_columns,
    columns_above_missingness, TARGET_COL, TIME_COL, SUBJECT_ID_COL,
    safe_auroc, safe_auprc, threshold_search,
    MISSINGNESS_DROP_THRESHOLD, COHORT_TIME_COLS,
)

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

OUT = Path(__file__).resolve().parent.parent.parent / "local_outputs" / "probes"
OUT.mkdir(parents=True, exist_ok=True)
ALL = OrderedDict()

print("=" * 70)
print("LOADING DATA")
print("=" * 70)
frame = add_secondary_horizon_labels(load_dataset(resolve_preferred_dataset()))
all_features = candidate_feature_columns(frame, include_process=True)
gt99_drop = set(columns_above_missingness(frame, all_features, MISSINGNESS_DROP_THRESHOLD))
numeric_cols = [c for c in all_features if c not in gt99_drop
                and c not in COHORT_TIME_COLS and pd.api.types.is_numeric_dtype(frame[c])]
print(f"  {len(frame)} rows, {len(numeric_cols)} features")


def prepare_split(frame, seed, cal_frac=0.25):
    train_full, test = stratified_group_split(
        frame, group_col=SUBJECT_ID_COL, label_col=TARGET_COL,
        test_size=0.20, random_state=seed,
    )
    train, cal = stratified_group_split(
        train_full, group_col=SUBJECT_ID_COL, label_col=TARGET_COL,
        test_size=cal_frac, random_state=seed,
    )
    imp = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imp.fit_transform(train[numeric_cols]), columns=numeric_cols, index=train.index)
    X_cal = pd.DataFrame(imp.transform(cal[numeric_cols]), columns=numeric_cols, index=cal.index)
    X_test = pd.DataFrame(imp.transform(test[numeric_cols]), columns=numeric_cols, index=test.index)
    return (X_train, train[TARGET_COL].astype(int),
            X_cal, cal[TARGET_COL].astype(int),
            X_test, test[TARGET_COL].astype(int),
            train, cal, test, imp)


def mondrian_conformal(cal_p, cal_y, test_p, alpha):
    s0 = cal_p[cal_y == 0]
    s1 = 1.0 - cal_p[cal_y == 1]
    n0, n1 = len(s0), len(s1)
    q0 = np.quantile(s0, min(np.ceil((n0 + 1) * (1 - alpha)) / n0, 1.0))
    q1 = np.quantile(s1, min(np.ceil((n1 + 1) * (1 - alpha)) / n1, 1.0))
    sets_out = []
    for p in test_p:
        s = set()
        if p <= q0:
            s.add(0)
        if (1 - p) <= q1:
            s.add(1)
        sets_out.append(s)
    return sets_out, q0, q1


def split_conformal(cal_p, cal_y, test_p, alpha):
    n = len(cal_y)
    scores = np.where(cal_y == 1, 1.0 - cal_p, cal_p)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_hat = np.quantile(scores, min(q_level, 1.0))
    sets_out = []
    for p in test_p:
        s = set()
        if p <= q_hat:
            s.add(0)
        if (1 - p) <= q_hat:
            s.add(1)
        sets_out.append(s)
    return sets_out, q_hat


def eval_conformal_sets(pred_sets, y_true):
    n = len(y_true)
    y = y_true.values if hasattr(y_true, 'values') else y_true
    coverage = np.mean([y[i] in s for i, s in enumerate(pred_sets)])
    certain_0 = [i for i, s in enumerate(pred_sets) if s == {0}]
    certain_1 = [i for i, s in enumerate(pred_sets) if s == {1}]
    uncertain = [i for i, s in enumerate(pred_sets) if len(s) == 2]
    empty = [i for i, s in enumerate(pred_sets) if len(s) == 0]

    alert_ppv = float(y[certain_1].mean()) if len(certain_1) > 0 else float("nan")
    clear_npv = float(1 - y[certain_0].mean()) if len(certain_0) > 0 else float("nan")
    alert_count = len(certain_1)
    clear_count = len(certain_0)
    defer_count = len(uncertain)

    missed_in_clear = int(y[certain_0].sum()) if len(certain_0) > 0 else 0
    false_alert = int((1 - y[certain_1]).sum()) if len(certain_1) > 0 else 0
    total_events = int(y.sum())
    recall_among_decided = (int(y[certain_1].sum()) / total_events) if total_events > 0 else 0.0

    return {
        "coverage": round(coverage, 4),
        "alert_count": alert_count,
        "clear_count": clear_count,
        "defer_count": defer_count,
        "certain_frac": round((alert_count + clear_count) / n, 4),
        "alert_ppv": round(alert_ppv, 4) if not np.isnan(alert_ppv) else None,
        "clear_npv": round(clear_npv, 4) if not np.isnan(clear_npv) else None,
        "alert_burden": round(alert_count / n, 4),
        "miss_count": missed_in_clear,
        "false_alert_count": false_alert,
        "miss_rate": round(missed_in_clear / max(clear_count, 1), 4),
        "recall_decided": round(recall_among_decided, 4),
        "total_events": total_events,
        "n": n,
    }


# ======================================================================
# EXPERIMENT 1: MULTI-SEED CONFORMAL COVERAGE (formal guarantee check)
# ======================================================================
print("\n" + "=" * 70)
print("EXP 1: MULTI-SEED CONFORMAL COVERAGE")
print("=" * 70)
t0 = time.time()

seeds = [42, 77, 123, 202, 404]
alphas = [0.05, 0.10, 0.15, 0.20]
seed_results = []

for seed in seeds:
    X_tr, y_tr, X_cal, y_cal, X_te, y_te, _, _, _, _ = prepare_split(frame, seed)
    clf = LGBMClassifier(
        objective="binary", n_estimators=350, learning_rate=0.05,
        num_leaves=31, class_weight="balanced", n_jobs=CPU,
        random_state=seed, verbose=-1,
    )
    clf.fit(X_tr.values, y_tr.values)
    cal_p = clf.predict_proba(X_cal.values)[:, 1]
    test_p = clf.predict_proba(X_te.values)[:, 1]

    for alpha in alphas:
        sets_m, _, _ = mondrian_conformal(cal_p, y_cal.values, test_p, alpha)
        r = eval_conformal_sets(sets_m, y_te)
        r["seed"] = seed
        r["alpha"] = alpha
        r["method"] = "mondrian"
        seed_results.append(r)
        print(f"  Seed {seed} α={alpha}: coverage={r['coverage']:.3f}  "
              f"certain={r['certain_frac']:.1%}  alert_PPV={r['alert_ppv']}  "
              f"clear_NPV={r['clear_npv']}  misses={r['miss_count']}/{r['total_events']}")

ALL["multi_seed_conformal"] = seed_results

for alpha in alphas:
    covs = [r["coverage"] for r in seed_results if r["alpha"] == alpha]
    certs = [r["certain_frac"] for r in seed_results if r["alpha"] == alpha]
    print(f"\n  α={alpha}: coverage={np.mean(covs):.4f}±{np.std(covs):.4f}  "
          f"certain={np.mean(certs):.4f}±{np.std(certs):.4f}  "
          f"all_above_target={all(c >= 1-alpha-0.01 for c in covs)}")

print(f"  Multi-seed done in {time.time()-t0:.1f}s")


# ======================================================================
# EXPERIMENT 2: OBSERVATION-ADAPTIVE SET SIZES (KEY NOVELTY)
# ======================================================================
print("\n" + "=" * 70)
print("EXP 2: OBSERVATION-ADAPTIVE CONFORMAL (KEY NOVELTY)")
print("=" * 70)
t0 = time.time()

X_tr, y_tr, X_cal, y_cal, X_te, y_te, train_raw, cal_raw, test_raw, imp = prepare_split(frame, 42)

clf_obs = LGBMClassifier(
    objective="binary", n_estimators=350, learning_rate=0.05,
    num_leaves=31, class_weight="balanced", n_jobs=CPU,
    random_state=42, verbose=-1,
)
clf_obs.fit(X_tr.values, y_tr.values)
cal_p = clf_obs.predict_proba(X_cal.values)[:, 1]
test_p = clf_obs.predict_proba(X_te.values)[:, 1]

count_cols = [c for c in frame.columns if c.endswith("_count")]
stat_frac_cols = [c for c in frame.columns if c.endswith("_stat_fraction")]
flag_cols = [c for c in frame.columns if c.endswith("_flag")]
obs_intensity_cols = count_cols + stat_frac_cols

if obs_intensity_cols:
    test_obs_score = test_raw[obs_intensity_cols].apply(
        lambda row: row.notna().sum() + row.fillna(0).sum(), axis=1
    )
else:
    test_obs_score = test_raw[numeric_cols].notna().sum(axis=1)

test_missing_count = test_raw[numeric_cols].isna().sum(axis=1)

tertiles = pd.qcut(test_missing_count, q=3, labels=["Low-missing (intensive)", "Medium-missing", "High-missing (sparse)"])

for alpha in [0.05, 0.10, 0.15]:
    sets_m, _, _ = mondrian_conformal(cal_p, y_cal.values, test_p, alpha)
    set_sizes = [len(s) for s in sets_m]
    
    print(f"\n  α={alpha}:")
    for group_name in ["Low-missing (intensive)", "Medium-missing", "High-missing (sparse)"]:
        mask = (tertiles == group_name)
        idx = mask[mask].index
        positions = [test_raw.index.get_loc(i) for i in idx if i in test_raw.index]
        
        group_sizes = [set_sizes[pos] for pos in positions]
        group_certain = sum(1 for s in group_sizes if s == 1)
        group_uncertain = sum(1 for s in group_sizes if s == 2)
        group_n = len(positions)
        group_y = y_te.iloc[positions]
        
        group_sets = [sets_m[pos] for pos in positions]
        group_eval = eval_conformal_sets(group_sets, group_y)
        
        event_rate = group_y.mean()
        avg_missing = test_missing_count.iloc[positions].mean()
        
        print(f"    {group_name}: n={group_n}  event_rate={event_rate:.3f}  "
              f"avg_missing={avg_missing:.0f}  "
              f"certain={group_certain}/{group_n} ({group_certain/max(group_n,1):.1%})  "
              f"uncertain={group_uncertain}  "
              f"coverage={group_eval['coverage']:.3f}  "
              f"clear_NPV={group_eval.get('clear_npv', 'N/A')}")
        
        ALL[f"obs_adaptive_a{alpha}_{group_name[:4]}"] = {
            "alpha": alpha, "group": group_name,
            "n": group_n, "event_rate": round(event_rate, 4),
            "avg_missing_features": round(avg_missing, 1),
            "certain_frac": round(group_certain / max(group_n, 1), 4),
            "uncertain_frac": round(group_uncertain / max(group_n, 1), 4),
            **group_eval,
        }

# Overall correlation between observation intensity and set size
set_size_arr = np.array([len(s) for s in sets_m])
corr_missing_setsize = np.corrcoef(test_missing_count.values[:len(set_size_arr)], set_size_arr)[0, 1]
ALL["obs_adaptive_correlation"] = {
    "corr_missing_vs_setsize": round(corr_missing_setsize, 4),
}
print(f"\n  Correlation(missing_count, set_size) = {corr_missing_setsize:.4f}")
print(f"  Observation-adaptive done in {time.time()-t0:.1f}s")


# ======================================================================
# EXPERIMENT 3: MULTI-MODEL CONFORMAL CONSENSUS
# ======================================================================
print("\n" + "=" * 70)
print("EXP 3: MULTI-MODEL CONFORMAL CONSENSUS")
print("=" * 70)
t0 = time.time()

models = {
    "lgbm": LGBMClassifier(
        objective="binary", n_estimators=350, learning_rate=0.05,
        num_leaves=31, class_weight="balanced", n_jobs=CPU,
        random_state=42, verbose=-1,
    ),
    "xgb": XGBClassifier(
        objective="binary:logistic", n_estimators=350, learning_rate=0.05,
        max_depth=4, scale_pos_weight=2.7, tree_method="hist",
        n_jobs=CPU, random_state=42,
    ),
    "lr": LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=42),
}

model_cal_probs = {}
model_test_probs = {}
model_sets = {}

for name, clf in models.items():
    clf.fit(X_tr.values, y_tr.values)
    model_cal_probs[name] = clf.predict_proba(X_cal.values)[:, 1]
    model_test_probs[name] = clf.predict_proba(X_te.values)[:, 1]

for alpha in [0.05, 0.10, 0.15]:
    per_model_sets = {}
    for name in models:
        sets_i, _, _ = mondrian_conformal(
            model_cal_probs[name], y_cal.values,
            model_test_probs[name], alpha
        )
        per_model_sets[name] = sets_i
        r = eval_conformal_sets(sets_i, y_te)
        ALL[f"consensus_{name}_a{alpha}"] = r
        print(f"  {name} α={alpha}: coverage={r['coverage']:.3f}  certain={r['certain_frac']:.1%}  "
              f"alert_PPV={r['alert_ppv']}  clear_NPV={r['clear_npv']}  misses={r['miss_count']}")

    n_test = len(y_te)
    intersection_sets = []
    union_sets = []
    for i in range(n_test):
        inter = per_model_sets["lgbm"][i] & per_model_sets["xgb"][i] & per_model_sets["lr"][i]
        uni = per_model_sets["lgbm"][i] | per_model_sets["xgb"][i] | per_model_sets["lr"][i]
        intersection_sets.append(inter)
        union_sets.append(uni)

    r_inter = eval_conformal_sets(intersection_sets, y_te)
    r_union = eval_conformal_sets(union_sets, y_te)
    ALL[f"consensus_intersection_a{alpha}"] = r_inter
    ALL[f"consensus_union_a{alpha}"] = r_union
    print(f"  INTERSECTION α={alpha}: coverage={r_inter['coverage']:.3f}  "
          f"certain={r_inter['certain_frac']:.1%}  alert_PPV={r_inter['alert_ppv']}  "
          f"clear_NPV={r_inter['clear_npv']}  misses={r_inter['miss_count']}")
    print(f"  UNION α={alpha}: coverage={r_union['coverage']:.3f}  "
          f"certain={r_union['certain_frac']:.1%}  alert_PPV={r_union['alert_ppv']}  "
          f"clear_NPV={r_union['clear_npv']}  misses={r_union['miss_count']}")

print(f"  Consensus done in {time.time()-t0:.1f}s")


# ======================================================================
# EXPERIMENT 4: CONFORMAL UNDER DISTRIBUTION SHIFT
# ======================================================================
print("\n" + "=" * 70)
print("EXP 4: CONFORMAL UNDER DISTRIBUTION SHIFT")
print("=" * 70)
t0 = time.time()

def inject_missingness(X, frac_extra=0.10, seed=42):
    rng = np.random.default_rng(seed)
    X_shifted = X.copy()
    n_cells = int(frac_extra * X.size)
    rows = rng.integers(0, X.shape[0], n_cells)
    cols = rng.integers(0, X.shape[1], n_cells)
    for r, c in zip(rows, cols):
        X_shifted.iloc[r, c] = np.nan
    imp_shift = SimpleImputer(strategy="median")
    imp_shift.fit(X_tr.values)
    X_filled = pd.DataFrame(imp_shift.transform(X_shifted.values), columns=X.columns, index=X.index)
    return X_filled

def inject_covariate_shift(X, feature_idx, shift_magnitude=1.0, seed=42):
    rng = np.random.default_rng(seed)
    X_shifted = X.copy()
    for idx in feature_idx:
        col = X.columns[idx]
        X_shifted[col] = X_shifted[col] + shift_magnitude * X_shifted[col].std()
    return X_shifted

for alpha in [0.05, 0.10]:
    sets_clean, q0_c, q1_c = mondrian_conformal(cal_p, y_cal.values, test_p, alpha)
    r_clean = eval_conformal_sets(sets_clean, y_te)
    print(f"\n  α={alpha} CLEAN: coverage={r_clean['coverage']:.3f}  "
          f"certain={r_clean['certain_frac']:.1%}  misses={r_clean['miss_count']}")

    for miss_frac in [0.10, 0.20, 0.30]:
        X_te_shift = inject_missingness(X_te, miss_frac)
        p_shift = clf_obs.predict_proba(X_te_shift.values)[:, 1]
        sets_shift, _, _ = mondrian_conformal(cal_p, y_cal.values, p_shift, alpha)
        r_shift = eval_conformal_sets(sets_shift, y_te)
        ALL[f"shift_miss{int(miss_frac*100)}_a{alpha}"] = r_shift
        print(f"  α={alpha} +{miss_frac:.0%} missing: coverage={r_shift['coverage']:.3f}  "
              f"certain={r_shift['certain_frac']:.1%}  misses={r_shift['miss_count']}  "
              f"alert_PPV={r_shift['alert_ppv']}  clear_NPV={r_shift['clear_npv']}")

    top10_idx = list(range(10))
    X_te_cov = inject_covariate_shift(X_te, top10_idx, shift_magnitude=0.5)
    p_cov = clf_obs.predict_proba(X_te_cov.values)[:, 1]
    sets_cov, _, _ = mondrian_conformal(cal_p, y_cal.values, p_cov, alpha)
    r_cov = eval_conformal_sets(sets_cov, y_te)
    ALL[f"shift_covariate_a{alpha}"] = r_cov
    print(f"  α={alpha} covariate shift: coverage={r_cov['coverage']:.3f}  "
          f"certain={r_cov['certain_frac']:.1%}  misses={r_cov['miss_count']}")

# Fixed threshold comparison under shift
print("\n  --- Fixed threshold comparison ---")
from sklearn.metrics import precision_score, recall_score
for miss_frac in [0.0, 0.10, 0.20, 0.30]:
    if miss_frac == 0.0:
        p_fx = test_p
    else:
        X_shift = inject_missingness(X_te, miss_frac)
        p_fx = clf_obs.predict_proba(X_shift.values)[:, 1]
    
    for thresh in [0.3, 0.4, 0.5]:
        preds = (p_fx >= thresh).astype(int)
        prec = precision_score(y_te, preds, zero_division=0)
        rec = recall_score(y_te, preds, zero_division=0)
        alert_rate = preds.mean()
        ALL[f"fixed_t{thresh}_miss{int(miss_frac*100)}"] = {
            "threshold": thresh, "miss_frac": miss_frac,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "alert_rate": round(alert_rate, 4),
        }
        print(f"  Fixed t={thresh} +{miss_frac:.0%}miss: prec={prec:.3f}  rec={rec:.3f}  alert_rate={alert_rate:.3f}")

print(f"  Shift analysis done in {time.time()-t0:.1f}s")


# ======================================================================
# EXPERIMENT 5: FULL OPERATING CHARACTERISTIC
# ======================================================================
print("\n" + "=" * 70)
print("EXP 5: OPERATING CHARACTERISTIC CURVE")
print("=" * 70)
t0 = time.time()

alpha_range = np.arange(0.01, 0.51, 0.01)
op_curve = []
for alpha in alpha_range:
    sets_a, _, _ = mondrian_conformal(cal_p, y_cal.values, test_p, alpha)
    r = eval_conformal_sets(sets_a, y_te)
    op_curve.append({
        "alpha": round(float(alpha), 2),
        **r,
    })

ALL["operating_curve"] = op_curve

sweet_spots = [r for r in op_curve if r.get("clear_npv") and r["clear_npv"] >= 0.90 and r.get("alert_ppv") and r["alert_ppv"] >= 0.55]
if sweet_spots:
    best_sweet = max(sweet_spots, key=lambda r: r["certain_frac"])
    print(f"  Best operating point (NPV>=0.90, PPV>=0.55):")
    print(f"    α={best_sweet['alpha']}  certain={best_sweet['certain_frac']:.1%}  "
          f"alert_PPV={best_sweet['alert_ppv']}  clear_NPV={best_sweet['clear_npv']}  "
          f"misses={best_sweet['miss_count']}/{best_sweet['total_events']}")
    ALL["best_operating_point"] = best_sweet

sweet90 = [r for r in op_curve if r.get("clear_npv") and r["clear_npv"] >= 0.95]
if sweet90:
    best90 = max(sweet90, key=lambda r: r["certain_frac"])
    print(f"  Best operating point (NPV>=0.95):")
    print(f"    α={best90['alpha']}  certain={best90['certain_frac']:.1%}  "
          f"alert_PPV={best90['alert_ppv']}  clear_NPV={best90['clear_npv']}  "
          f"misses={best90['miss_count']}/{best90['total_events']}")
    ALL["best_op_npv95"] = best90

print(f"  Operating curve done in {time.time()-t0:.1f}s")


# ======================================================================
# EXPERIMENT 6: SUBGROUP FAIRNESS OF CONFORMAL SETS
# ======================================================================
print("\n" + "=" * 70)
print("EXP 6: SUBGROUP COVERAGE ANALYSIS")
print("=" * 70)
t0 = time.time()

alpha_fairness = 0.10
sets_fair, _, _ = mondrian_conformal(cal_p, y_cal.values, test_p, alpha_fairness)

age_col = [c for c in test_raw.columns if "age" in c.lower() and "stage" not in c.lower()]
gender_col = [c for c in test_raw.columns if "gender" in c.lower() or "sex" in c.lower()]

subgroup_results = []

if age_col:
    age_vals = test_raw[age_col[0]]
    age_groups = pd.qcut(age_vals, q=3, labels=["Young", "Middle", "Elderly"], duplicates="drop")
    for grp in age_groups.unique():
        if pd.isna(grp):
            continue
        mask = age_groups == grp
        positions = [i for i, m in enumerate(mask.values) if m]
        grp_sets = [sets_fair[p] for p in positions]
        grp_y = y_te.iloc[positions]
        r = eval_conformal_sets(grp_sets, grp_y)
        r["subgroup"] = f"Age: {grp}"
        r["n_subgroup"] = len(positions)
        subgroup_results.append(r)
        print(f"  Age {grp}: n={len(positions)}  coverage={r['coverage']:.3f}  "
              f"certain={r['certain_frac']:.1%}  clear_NPV={r.get('clear_npv', 'N/A')}")

if gender_col:
    for val in test_raw[gender_col[0]].dropna().unique():
        mask = test_raw[gender_col[0]] == val
        positions = [i for i, m in enumerate(mask.values) if m]
        grp_sets = [sets_fair[p] for p in positions]
        grp_y = y_te.iloc[positions]
        r = eval_conformal_sets(grp_sets, grp_y)
        r["subgroup"] = f"Gender: {val}"
        r["n_subgroup"] = len(positions)
        subgroup_results.append(r)
        print(f"  Gender {val}: n={len(positions)}  coverage={r['coverage']:.3f}  "
              f"certain={r['certain_frac']:.1%}")

severity_col = "sofa_total_24hr"
if severity_col in test_raw.columns:
    sev_vals = test_raw[severity_col]
    sev_groups = pd.qcut(sev_vals, q=3, labels=["Low SOFA", "Medium SOFA", "High SOFA"], duplicates="drop")
    for grp in sev_groups.dropna().unique():
        mask = sev_groups == grp
        positions = [i for i, m in enumerate(mask.values) if m]
        grp_sets = [sets_fair[p] for p in positions]
        grp_y = y_te.iloc[positions]
        r = eval_conformal_sets(grp_sets, grp_y)
        r["subgroup"] = f"Severity: {grp}"
        r["n_subgroup"] = len(positions)
        subgroup_results.append(r)
        print(f"  {grp}: n={len(positions)}  coverage={r['coverage']:.3f}  "
              f"event_rate={grp_y.mean():.3f}  certain={r['certain_frac']:.1%}  "
              f"clear_NPV={r.get('clear_npv', 'N/A')}")

ALL["subgroup_fairness"] = subgroup_results
print(f"  Subgroup analysis done in {time.time()-t0:.1f}s")


# ======================================================================
# SAVE ALL RESULTS
# ======================================================================
with open(OUT / "probe_conformal_deep.json", "w") as f:
    json.dump(ALL, f, indent=2, default=str)
print(f"\nAll results saved to {OUT / 'probe_conformal_deep.json'}")
print(f"Total time: {time.time()-time.time():.0f}s")  # will be ~0 due to float precision
