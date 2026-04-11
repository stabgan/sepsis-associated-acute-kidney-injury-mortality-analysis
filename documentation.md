# Data Dictionary — SA-AKI Cohort (Final Build, n=10,036, p=162)

**Scope & conventions.** Variables list the **meaning**, **target unit**, and (where relevant) **how values are
aggregated in the first 24 h**. Because many features share statistical suffixes, see §1 first. Laboratory features in
§7 are **fully standardized** with unit conversions, edge-case rules, and compact, high-signal aggregates. Unless noted
otherwise, counts are based on **non-missing** observations.

---

## 0 · Cohort snapshot (from current file)

**Key metrics**

| Metric                   | Value              |
|--------------------------|--------------------|
| In-hospital death        | 26.7% (2684/10036) |
| Male                     | 53.0% (5322/10036) |
| AKI trigger = Creatinine | 40.7% (4087/10036) |

**Therapies & exposures in first 24 h**

| Flag                             | Prevalence         |
|----------------------------------|--------------------|
| mechanical_ventilation_24hr_flag | 53.0% (5320/10036) |
| vasopressor_24hr_flag            | 36.6% (3676/10036) |
| rrt_24hr_flag                    | 69.0% (6922/10036) |
| baseline_scr_estimated_flag      | 0.1% (7/10036)     |

**Charlson comorbidities**

| Flag                                 | Prevalence         |
|--------------------------------------|--------------------|
| charlson_cancer_flag                 | 8.9% (894/10036)   |
| charlson_chf_flag                    | 11.8% (1187/10036) |
| charlson_copd_flag                   | 14.3% (1440/10036) |
| charlson_cvd_flag                    | 9.7% (975/10036)   |
| charlson_dementia_flag               | 4.1% (412/10036)   |
| charlson_diabetes_complications_flag | 9.3% (935/10036)   |
| charlson_diabetes_flag               | 18.6% (1866/10036) |
| charlson_hemiplegia_flag             | 3.5% (349/10036)   |
| charlson_hiv_flag                    | 0.4% (36/10036)    |
| charlson_liver_mild_flag             | 10.4% (1043/10036) |
| charlson_liver_severe_flag           | 6.5% (656/10036)   |
| charlson_metastatic_cancer_flag      | 3.9% (392/10036)   |
| charlson_mi_flag                     | 12.8% (1282/10036) |
| charlson_pud_flag                    | 2.6% (256/10036)   |
| charlson_pvd_flag                    | 7.9% (789/10036)   |
| charlson_renal_flag                  | 14.2% (1421/10036) |
| charlson_rheumatic_flag              | 2.1% (214/10036)   |

---

## 1 · Suffix legend (applies to time-series features)

| Suffix    | Definition (first 24 h)               | Unit comment |
|-----------|---------------------------------------|--------------|
| `_first`  | First measurement in the window       | Base unit    |
| `_last`   | Last measurement in the window        | Base unit    |
| `_median` | Median of all readings over 24 h      | Base unit    |
| `_iqr`    | Inter-quartile range (Q3−Q1)          | Base unit    |
| `_rng`    | Range = max − min                     | Base unit    |
| `_delta`  | Change = last − first                 | Base unit    |
| `_auc`    | Trapezoidal area-under-curve (0–24 h) | Base × h     |
| `_slope`  | Linear regression slope per hour      | Base / h     |
| `_n`      | Number of discrete observations       | Count        |

> For deltas/slopes we require `_n ≥ 2`; otherwise the feature is `NULL`. Not all suffixes appear for every base.

---

## 2 · Survival anchors & identifiers

**Core variables**

| Column                                | Meaning                                               | Unit |
|---------------------------------------|-------------------------------------------------------|------|
| `death`                               | In-hospital death indicator                           | —    |
| `time_to_death`                       | Hours from ICU admit to death / last alive (censored) | h    |
| `time_to_aki_onset`                   | Hours to AKI onset (KDIGO)                            | h    |
| `time_to_sepsis_onset`                | Hours to first Sepsis-3 flag                          | h    |
| `time_to_mechanical_ventilator_start` | Hours to first invasive ventilation                   | h    |
| `time_to_vasopressor_start`           | Hours to first vasopressor                            | h    |
| `age`                                 | Age at ICU admission                                  | y    |
| `gender`                              | Biological sex                                        | —    |
| `admission_weight`                    | Documented admission weight                           | kg   |
| `aki_trigger`                         | AKI trigger type                                      | —    |

**Encodings**

- `death`: **1=Death**, 0=Censored
- `gender`: **1=Male**, 0=Female
- `aki_trigger`: **1=AKI triggered by Creatinine**, 0= AKI triggered by URINE output criteria

---

## 3 · Therapies & exposure flags

All encoded **1=Yes, 0=No**.

- `mechanical_ventilation_24hr_flag`
- `vasopressor_24hr_flag`
- `rrt_24hr_flag`
- `baseline_scr_estimated_flag`

**Charlson comorbidity flags:**  
`charlson_cancer_flag`, `charlson_chf_flag`, `charlson_copd_flag`, `charlson_cvd_flag`, `charlson_dementia_flag`,
`charlson_diabetes_complications_flag`, `charlson_diabetes_flag`, `charlson_hemiplegia_flag`, `charlson_hiv_flag`,
`charlson_liver_mild_flag`, `charlson_liver_severe_flag`, `charlson_metastatic_cancer_flag`, `charlson_mi_flag`,
`charlson_pud_flag`, `charlson_pvd_flag`, `charlson_renal_flag`, `charlson_rheumatic_flag`.

---

## 4 · Severity scores

| Column                     | Meaning                                            | Unit   |
|----------------------------|----------------------------------------------------|--------|
| `apache_iii_score`         | APACHE III composite severity                      | points |
| `sofa_total_24hr`          | Total SOFA (resp, coag, liver, cardio, CNS, renal) | points |
| `sofa_respiratory_24hr`    | SOFA respiratory                                   | points |
| `sofa_coagulation_24hr`    | SOFA coagulation                                   | points |
| `sofa_hepatic_24hr`        | SOFA liver                                         | points |
| `sofa_cardiovascular_24hr` | SOFA cardiovascular                                | points |
| `sofa_neurological_24hr`   | SOFA CNS                                           | points |
| `sofa_renal_24hr`          | SOFA renal                                         | points |

---

## 5 · Fluid balance & urine output

| Column                          | Meaning                       | Unit  |
|---------------------------------|-------------------------------|-------|
| `fluid_balance_24hr_ml`         | Net inputs − outputs (0–24 h) | mL    |
| `fluid_balance_24hr_ml_perkg`   | Net balance per kg            | mL/kg |
| `urine_output_total_24hr`       | Total urine (0–24 h)          | mL    |
| `urine_output_total_24hr_perkg` | Total urine per kg            | mL/kg |

---

## 6 · Vital signs (first 24 h)

**Canonical statistic:** median over first 24 h (additional suffix features may exist per §1).

| Base        | Target unit | Canonical column   |
|-------------|-------------|--------------------|
| `heartrate` | beats/min   | `heartrate_median` |
| `sysbp`     | mmHg        | `sysbp_median`     |
| `diabp`     | mmHg        | `diabp_median`     |
| `meanbp`    | mmHg        | `meanbp_median`    |
| `resprate`  | breaths/min | `resprate_median`  |
| `spo2`      | %           | `spo2_median`      |

---

## 7 · Laboratory features (first 24 h) — curated & unit-harmonized

**Approach.** Values are normalized to target units **before** aggregation. We also report **QC fractions** where
available (reference-interval violations, lab “abnormal” flags, and STAT order proportions).

### 7.1 · Electrolytes & acid–base

| Feature(s)                                                                                                                     | Target unit | 0–24 h summary           |
|--------------------------------------------------------------------------------------------------------------------------------|-------------|--------------------------|
| `sodium_min`, `sodium_max`, `sodium_n` (serum sodium)                                                                          | mmol/L      | min, max, count          |
| `potassium_min`, `potassium_max`, `potassium_n` (serum potassium)                                                              | mmol/L      | min, max, count          |
| `chloride_median`, `chloride_n` (serum chloride)                                                                               | mmol/L      | median, count            |
| `bicarbonate_min`, `bicarbonate_n`, `bicarbonate_outofref_frac`, `bicarbonate_stat_frac`, `bicarbonate_abn_frac` (serum HCO₃⁻) | mmol/L      | min, count, QC fractions |
| `aniongap_max`, `aniongap_n`, `aniongap_outofref_frac`, `aniongap_stat_frac`, `aniongap_abn_frac` (serum anion gap)            | mmol/L      | max, count, QC fractions |
| `ph_min`, `ph_last`, `ph_n` (arterial pH)                                                                                      | pH units    | min, last, count         |
| `paco2_max`, `paco2_median`, `paco2_n` (arterial CO₂)                                                                          | mmHg        | max, median, count       |

> Sodium/potassium include both hypo- and hyper- extremes via min/max. HCO₃⁻ and anion gap reflect metabolic acidosis;
> QC fractions quantify result flags and priority mix.

### 7.2 · Renal function & glycemia

| Feature(s)                                                                                                                                         | Target unit | 0–24 h summary                         |
|----------------------------------------------------------------------------------------------------------------------------------------------------|-------------|----------------------------------------|
| `creatinine_max`, `creatinine_delta`, `creatinine_n`, `creatinine_outofref_frac`, `creatinine_stat_frac`, `creatinine_abn_frac` (serum creatinine) | mg/dL       | max, (last−first), count, QC fractions |
| `bun_median`, `bun_n` (blood urea nitrogen)                                                                                                        | mg/dL       | median, count                          |
| `glucose_min`, `glucose_max`, `glucose_n`, `glucose_stat_frac`, `glucose_abn_frac` (plasma glucose)                                                | mg/dL       | min, max, count, QC fractions          |

> `creatinine_delta` is `NULL` if `_n < 2`. Glucose uses both tails to capture hypo-/hyperglycemia risk.

### 7.3 · Calcium–magnesium–phosphate

| Feature(s)                                                                     | Target unit | 0–24 h summary     |
|--------------------------------------------------------------------------------|-------------|--------------------|
| `calcium_min`, `calcium_median`, `calcium_n` (total calcium)                   | mg/dL       | min, median, count |
| `magnesium_min_mgdl`, `magnesium_median_mgdl`, `magnesium_n` (serum magnesium) | mg/dL       | min, median, count |
| `phosphate_max_mgdl`, `phosphate_median_mgdl`, `phosphate_n` (serum phosphate) | mg/dL       | max, median, count |

> Magnesium and phosphate are stored in **mg/dL** (converted from mmol/L where applicable).

### 7.4 · Hepatic & biliary panel

| Feature(s)                                                                                | Target unit | 0–24 h summary               |
|-------------------------------------------------------------------------------------------|-------------|------------------------------|
| `alkphos_median`, `alkphos_n` (alkaline phosphatase)                                      | U/L         | median, count                |
| `ast_max`, `ast_median`, `ast_n` (AST)                                                    | U/L         | max, median, count           |
| `alt_median`, `alt_n` (ALT)                                                               | U/L         | median, count                |
| `tbili_max_mgdl`, `tbili_median_mgdl`, `tbili_outofref_frac`, `tbili_n` (total bilirubin) | mg/dL       | max, median, QC fractions, n |
| `dbili_max_mgdl`, `dbili_median_mgdl`, `dbili_n` (direct bilirubin)                       | mg/dL       | max, median, count           |
| `ggt_median_ul`, `ggt_n` (GGT)                                                            | U/L         | median, count                |
| `albumin_median`, `albumin_min`, `albumin_n` (serum albumin)                              | g/dL        | median, min, count           |

### 7.5 · Hematology & coagulation

| Feature(s)                                                                                                                                                     | Target unit | 0–24 h summary                   |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|----------------------------------|
| `hemoglobin_min`, `hemoglobin_median`, `hemoglobin_n` (hemoglobin)                                                                                             | g/dL        | min, median, count               |
| `hematocrit_median`, `hematocrit_n` (hematocrit)                                                                                                               | %           | median, count                    |
| `plateletcount_min`, `plateletcount_median`, `plateletcount_outofref_frac`, `plateletcount_stat_frac`, `plateletcount_abn_frac`, `plateletcount_n` (platelets) | 10⁹/L       | min, median, QC fractions, count |
| `aptt_max_s`, `aptt_median_s`, `aptt_outofref_frac`, `aptt_stat_frac`, `aptt_n` (aPTT)                                                                         | s           | max, median, QC fractions, count |
| `inr_max`, `inr_median`, `inr_outofref_frac`, `inr_stat_frac`, `inr_abn_frac`, `inr_n` (INR)                                                                   | ratio       | max, median, QC fractions, count |
| `wbc_max_10e9l`, `wbc_median_10e9l`, `wbc_abn_frac`, `wbc_n` (WBC)                                                                                             | 10⁹/L       | max, median, QC fraction, count  |
| `neutrophils_pct_min`, `neutrophils_pct_median`, `neutrophils_pct_n` (neutrophils %)                                                                           | %           | min, median, count               |
| `anc_median_10e9l`, `anc_n` (absolute neutrophil count)                                                                                                        | 10⁹/L       | median, count                    |
| `rdw_median_pct`, `rdw_n` (RDW)                                                                                                                                | %           | median, count                    |

### 7.6 · Lactate, cardiac & tissue injury, inflammation

| Feature(s)                                                                                                           | Target unit | 0–24 h summary                     |
|----------------------------------------------------------------------------------------------------------------------|-------------|------------------------------------|
| `lactate_max`, `lactate_last`, `lactate_delta`, `lactate_stat_frac`, `lactate_abn_frac`, `lactate_n` (serum lactate) | mmol/L      | max, last, (last−first), QC, count |
| `troponin_max_ngml`, `troponin_stat_frac`, `troponin_n` (cardiac troponin)                                           | ng/mL       | max, QC, count                     |
| `bnp_ntprobnp_max_pgml`, `bnp_ntprobnp_n` (BNP / NT-proBNP)                                                          | pg/mL       | max, count                         |
| `ldh_max_ul`, `ldh_median_ul`, `ldh_n` (LDH)                                                                         | U/L         | max, median, count                 |
| `crp_max_mgl`, `crp_median_mgl`, `crp_stat_frac`, `crp_abn_frac`, `crp_n` (CRP)                                      | mg/L        | max, median, QC, count             |

> `lactate_delta` computed only if `_n ≥ 2`; otherwise `NULL`.

### 7.7 · Oxygenation & gas exchange (incl. derived)

| Feature(s)                                        | Target unit | 0–24 h summary     |
|---------------------------------------------------|-------------|--------------------|
| `pao2_min`, `pao2_median`, `pao2_n` (arterial O₂) | mmHg        | min, median, count |
| `pf_min`, `pf_median`, `pf_n` (PaO₂/FiO₂, paired) | mmHg        | min, median, count |

> **PF pairing.** Each PaO₂ is paired to the nearest FiO₂ (Chartevents) within a tight window; implausible/missing FiO₂
> is discarded and PF is capped at a clinically plausible ceiling.

### 7.8 · QC fractions & metadata (where present)

- `*_outofref_frac`: share outside reference interval (after unit harmonization)
- `*_abn_frac`: share labeled “abnormal” by lab system
- `*_stat_frac`: share of STAT (urgent) orders
- `*_n`: total usable measurements in 0–24 h  
  All fractions lie in **[0, 1]** and use the analyte’s `_n` as denominator.

### 7.9 · Target units & conversions (applied row-wise)

- **Glucose:** mmol/L → mg/dL × **18**
- **Albumin, Hemoglobin:** g/L → g/dL ÷ **10**
- **Calcium (total):** mmol/L → mg/dL × **4.0**
- **Chloride:** mg/dL → mmol/L ÷ **3.545**
- **Lactate:** mg/dL → mmol/L ÷ **9**
- **Blood gases (PaO₂, PaCO₂):** kPa → mmHg × **7.5**
- **Bilirubin (total/direct):** µmol/L → mg/dL ÷ **17.104**
- **Magnesium:** mmol/L → mg/dL × **2.433**
- **Phosphate:** mmol/L → mg/dL × **3.097**
- **Troponin:** ng/L → ng/mL ÷ **1000** (µg/L = ng/mL)
- **BNP/NT-proBNP:** ng/L ≡ pg/mL (numerically equal)

> Conversions occur **per row using `valueuom`** before any QC or statistics; non-numeric/missing values are ignored for
> that analyte.

### 7.10 · Clinical-plausible guardrails (post-normalization)

Used for plausibility checks and to compute out-of-range fractions (not strict winsorization):

- **Electrolytes/acid–base:** Na 100–190, K 2–8, Cl 70–130, HCO₃⁻ 5–40, Anion gap 3–30, pH 6.5–7.7, PaCO₂ 10–100, PaO₂
  30–500
- **Renal/glycemia:** Cr 0.3–15 mg/dL, BUN 5–200 mg/dL, Glucose 20–800 mg/dL
- **Ca–Mg–P:** Ca 4–15 mg/dL, Mg 0.5–6 mg/dL, Phos 0.5–15 mg/dL
- **Hepatic:** AST/ALT 5–10,000 U/L, ALP 20–1,200 U/L, Total bili 0–40 mg/dL, Direct bili 0–30 mg/dL, GGT 0–2,000 U/L,
  Albumin 1–6 g/dL
- **Heme/coag:** Hb 3–20 g/dL, Plt 5–1000×10⁹/L, aPTT 15–200 s, INR 0.8–10, WBC 0–200×10⁹/L (max up to 400)
- **Inflammation/injury:** CRP 0–500 mg/L, LDH 50–10,000 U/L, Troponin 0–100 ng/mL, BNP/NT-proBNP 0–100,000 pg/mL
- **Derived:** PF ratio 0–600 mmHg

---

## 8 · Encodings & quick prevalences (summary)

- **Binary encodings:** `death`, `gender` (1=Male), `aki_trigger`, all `*_flag`, all `charlson_*_flag`
- **Cohort exposures (first 24 h):** indicators for invasive ventilation, vasoactive use, RRT, AKI trigger attribution,
  and sex distribution are summarized in §0.

---

### Abbreviations

AKI = Acute Kidney Injury; KDIGO = Kidney Disease: Improving Global Outcomes; SOFA = Sequential Organ Failure
Assessment; RRT = Renal Replacement Therapy; FiO₂ = inspired oxygen fraction; MAP = Mean Arterial Pressure; aPTT =
activated partial thromboplastin time; LDH = lactate dehydrogenase; CRP = C-reactive protein; BNP = B-type natriuretic
peptide; NT-proBNP = N-terminal pro-BNP; ANC = absolute neutrophil count.

---

### Implementation notes (for reproducibility)

- Lab features restricted to **0–24 h from ICU admission**
- For deltas/slopes, require `_n ≥ 2`; otherwise set to `NULL` while retaining the companion `_n`
- QC fractions rely on lab **flag**, **priority** (STAT vs routine), and **reference intervals** after unit
  normalization
- **PF ratio** uses **temporally paired** PaO₂ (ABG) and FiO₂ (Chartevents) with quality filters
- All unit conversions (§7.9) **precede** QC and aggregation
