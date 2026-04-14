# Style Audit Report — arXiv LaTeX Source Files

**Files audited:**
1. `main.tex` (618 lines)
2. `section_results.tex` (176 lines)
3. `section_discussion_to_appendices.tex` (271 lines)

---

## 1. Em Dashes (`---`) — AI Writing Markers

Em dashes are a strong AI-writing signal. All instances in body text (LaTeX comment-line separators excluded):

### main.tex

| Line | Content (excerpt) |
|------|-------------------|
| 88 | `decisions---\emph{Alert}~$\{1\}$, \emph{Clear}~$\{0\}$, or` |
| 89 | `\emph{Defer}~$\{0,1\}$---with finite-sample coverage guarantees.` |
| 133 | `escalation of care---whether that involves intensifying haemodynamic support,` |
| 141 | `that matters at the bedside---\emph{when should the model's output actually` |
| 152 | `found that the strongest conventional models---logistic regression, XGBoost,` |
| 153 | `LightGBM, and CatBoost---occupy a narrow performance band` |
| 171 | `interpretable forms: $\{1\}$ (\emph{Alert}---the model is confident the patient` |
| 172 | `is high-risk), $\{0\}$ (\emph{Clear}---the model is confident the patient is` |
| 173 | `lower-risk), or $\{0,1\}$ (\emph{Defer}---the model's uncertainty is too large` |
| 185 | `positive predictive value (PPV), and clear negative predictive value (NPV)---with` |
| 191 | `defines when a model should---and should not---be trusted.` |
| 209 | `tracks---conventional baselines, disagreement-based selective triage` |
| 210 | `(ablation), and Mondrian conformal selective triage (primary)---feed into` |
| 231 | `Specifically, systemic antibiotic and vasopressor flags---required for Sepsis-3` |
| 232 | `labelling and feature construction, respectively---were derived by mapping` |
| 332 | `information after T24---including total ICU length of stay and whole-stay` |
| 333 | `aggregates---was included in the predictor set.` |
| 461 | `$C(x) = \{1\}$: \textbf{Alert}---the model is confident the patient is` |
| 463 | `$C(x) = \{0\}$: \textbf{Clear}---the model is confident the patient is` |
| 465 | `$C(x) = \{0,1\}$: \textbf{Defer}---the model's uncertainty is too large` |

### section_results.tex

| Line | Content (excerpt) |
|------|-------------------|
| 43 | `the honest grouped setting---which excludes any post-T24 temporal information---remains at 0.763.` |
| 47 | `AUROC 0.579---both well below the ML baselines.` |
| 55 | `At a decision threshold of 0.20---a clinically plausible threshold for initiating heightened monitoring` |
| 80 | `The miss count of 23.1 patients per seed---cases where the true label falls outside the prediction set---is the price paid` |
| 83 | `This is not high enough for autonomous escalation decisions---a clinician would still need to review each alert---but it is high enough` |
| 85 | `Under the manuscript sweet-spot criterion---requiring both NPV $\geq 0.90$ and PPV $\geq 0.55$---the best operating point` |
| 85 | (second instance) `at $\alpha=0.20$, the certain-decision fraction reaches 0.782 but alert PPV falls to 0.509---barely above the prior probability` |
| 89 | `intersection (a patient is classified only if all three models---XGBoost, LightGBM, and logistic regression---agree` |
| 108 | `alert PPV of 0.729 is the highest among all operating points reported in this paper---when the union consensus flags a patient` |
| 140 | `the nominal 0.95 target reflects the conservatism of the conformal calibration under shift---the nonconformity thresholds` |
| 142 | `recall drops to 0.171---a 42\% relative decline---while precision actually increases slightly` |
| 145 | `alert PPV of 0.646---comparable to the clean baseline.` |
| 172 | `coverage of only 0.866---below the nominal 0.90 target---with alert PPV 0.659` |

### section_discussion_to_appendices.tex

| Line | Content (excerpt) |
|------|-------------------|
| 18 | `uncertainty-aware action restriction improves reliability---actionable error rate dropped from 0.343 under fixed thresholds to 0.254 under selective triage---it provides no finite-sample guarantee` |
| 22 | `Under simulated distribution shift---random missingness injection at 10--30\% severity, measurement-process dropout, and care-process dropout---conformal coverage remained stable` |
| 38 | `The finite-sample coverage guarantee provided by conformal prediction holds marginally under the exchangeability assumption---that is, averaged over the entire test population.` |
| 46 | `Features from later time points---treatment response trajectories, serial organ failure trends, and evolving care-process markers---could substantially improve discrimination` |
| 52 | `The conventional benchmark story---XGBoost with isotonic calibration at AUROC 0.768, outperforming APACHE-III (0.579) and SOFA (0.646) on decision-curve net benefit---establishes that machine learning adds value` |

**Total em dash instances: ~38 in body text across all three files.**

**Recommendation:** Replace every em dash with proper subordinate clauses, parenthetical constructions, semicolons, or sentence restructuring. Em dashes are the single strongest AI-writing fingerprint in this manuscript.

---

## 2. "We" Usage — Sentence Starters and Frequent Active Voice

### main.tex

| Line | Content (excerpt) |
|------|-------------------|
| 80 | `We analysed 10,036 ICU stays (9,002 unique subjects; 26.7\% mortality) from the` (Abstract — sentence starter) |
| 151 | `We analysed 10,036 ICU stays from MIMIC-IV~v3.1 and` (Introduction — sentence starter) |
| 163 | `We therefore compare two uncertainty-aware selective-triage strategies.` (Introduction — sentence starter) |
| 474 | `...we constructed a union consensus across multiple base models.` (Methods) |
| 561 | `For conformal selective triage, we reported: empirical coverage...` (Methods) |

### section_discussion_to_appendices.tex

| Line | Content (excerpt) |
|------|-------------------|
| 12 | `...and we argue that this gap reflects methodological discipline rather than model inferiority.` |

### section_results.tex

No "We" sentence starters found. This file uses passive voice consistently.

**Total "We" instances: 6 (4 as sentence starters, 2 mid-sentence)**

**Recommendation:** Convert sentence-starting "We" to passive voice. E.g., "We analysed 10,036 ICU stays" → "A total of 10,036 ICU stays were analysed" or "The analysis comprised 10,036 ICU stays." Mid-sentence "we" in Methods is more acceptable but should still be reviewed for consistency.

---

## 3. Filler Phrases

### section_results.tex

| Line | Phrase | Content (excerpt) |
|------|--------|-------------------|
| 174 | `noteworthy` | `The ventilation result is noteworthy: mechanical ventilation status appears to sharpen the model's alert channel` |

### main.tex — None found
### section_discussion_to_appendices.tex — None found

**Total filler phrase instances: 1**

**Recommendation:** Remove "is noteworthy" and restructure: "Mechanical ventilation status appears to sharpen the model's alert channel, possibly because..."

---

## 4. Contractions

### main.tex

| Line | Contraction | Content (excerpt) |
|------|-------------|-------------------|
| 177 | `don't` | `explicit ``I don't know'' channel that fixed-threshold classifiers lack.` |

### section_results.tex — None found
### section_discussion_to_appendices.tex — None found

**Total contraction instances: 1**

**Recommendation:** This is inside a quoted conceptual phrase ("I don't know" channel). Consider rephrasing to: `explicit ``uncertainty'' channel` or `explicit abstention channel`. Contractions are inappropriate in formal academic writing even inside conceptual labels.

---

## 5. Informal Language

### main.tex

| Line | Phrase | Content (excerpt) |
|------|--------|-------------------|
| 140 | `stop short of` | `declare the model ``promising,'' and stop short of addressing the operational question` |
| 160 | `the key challenge` | `the key challenge is not how to extract another $0.01$ of AUROC` |

### section_results.tex

| Line | Phrase | Content (excerpt) |
|------|--------|-------------------|
| 80 | `but it speaks infrequently` | `...it is relatively reliable (PPV 0.643 for alerts, NPV 0.948 for clears), but it speaks infrequently.` |
| 83 | `when it speaks` | Anthropomorphizing the model — `when it speaks, it is relatively reliable` |

### section_discussion_to_appendices.tex

| Line | Phrase | Content (excerpt) |
|------|--------|-------------------|
| 20 | `This is not a limitation but a feature` | Software-industry idiom, not academic register |
| 52 | `benchmark story` | `The conventional benchmark story---XGBoost with isotonic calibration...` — informal framing |

**Total informal language instances: 6**

**Recommendation:**
- "stop short of" → "do not address" or "omit"
- "the key challenge" → "the central challenge" or "the primary challenge"
- "but it speaks infrequently" → "but it issues recommendations infrequently"
- "when it speaks" → "when it issues a recommendation"
- "not a limitation but a feature" → "a deliberate design property" or "an intentional mechanism"
- "benchmark story" → "benchmark analysis" or "benchmark comparison"

---

## 6. Bullet Point / Enumerated Lists That Should Be Prose

### main.tex

| Lines | Section | Description |
|-------|---------|-------------|
| 253–262 | §2.2 Inclusion criteria | `\begin{enumerate}` with 3 items — could be a prose paragraph |
| 280–290 | §2.2 KDIGO criteria | `\begin{itemize}` with 3 items — could be prose |
| 310–318 | §2.2 Exclusion criteria | `\begin{enumerate}` with 2 items — could be prose |
| 337–356 | §2.3 Time-varying features | `\begin{enumerate}` with 9 items — acceptable as a definition list, but long |
| 389–397 | §2.4 Conventional baselines | `\begin{itemize}` with 3 items — could be prose |
| 460–468 | §2.5 Set-valued predictions | `\begin{itemize}` with 3 items (Alert/Clear/Defer) — acceptable as definitions |
| 500–506 | §2.6 Disagreement models | `\begin{itemize}` with 2 items — should be prose |

### section_discussion_to_appendices.tex

| Lines | Section | Description |
|-------|---------|-------------|
| 28–49 | Limitations | `\begin{enumerate}` with 10 items — standard for limitations, acceptable |
| 260–264 | Reproducibility | `\begin{enumerate}` with 3 items — pipeline steps, could be prose |

### section_results.tex — No bullet/enumerated lists in body text (all lists are in tables).

**Recommendation:** The inclusion criteria (3 items), exclusion criteria (2 items), and disagreement model descriptions (2 items) in main.tex should be converted to flowing prose paragraphs. The 9-item feature aggregation list and the Alert/Clear/Defer definitions are acceptable as lists. The Limitations enumeration is standard journal practice and can remain.

---

## 7. `\texttt{}` References to CSV Filenames and Internal Code Artifacts

### section_results.tex

| Line | Reference | Issue |
|------|-----------|-------|
| 6 | `\texttt{mimic\_saaki\_raw\_v2.csv}` | Raw CSV filename exposed — should not appear in a published paper |
| 6 | `\texttt{event\_observed}` | Internal variable name — rephrase as "the binary mortality indicator" |
| 6 | `\texttt{subject\_id}` | Internal column name — rephrase as "unique subject identifier" |
| 29 | `\texttt{time\_to\_event\_hrs}` (in figure caption) | Internal variable name in a caption |
| 43 | `\texttt{time\_to\_event\_hrs}` | Internal variable name in body text |

### main.tex

| Line | Reference | Issue |
|------|-----------|-------|
| 60 | `\texttt{ch23m514@smail.iitm.ac.in}` and `\texttt{kaustabhganguly@gmail.com}` | Author emails — acceptable in author block |
| 224–226 | `\texttt{hosp}`, `\texttt{icu}`, `\texttt{ed}` | MIMIC-IV module names — acceptable as database identifiers |
| 321 | `\texttt{event\_observed}` | Internal variable name |
| 322 | `\texttt{time\_to\_event\_hrs}` | Internal variable name |
| 543 | `\texttt{subject\_id}` | Internal column name |

### section_discussion_to_appendices.tex

| Line | Reference | Issue |
|------|-----------|-------|
| 10 | `\texttt{time\_to\_event\_hrs}` | Internal variable name |
| 261 | `\texttt{saaki/deployment\_analysis.py}` | Internal script path |
| 261 | `\texttt{local\_outputs/artifacts/}` | Internal directory path |
| 262 | `\texttt{saaki/build\_jamia\_manuscript.py}` | Internal script path |
| 262 | `\texttt{saaki/jamia\_manuscript/}` | Internal directory path |
| 263 | `\texttt{saaki/build\_jamia\_tex.py}` | Internal script path |
| 263 | `\texttt{main.tex}`, `\texttt{refs.bib}` | Internal file names |
| 263 | `\texttt{jamia\_tex/}` | Internal directory path |

**Critical issues (must fix):**
- `\texttt{mimic\_saaki\_raw\_v2.csv}` (section_results.tex, line 6) — a raw CSV filename has no place in a published paper. Rephrase as "the analytic dataset" or "the derived SA-AKI cohort file."
- `\texttt{event\_observed}` — replace with "the binary in-hospital mortality indicator" or define it once and use the clinical term thereafter.
- `\texttt{time\_to\_event\_hrs}` — replace with "time-to-event (hours)" or "the time-to-event variable."
- `\texttt{subject\_id}` — replace with "unique subject identifier" or "patient-level grouping key."

**Acceptable uses:**
- `\texttt{hosp}`, `\texttt{icu}`, `\texttt{ed}` — these are official MIMIC-IV module names and are appropriate.
- Author email addresses — standard practice.
- Reproducibility appendix script paths (lines 261–263) — borderline acceptable in a reproducibility section, but consider whether these internal paths add value for readers who cannot access the repository.

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Em dashes (`---`) in body text | ~38 |
| "We" usage (sentence starters + mid-sentence) | 6 |
| Filler phrases | 1 |
| Contractions | 1 |
| Informal language | 6 |
| Lists that should be prose | 4–5 |
| `\texttt{}` CSV/variable/path references to fix | 12 (4 critical, 8 review) |

**Priority order for fixes:**
1. **Em dashes** — highest volume, strongest AI signal, fix all ~38
2. **`\texttt{}` CSV/variable names** — the `mimic_saaki_raw_v2.csv` reference is the most egregious single issue
3. **Informal language** — 6 instances that undermine academic register
4. **"We" sentence starters** — 4 instances to convert to passive voice
5. **Lists → prose** — 4–5 short lists to convert
6. **Contraction** — 1 instance
7. **Filler phrase** — 1 instance
