# Appendix A: Data And Cohort

## Audit Summary

- Working dataset: `mimic_saaki_raw_v2.csv`
- Rows: `10036`
- Columns: `254`
- Unique subjects: `9002`
- Repeated subject rows: `1034`
- Event prevalence: `0.267`

## Data Contract Notes

- Row definition: One ICU-stay-level SA-AKI cohort row scored at T24 using features derived from the first 24 hours of ICU stay; repeated patients may still appear across multiple rows.
- Prediction time: 24 hours after ICU admission (T24).
- Target definition: `event_observed` is the in-hospital death label; `time_to_event_hrs` is hours from ICU admission to death or last known alive/discharge censoring.
- Time anchor: All predictor features must be available by T24; post-T24 information is excluded from deployment claims even if present elsewhere in project documents.

## Explicitly Excluded Claims

- No bedside-readiness claim.
- No external-validation or multi-center transportability claim.
- No claim that the v2 cohort exactly matches the thesis cohort description without ETL reconciliation.
- No reliance on stay-level metrics for deployment-readiness claims.
