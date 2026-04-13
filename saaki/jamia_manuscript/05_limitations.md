# Limitations

This work has several important limitations.

1. The study remains internally validated only. The current checked-in cohort does not expose a defensible calendar timestamp axis for a true temporal split, and no external cohort is included, so the manuscript supports strong internal claims and cautious deployment framing rather than transportability claims.
2. The cohort remains affected by ETL and documentation mismatches already documented in the audit and data-contract artifacts.
3. Repeated patients remain present in the working cohort, which is why subject-grouped evaluation is mandatory.
4. The process-rich feature space may not transport as well as physiology-only inputs.
5. The conformal guarantees are marginal under exchangeability and do not imply perfect subgroup-conditional coverage.
6. The secondary horizon results, especially the 48-hour endpoint, are limited by low event counts and should be treated as sensitivity analyses rather than standalone deployment targets.
7. The manuscript focuses on retrospective decision support and does not evaluate clinical workflow adoption, clinician behavior change, or prospective impact.
8. The disagreement-based baseline is heuristic by design and is included as a comparison, not as the final methodological recommendation.
