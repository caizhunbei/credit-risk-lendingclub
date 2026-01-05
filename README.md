# LendingClub Credit Risk Modeling (PD/LGD/EL)

End-to-end credit risk pipeline using LendingClub loan-level data.

## Highlights

- Largest EL by grade: **C** (sum_EL ≈ 2,716,515,691)
- Highest EL intensity by purpose: **small_business** (EL_per_EAD ≈ 56.77%)
- Total expected loss (approx): **8,671,378,033**

## Method

- **PD**: Logistic Regression (time split: train ≤2016, test ≥2017)

- **LGD**: Two-stage

  1) P(recovery>0)

  2) recovery_rate | recovery>0 (logit transform + Ridge)

- **EL**: `PD_hat × LGD_hat × EAD`

## Outputs

- `reports/summary.md` (auto-generated)

- `reports/el_scored.parquet`

- `reports/top_risk_loans.csv`

- `reports/el_by_grade.csv`, `reports/el_by_term.csv`, `reports/el_by_purpose.csv`

## How to run

```bash
python scripts/05_el_merge_and_report.py
python scripts/06_el_add_dims_and_group_reports.py
python scripts/07_make_readme_summary.py
```
