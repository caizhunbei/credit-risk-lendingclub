# LendingClub Credit Risk (PD/LGD/EL) — Summary

_Generated: 2026-01-05 14:24_

## Project Overview

- Dataset: LendingClub loan-level data (Parquet shards merged into modeling tables)

- Goal: Build an end-to-end credit risk pipeline: **PD → LGD → EL**

- Modeling:

  - **PD**: logistic regression (time-based split: train ≤2016, test ≥2017)

  - **LGD**: two-stage approach

    - Stage 1: P(recovery > 0)

    - Stage 2: recovery rate | recovery>0 (logit transform + Ridge)

  - **EL**: `EL_amount = PD_hat × LGD_hat × EAD` ; `EL_rate = PD_hat × LGD_hat`

## Key Findings (from reports)

- Estimated total expected loss (sum over loans): **8,671,378,033**

- **Grade** — largest total EL: `C` (sum_EL=2,716,515,691)
- **Grade** — highest EL intensity (EL_per_EAD): `G` (EL_per_EAD=75.13%)
- **Term** — largest total EL: ` 36 months` (sum_EL=4,758,507,991)
- **Term** — highest EL intensity (EL_per_EAD): ` 60 months` (EL_per_EAD=60.91%)
- **Purpose** — largest total EL: `debt_consolidation` (sum_EL=5,535,690,367)
- **Purpose** — highest EL intensity (EL_per_EAD): `small_business` (EL_per_EAD=56.77%)

## Breakdown Tables (Top rows)

### EL by Grade (Top)

| grade   |      n |   bad_rate |   mean_PD |   mean_LGD |   mean_EL_rate |      sum_EL |   EL_per_EAD |
|:--------|-------:|-----------:|----------:|-----------:|---------------:|------------:|-------------:|
| C       | 373583 |  0.225655  |  0.519753 |   0.962115 |       0.500005 | 2.71652e+09 |     0.510029 |
| B       | 380713 |  0.134335  |  0.37039  |   0.968876 |       0.358721 | 1.83753e+09 |     0.362783 |
| D       | 195641 |  0.306362  |  0.622641 |   0.958818 |       0.596921 | 1.82809e+09 |     0.608606 |
| E       |  90798 |  0.388951  |  0.699461 |   0.955105 |       0.667882 | 1.08739e+09 |     0.677674 |
| A       | 225010 |  0.0604595 |  0.202442 |   0.976284 |       0.197523 | 6.31742e+08 |     0.198724 |
| F       |  31005 |  0.456991  |  0.761764 |   0.951603 |       0.724763 | 4.33723e+08 |     0.732113 |
| G       |   8814 |  0.5059    |  0.790591 |   0.947294 |       0.748824 | 1.36396e+08 |     0.75126  |


### EL by Term (Top)

| term      |      n |   bad_rate |   mean_PD |   mean_LGD |   mean_EL_rate |      sum_EL |   EL_per_EAD |
|:----------|-------:|-----------:|----------:|-----------:|---------------:|------------:|-------------:|
| 36 months | 991672 |   0.161394 |  0.400241 |   0.967127 |       0.386169 | 4.75851e+09 |     0.3798   |
| 60 months | 313892 |   0.327753 |  0.636326 |   0.959099 |       0.609607 | 3.91287e+09 |     0.609139 |


### EL by Purpose (Top)

| purpose            |      n |   bad_rate |   mean_PD |   mean_LGD |   mean_EL_rate |      sum_EL |   EL_per_EAD |
|:-------------------|-------:|-----------:|----------:|-----------:|---------------:|------------:|-------------:|
| debt_consolidation | 761666 |   0.213011 |  0.475595 |   0.964585 |       0.45759  | 5.53569e+09 |     0.475215 |
| credit_card        | 290148 |   0.170396 |  0.409851 |   0.968477 |       0.395859 | 1.76733e+09 |     0.409754 |
| home_improvement   |  84522 |   0.179291 |  0.419305 |   0.964221 |       0.402862 | 5.13976e+08 |     0.427148 |
| other              |  73876 |   0.213195 |  0.475791 |   0.961569 |       0.456453 | 3.48646e+08 |     0.475526 |
| major_purchase     |  27239 |   0.192848 |  0.450295 |   0.965376 |       0.433349 | 1.53357e+08 |     0.464013 |
| small_business     |  13585 |   0.301877 |  0.580146 |   0.962562 |       0.55735  | 1.22772e+08 |     0.567747 |
| medical            |  14861 |   0.220914 |  0.478982 |   0.962237 |       0.459811 | 6.42702e+07 |     0.478708 |
| house              |   6872 |   0.222497 |  0.477536 |   0.957669 |       0.455828 | 4.95675e+07 |     0.464578 |


## Top Risk Loans (Top 20 by EL_amount)

|        id | issue_d    |   year | loan_status   | EAD    | pd_hat   | lgd_hat   | EL_rate   | EL_amount   |
|----------:|:-----------|-------:|:--------------|:-------|:---------|:----------|:----------|:------------|
|  91974215 | 2016-11-01 |   2016 | Charged Off   | 40,000 | 100.00%  | 100.00%   | 100.00%   | 39,999      |
| 113845057 | 2017-08-01 |   2017 | Charged Off   | 40,000 | 100.00%  | 99.94%    | 99.94%    | 39,976      |
| 129738450 | 2018-03-01 |   2018 | Fully Paid    | 40,000 | 99.90%   | 99.86%    | 99.76%    | 39,906      |
| 143007132 | 2018-11-01 |   2018 | Fully Paid    | 40,000 | 99.89%   | 99.21%    | 99.10%    | 39,638      |
| 113203256 | 2017-07-01 |   2017 | Charged Off   | 40,000 | 99.20%   | 98.87%    | 98.08%    | 39,232      |
| 131485404 | 2018-04-01 |   2018 | Fully Paid    | 40,000 | 98.89%   | 97.91%    | 96.82%    | 38,729      |
| 118207022 | 2017-09-01 |   2017 | Charged Off   | 40,000 | 98.85%   | 97.81%    | 96.68%    | 38,672      |
| 138634747 | 2018-08-01 |   2018 | Fully Paid    | 40,000 | 97.82%   | 98.57%    | 96.42%    | 38,567      |
| 123935285 | 2017-11-01 |   2017 | Fully Paid    | 40,000 | 98.52%   | 97.12%    | 95.68%    | 38,272      |
| 113192566 | 2017-07-01 |   2017 | Charged Off   | 40,000 | 98.55%   | 96.64%    | 95.23%    | 38,092      |
| 121712717 | 2017-10-01 |   2017 | Fully Paid    | 37,500 | 100.00%  | 99.96%    | 99.96%    | 37,484      |
| 135312865 | 2018-06-01 |   2018 | Fully Paid    | 40,000 | 95.81%   | 97.37%    | 93.28%    | 37,312      |
| 132614251 | 2018-05-01 |   2018 | Fully Paid    | 37,200 | 100.00%  | 99.91%    | 99.91%    | 37,166      |
|  99984927 | 2017-03-01 |   2017 | Fully Paid    | 40,000 | 95.31%   | 97.40%    | 92.83%    | 37,133      |
| 133830013 | 2018-06-01 |   2018 | Fully Paid    | 40,000 | 94.77%   | 97.45%    | 92.35%    | 36,939      |
| 130242528 | 2018-03-01 |   2018 | Fully Paid    | 37,000 | 99.99%   | 99.78%    | 99.77%    | 36,915      |
| 140735932 | 2018-09-01 |   2018 | Fully Paid    | 40,000 | 94.38%   | 96.89%    | 91.44%    | 36,577      |
| 116879174 | 2017-08-01 |   2017 | Charged Off   | 36,000 | 99.90%   | 99.19%    | 99.09%    | 35,672      |
| 127769204 | 2018-01-01 |   2018 | Fully Paid    | 40,000 | 91.48%   | 96.68%    | 88.45%    | 35,379      |
| 132995790 | 2018-05-01 |   2018 | Fully Paid    | 40,000 | 90.81%   | 97.10%    | 88.18%    | 35,272      |


## Reproducibility

- Key outputs:

  - `reports/el_scored.parquet`

  - `reports/top_risk_loans.csv`

  - `reports/el_by_grade.csv`, `reports/el_by_term.csv`, `reports/el_by_purpose.csv`

- Run order:

  1. Train/score PD → `pd_scored.parquet`

  2. Train/score LGD → `lgd_scored.parquet`

  3. Merge to EL + reports → `el_scored.parquet` + CSVs

  4. Generate this summary → `summary.md`
