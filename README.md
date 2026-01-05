# 信用风险建模（LendingClub）— PD / LGD / 预期损失 EL  
# Credit Risk Modeling (LendingClub) — PD / LGD / Expected Loss (EL)

---

## 项目简介（中文）
本项目基于 LendingClub 借款表现数据，构建一个面向**金融工程 / 风险分析 / 数据分析实习**的端到端信用风险 pipeline，涵盖：
- **PD（Probability of Default）违约概率**建模与评分
- **LGD（Loss Given Default）违约损失率**两阶段建模（是否回收 + 回收率）
- **EL（Expected Loss）预期损失**：逐笔贷款级别输出 `PD_hat`、`LGD_hat`、`EAD` 与 `EL_amount`，并提供组合维度汇总（如 grade / term / purpose）

> 说明：原始数据体积较大，本仓库不上传全量数据，仅提供可复现脚本与轻量级报告（适合作品集展示）。

---

## Overview (English)
This is a **portfolio-ready** end-to-end credit risk project using LendingClub loan performance data:
- **PD (Probability of Default)** modeling & scoring  
- **LGD (Loss Given Default)** two-stage modeling (recovery event + recovery rate)  
- **EL (Expected Loss)** at loan level: `PD_hat`, `LGD_hat`, `EAD`, and `EL_amount`, plus grouped portfolio reports (e.g., grade / term / purpose)

> Note: Raw data is large and is NOT committed to this repo. This repository focuses on reproducible code + lightweight summaries.

---

## 亮点 / Highlights
- **时间切分评估**（train ≤ 2016, test ≥ 2017），降低前视偏差  
  **Time-based split** to reduce look-ahead bias  
- **高维稀疏特征处理**：One-Hot + `LogisticRegression(saga)`，可在大样本上跑通  
  **Sparse high-dimensional modeling** with OHE + `LogReg(saga)`  
- **LGD 两阶段更贴近业务**：先预测是否回收，再预测回收率并合成 LGD  
  **Two-stage LGD**: recovery probability + recovery rate  
- 输出可直接用于**风险排序 / 组合损失分布 / 风险偏好策略筛选**  
  Outputs support risk ranking, portfolio loss profiling, and screening rules.

---

## 仓库结构 / Repo Structure
建议的作品集结构如下（只上传代码与轻量报告）：

```text
.
├── scripts/
│   ├── 01_build_model_tables.py
│   ├── 02_train_pd_logit.py
│   ├── 03_pd_predict_and_save.py
│   ├── 04_lgd_train_score_and_save_v3.py
│   ├── 05_el_merge_and_report.py
│   ├── 06_el_add_dims_and_group_reports.py
│   └── 07_make_readme_summary.py
├── reports/
│   ├── summary.md
│   └── README_snippet.md
├── requirements.txt
├── .gitignore
└── README.md
