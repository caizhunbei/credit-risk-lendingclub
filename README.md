# LendingClub 信用风险建模项目：PD / LGD / 预期损失（EL）

本项目基于 LendingClub 借款表现数据，搭建一个面向**金融工程 / 风险分析 / 数据分析实习**的端到端信用风险建模流程（pipeline），目标是输出每笔贷款的：
- **PD（违约概率）**：`pd_hat`
- **LGD（违约损失率）**：`lgd_hat`
- **EAD（违约暴露）**：`EAD`
- **EL（预期损失金额）**：`EL_amount = pd_hat × lgd_hat × EAD`

同时生成按 **grade / term / purpose** 等维度的组合汇总报表，便于做风险排序、组合损失分布分析与策略筛选。

> 说明：原始数据体积较大（parquet/csv 分片），本仓库不上传全量数据，只上传脚本与轻量级报告，保证作品集可展示、可复现。

---

## 1. 你将看到什么（项目产出）

运行完成后，你会得到两类核心产出：

### （1）逐笔贷款的风险评分结果
- `pd_hat`：违约概率预测
- `lgd_hat`：违约损失率预测
- `EAD`：暴露金额（简化口径）
- `EL_amount`：预期损失金额（用于排序/筛选）

### （2）组合维度的风险报表（Portfolio Reports）
- `reports/summary.md`：项目摘要（样本规模、指标、EL 分布）
- `reports/el_by_grade.csv`：按等级汇总
- `reports/el_by_term.csv`：按期限汇总
- `reports/el_by_purpose.csv`：按用途汇总

---

## 2. 方法框架（从业务到模型）

信用风险的经典分解是：

- **违约发生概率（PD）**：这笔贷款未来会不会违约？
- **违约后的损失程度（LGD）**：一旦违约能回收多少？损失多少？
- **违约时的暴露（EAD）**：违约发生时还剩多少未偿余额/金额？

因此预期损失为：

> **EL = PD × LGD × EAD**

本项目严格按这个结构实现，并将三者输出到同一张表，便于后续做：
- Top 风险贷款筛选（按 EL 排序）
- 组合风险分层（按 grade/term/purpose）
- 风险策略对比（例如不同筛选规则下的平均 EL）

---

## 3. 数据与切分（避免信息泄漏）

### 数据来源
LendingClub accepted loans（Kaggle 常见公开版本）。

### 关键字段（概念层面）
- `issue_d`：放款日期（用于时间切分）
- `loan_status`：贷款状态（用于构造违约标签与回收相关变量）
- 数值特征：贷款金额、收入、负债比、利率等
- 类别特征：等级、用途、期限、地区等

### 时间切分（重点）
为了避免“前视偏差/信息泄漏”，本项目采用**时间切分评估**：
- 训练集：`year <= 2016`
- 测试集：`year >= 2017`

---

## 4. 模型设计（PD / LGD）

## 4.1 PD 模型（违约概率）
- **目标变量**：`y_bad`（由 `loan_status` 构造的违约标记）
- **特征处理**：
  - 数值特征：中位数填充（`median`）
  - 类别特征：众数填充 + One-Hot
- **模型**：Logistic Regression（`solver="saga"`，适合稀疏高维；`class_weight="balanced"` 应对不平衡）
- **评价指标**：AUC、PR-AUC（PR-AUC 能更真实反映不平衡数据的效果）

> 你的一次基线运行中，PD 模型达到约：AUC≈0.70，PR-AUC≈0.37（明显高于随机 PR 基线≈坏账率）。

---

## 4.2 LGD 模型（违约损失率，两阶段）
LGD 的难点在于“回收”本身具有**两层不确定性**：
1) 会不会发生回收（很多违约最终回收为 0）
2) 如果发生回收，能回收多少（回收率分布偏斜）

因此本项目采用更贴近实务的 **Two-Stage LGD**：

### Stage 1：回收事件（是否有回收）
- 目标：`has_recovery`（0/1）
- 输出：`p_recovery_hat`（发生回收的概率）

### Stage 2：回收率（仅在发生回收的样本子集上）
- 目标：`recovery_rate`（0~1）
- 输出：`recovery_rate_hat_pos`（“发生回收条件下”的回收率预测）

### 合成 LGD
- `recovery_rate_hat = p_recovery_hat * recovery_rate_hat_pos`
- **`lgd_hat = 1 - recovery_rate_hat`**

> 说明：LGD 通常比 PD 更难做得“很好看”，本项目目标是构建可复现、可用于 EL 估计与风险排序的实用 pipeline。

---

## 5. EAD 与 EL（落地输出）

### EAD（简化口径）
本项目使用贷款金额/放款金额作为 EAD 的近似（作品集实现口径），用于演示 EL 的完整计算链路。

### EL（预期损失金额）
最终逐笔计算：

> **`EL_amount = pd_hat × lgd_hat × EAD`**

并输出：
- `reports/top_risk_loans.csv`（按 EL 排序的高风险贷款列表）
- 分维度汇总报表（grade/term/purpose）

---

## 6. 仓库结构（推荐的作品集组织方式）
├── scripts/
│ ├── 01_build_model_tables.py # 数据准备：生成 PD/LGD 建模表
│ ├── 02_train_pd_logit.py # 训练 PD 模型（时间切分/抽样）
│ ├── 03_pd_predict_and_save.py # 全量评分：输出 pd_hat 与 EAD
│ ├── 04_lgd_train_score_and_save_v3.py # 训练 + 评分 LGD（两阶段）
│ ├── 05_el_merge_and_report.py # 合并 PD/LGD/EAD 并生成 EL 报表
│ ├── 06_el_add_dims_and_group_reports.py# 补齐维度并输出分组汇总
│ └── 07_make_readme_summary.py # 生成 reports/summary.md（摘要）
├── reports/
│ ├── summary.md
│ └── README_snippet.md
├── requirements.txt
├── .gitignore
└── README.md


---

## 7. 如何本地复现（从 0 跑到结果）

### 7.1 安装依赖
```bash
pip install -r requirements.txt

7.2 按顺序运行脚本
python scripts/01_build_model_tables.py
python scripts/02_train_pd_logit.py
python scripts/03_pd_predict_and_save.py
python scripts/04_lgd_train_score_and_save_v3.py
python scripts/05_el_merge_and_report.py
python scripts/06_el_add_dims_and_group_reports.py
python scripts/07_make_readme_summary.py


运行完成后，重点查看：

reports/summary.md

reports/el_by_grade.csv / reports/el_by_term.csv / reports/el_by_purpose.csv

8. 作品集说明（为什么这个项目有“专业差异性”）

不是简单“预测违约”，而是完整拆解并实现 PD + LGD + EL 的信用风险框架

使用时间切分评估，更符合真实业务上线场景

LGD 采用更实务的两阶段方式，避免“回收率直接回归”造成的偏差

输出不仅是模型指标，还包括风险报表与可直接用于筛选的结果文件

9. 后续可扩展（加分项方向）

PD 概率校准（Platt/Isotonic）、PSI 稳定性/漂移监控

分箱与 scorecard 输出（更贴近银行风控落地）

使用 LightGBM/XGBoost 并严格控制泄漏

LGD 用 Beta 回归/分位数回归等改进回收率拟合

从 EL 扩展到组合风险：UL、VaR/ES、vintage curve 等

作者

GitHub：caizhunbei

::contentReference[oaicite:0]{index=0}

