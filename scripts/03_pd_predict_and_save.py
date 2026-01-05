import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# ========= 配置 =========
PD_PATH = "data_raw/model/pd_model.parquet"
OUT_PATH = "data_raw/model/pd_scored.parquet"

TRAIN_END_YEAR = 2016
SCORE_ALL = True     # True: 全量打分；False: 只打分 test(>=2017)

N_TRAIN = 300_000    # 抽样训练（想更强就加大，比如 500_000）
SEED = 42

# ========= 读数据（只读需要的列）=========
need_cols = [
    "id","issue_d","y_bad","loan_status",
    "loan_amnt","funded_amnt","term","int_rate",
    "grade","sub_grade","emp_length","home_ownership","annual_inc",
    "verification_status","purpose","addr_state","dti",
    "delinq_2yrs","inq_last_6mths","open_acc","pub_rec","revol_bal",
    "revol_util","total_acc","fico_range_low","fico_range_high",
]
df = pd.read_parquet(PD_PATH, columns=need_cols)

df = df[df["issue_d"].notna()].copy()
df["issue_d"] = pd.to_datetime(df["issue_d"])
df["year"] = df["issue_d"].dt.year

train_full = df[df["year"] <= TRAIN_END_YEAR].copy()
test_full  = df[df["year"] >= TRAIN_END_YEAR + 1].copy()

print("train_full:", train_full.shape, "bad_rate:", train_full["y_bad"].mean())
print("test_full :", test_full.shape,  "bad_rate:", test_full["y_bad"].mean())

# ========= 分层抽样（稳）=========
def stratified_sample_binary(d: pd.DataFrame, y_col: str, n: int, seed=42) -> pd.DataFrame:
    n = min(n, len(d))
    vc = d[y_col].value_counts()
    n1_avail = int(vc.get(1, 0))
    n0_avail = int(vc.get(0, 0))

    if n0_avail == 0 or n1_avail == 0:
        return d.sample(n=n, random_state=seed)

    p1 = n1_avail / (n0_avail + n1_avail)
    n1 = int(round(n * p1))
    n0 = n - n1

    n1 = min(n1, n1_avail)
    n0 = min(n0, n0_avail)

    remaining = n - (n0 + n1)
    if remaining > 0:
        cap0 = n0_avail - n0
        cap1 = n1_avail - n1
        add0 = min(remaining, cap0); n0 += add0; remaining -= add0
        add1 = min(remaining, cap1); n1 += add1; remaining -= add1

    d0 = d[d[y_col] == 0].sample(n=n0, random_state=seed)
    d1 = d[d[y_col] == 1].sample(n=n1, random_state=seed)
    return pd.concat([d0, d1], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

train = stratified_sample_binary(train_full, "y_bad", N_TRAIN, seed=SEED)

# ========= 拆分 X/y =========
y_train = train["y_bad"]

drop_cols = {"y_bad","loan_status","issue_d","id","year"}
X_train = train[[c for c in train.columns if c not in drop_cols]].copy()

# 评分数据集（全量 or test）
score_df = df if SCORE_ALL else test_full
X_score = score_df[[c for c in score_df.columns if c not in drop_cols]].copy()

cat_cols = X_train.select_dtypes(include=["object","string"]).columns.tolist()
num_cols = [c for c in X_train.columns if c not in cat_cols]

# StringDtype 的 <NA> -> np.nan
for col in cat_cols:
    X_train[col] = X_train[col].astype(object).where(X_train[col].notna(), np.nan)
    X_score[col] = X_score[col].astype(object).where(X_score[col].notna(), np.nan)

# ========= 预处理 + 模型 =========
try:
    ohe = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=0.005,
        sparse_output=True,
        dtype=np.float32
    )
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore")

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("oh", ohe),
    ]), cat_cols),
])

clf = LogisticRegression(
    solver="saga",
    penalty="l2",
    C=0.5,
    max_iter=1000,
    tol=1e-3,
    n_jobs=-1,
    class_weight="balanced",
    verbose=1
)

model = Pipeline([
    ("prep", preprocess),
    ("clf", clf),
])

# ========= 训练 =========
model.fit(X_train, y_train)

# ========= 在 test 上做一次评估（全量 test，不抽样）=========
X_test = test_full[[c for c in test_full.columns if c not in drop_cols]].copy()
for col in cat_cols:
    X_test[col] = X_test[col].astype(object).where(X_test[col].notna(), np.nan)

proba_test = model.predict_proba(X_test)[:, 1]
print("TEST AUC   =", roc_auc_score(test_full["y_bad"], proba_test))
print("TEST PR-AUC=", average_precision_score(test_full["y_bad"], proba_test))
print("Random PR baseline (bad_rate):", test_full["y_bad"].mean())

# ========= 打分并保存 =========
pd_hat = model.predict_proba(X_score)[:, 1]

out = score_df[["id","issue_d","year","y_bad","loan_status","loan_amnt","funded_amnt"]].copy()
out["pd_hat"] = pd_hat
# EAD 近似
out["EAD"] = out["funded_amnt"].fillna(out["loan_amnt"])

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(OUT_PATH, index=False)

print(f"[OK] saved scored file -> {OUT_PATH}")
print(out.head())
