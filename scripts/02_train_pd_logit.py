import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# ===== 1) 只读必要列（更省内存更快）=====
need_cols = [
    "id","issue_d","y_bad","loan_status",
    "loan_amnt","funded_amnt","term","int_rate",
    "grade","sub_grade","emp_length","home_ownership","annual_inc",
    "verification_status","purpose","addr_state","dti",
    "delinq_2yrs","inq_last_6mths","open_acc","pub_rec","revol_bal",
    "revol_util","total_acc","fico_range_low","fico_range_high",
]
df = pd.read_parquet("data_raw/model/pd_model.parquet", columns=need_cols)

df = df[df["issue_d"].notna()].copy()
df["year"] = pd.to_datetime(df["issue_d"]).dt.year

train = df[df["year"] <= 2016].copy()
test  = df[df["year"] >= 2017].copy()

print("train rows:", len(train), "bad_rate:", train["y_bad"].mean())
print("test  rows:", len(test),  "bad_rate:", test["y_bad"].mean())

# ===== 2) 稳定的分层抽样（不会超抽，也不会比例跑偏）=====
n_train = 300_000
n_test  = 200_000

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

train_s = stratified_sample_binary(train, "y_bad", n_train, seed=42)
test_s  = stratified_sample_binary(test,  "y_bad", n_test,  seed=43)

print("train_s bad_rate:", train_s["y_bad"].mean(), " test_s bad_rate:", test_s["y_bad"].mean())

y_train = train_s["y_bad"]
y_test  = test_s["y_bad"]

drop_cols = {"y_bad","loan_status","issue_d","id","year"}
X_train = train_s[[c for c in train_s.columns if c not in drop_cols]].copy()
X_test  = test_s[[c for c in test_s.columns if c not in drop_cols]].copy()

cat_cols = X_train.select_dtypes(include=["object","string"]).columns.tolist()
num_cols = [c for c in X_train.columns if c not in cat_cols]

# StringDtype 的 <NA> -> np.nan
for col in cat_cols:
    X_train[col] = X_train[col].astype(object).where(X_train[col].notna(), np.nan)
    X_test[col]  = X_test[col].astype(object).where(X_test[col].notna(), np.nan)

# ===== 3) OHE 降维（低频合并）=====
try:
    ohe = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=0.005,   # 比 0.01 温和一点，信息损失更少
        sparse_output=True,
        dtype=np.float32
    )
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore")

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))  # 对稀疏友好，帮助收敛
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("oh", ohe)
    ]), cat_cols),
])

# ===== 4) Logit：更容易收敛的设置 =====
model = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(
        solver="saga",
        penalty="l2",
        C=0.5,          # 正则更强一点，更稳也更易收敛
        max_iter=1000,
        tol=1e-3,
        n_jobs=-1,
        class_weight="balanced",
        verbose=1
    ))
])

model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]

print("Train size:", X_train.shape, "Test size:", X_test.shape)
print("Test AUC   =", roc_auc_score(y_test, proba))
print("Test PR-AUC=", average_precision_score(y_test, proba))
print("Random PR baseline (≈ bad_rate):", y_test.mean())
