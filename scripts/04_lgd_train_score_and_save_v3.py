import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

# =========================
# 配置
# =========================
LGD_TRAIN_PATH = "data_raw/model/lgd_model.parquet"
PD_ALL_PATH    = "data_raw/model/pd_model.parquet"
OUT_PATH       = "data_raw/model/lgd_scored.parquet"

TRAIN_END_YEAR = 2016
N_TRAIN_BAD    = 220_000
SEED = 42

# logit 变换保护
EPS2 = 1e-3

feat_cols = [
    "loan_amnt","funded_amnt","term","int_rate",
    "grade","sub_grade","emp_length","home_ownership","annual_inc",
    "verification_status","purpose","addr_state","dti",
    "delinq_2yrs","inq_last_6mths","open_acc","pub_rec","revol_bal",
    "revol_util","total_acc","fico_range_low","fico_range_high",
]

# =========================
# 工具函数
# =========================
def to_datetime_safe(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def convert_cat_na(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return X

def stratified_sample_binary(d: pd.DataFrame, y_col: str, n: int, seed: int = 42) -> pd.DataFrame:
    n = min(n, len(d))
    vc = d[y_col].value_counts()
    n1_avail = int(vc.get(1, 0))
    n0_avail = int(vc.get(0, 0))
    if n0_avail == 0 or n1_avail == 0:
        return d.sample(n=n, random_state=seed)

    p1 = n1_avail / (n0_avail + n1_avail)
    n1 = int(round(n * p1)); n0 = n - n1
    n1 = min(n1, n1_avail); n0 = min(n0, n0_avail)

    remaining = n - (n0 + n1)
    if remaining > 0:
        cap0 = n0_avail - n0
        cap1 = n1_avail - n1
        add0 = min(remaining, cap0); n0 += add0; remaining -= add0
        add1 = min(remaining, cap1); n1 += add1; remaining -= add1

    d0 = d[d[y_col] == 0].sample(n=n0, random_state=seed)
    d1 = d[d[y_col] == 1].sample(n=n1, random_state=seed)
    return pd.concat([d0, d1], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

def make_preprocess(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    try:
        ohe = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=0.005,
            sparse_output=True,
            dtype=np.float32
        )
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore")

    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("oh", ohe),
        ]), cat_cols),
    ])

def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1 / (1 + np.exp(-x))

def unique_keep_order(cols):
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

# =========================
# 0) 读 LGD 训练集（坏账样本）
# =========================
need_lgd_cols = ["id","issue_d","loan_status","EAD","lgd","recovery_rate","has_recovery"] + feat_cols
lgd_df = pd.read_parquet(LGD_TRAIN_PATH, columns=need_lgd_cols)

lgd_df = lgd_df[lgd_df["issue_d"].notna()].copy()
lgd_df["issue_d"] = to_datetime_safe(lgd_df["issue_d"])
lgd_df = lgd_df[lgd_df["issue_d"].notna()].copy()
lgd_df["year"] = lgd_df["issue_d"].dt.year

train_full = lgd_df[lgd_df["year"] <= TRAIN_END_YEAR].copy()
test_full  = lgd_df[lgd_df["year"] >= TRAIN_END_YEAR + 1].copy()

print("LGD train_full:", train_full.shape, "has_recovery rate:", train_full["has_recovery"].mean())
print("LGD test_full :", test_full.shape,  "has_recovery rate:", test_full["has_recovery"].mean())

train = stratified_sample_binary(train_full, "has_recovery", N_TRAIN_BAD, seed=SEED)

# =========================
# 1) 特征矩阵（严格用 feat_cols 顺序）
# =========================
X_train = train.loc[:, feat_cols].copy()
X_test  = test_full.loc[:, feat_cols].copy()

cat_cols = X_train.select_dtypes(include=["object","string"]).columns.tolist()
num_cols = [c for c in feat_cols if c not in cat_cols]

X_train = convert_cat_na(X_train, cat_cols)
X_test  = convert_cat_na(X_test, cat_cols)

# =========================
# 2) Stage1：是否有回收（has_recovery）
# =========================
y1_train = train["has_recovery"].astype(int).values
y1_test  = test_full["has_recovery"].astype(int).values

stage1 = Pipeline([
    ("prep", make_preprocess(num_cols, cat_cols)),
    ("clf", LogisticRegression(
        solver="saga",
        penalty="l2",
        C=0.5,
        max_iter=1000,
        tol=1e-3,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

stage1.fit(X_train, y1_train)
p_rec_test = stage1.predict_proba(X_test)[:, 1]

print("Stage1 AUC   =", roc_auc_score(y1_test, p_rec_test))
print("Stage1 PR-AUC=", average_precision_score(y1_test, p_rec_test))
print("Stage1 PR baseline ≈ has_recovery rate:", y1_test.mean())

# =========================
# 3) Stage2：回收率回归（logit 变换 + Ridge，稳定连续）
# =========================
train_pos = train[train["has_recovery"] == 1].copy()
test_pos  = test_full[test_full["has_recovery"] == 1].copy()

X2_train = convert_cat_na(train_pos.loc[:, feat_cols].copy(), cat_cols)
X2_test  = convert_cat_na(test_pos.loc[:, feat_cols].copy(), cat_cols)

y2_train = train_pos["recovery_rate"].astype(float).values
y2_test  = test_pos["recovery_rate"].astype(float).values

# logit(y) 作为回归目标
y2_train_clip = np.clip(y2_train, EPS2, 1 - EPS2)
t2_train = np.log(y2_train_clip / (1 - y2_train_clip))

stage2 = Pipeline([
    ("prep", make_preprocess(num_cols, cat_cols)),
    ("reg", Ridge(alpha=5.0))  # alpha 可试 1/5/10，数值越大越平滑
])

stage2.fit(X2_train, t2_train)

t2_pred = stage2.predict(X2_test)
rr_pos_test = sigmoid(t2_pred)

print("Stage2 MAE (recovered subset) =", mean_absolute_error(y2_test, rr_pos_test))
print("Stage2 rr_pos_test percentiles:", np.quantile(rr_pos_test, [0.01,0.05,0.5,0.95,0.99]))

# =========================
# 4) 在坏账 test 上合成 rr_hat / lgd_hat，并评估
# =========================
rr_pos_all_test = sigmoid(stage2.predict(X_test))
rr_hat_test = np.clip(p_rec_test * rr_pos_all_test, 0, 1)
lgd_hat_test = np.clip(1 - rr_hat_test, 0, 1)

print("Overall LGD MAE (bad loans test) =", mean_absolute_error(test_full["lgd"].values, lgd_hat_test))

# =========================
# 5) 给全量贷款打分 lgd_hat，并保存
# =========================
need_all_cols = unique_keep_order(["id","issue_d","loan_status"] + feat_cols)
all_df = pd.read_parquet(PD_ALL_PATH, columns=need_all_cols)

all_df = all_df[all_df["issue_d"].notna()].copy()
all_df["issue_d"] = to_datetime_safe(all_df["issue_d"])
all_df = all_df[all_df["issue_d"].notna()].copy()
all_df["year"] = all_df["issue_d"].dt.year

X_all = convert_cat_na(all_df.loc[:, feat_cols].copy(), cat_cols)

p_rec_all = stage1.predict_proba(X_all)[:, 1]
rr_pos_all = sigmoid(stage2.predict(X_all))

rr_hat_all = np.clip(p_rec_all * rr_pos_all, 0, 1)
lgd_hat_all = np.clip(1 - rr_hat_all, 0, 1)

out = all_df[["id","issue_d","year","loan_status","loan_amnt","funded_amnt"]].copy()
out["EAD"] = out["funded_amnt"].fillna(out["loan_amnt"])
out["p_recovery_hat"] = p_rec_all
out["recovery_rate_hat_pos"] = rr_pos_all
out["recovery_rate_hat"] = rr_hat_all
out["lgd_hat"] = lgd_hat_all

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(OUT_PATH, index=False)

print(f"[OK] saved -> {OUT_PATH}")
print(out.head())
print("recovery_rate_hat_pos share in {0,1} =", float(out["recovery_rate_hat_pos"].isin([0,1]).mean()))
print("lgd_hat percentiles:", out["lgd_hat"].quantile([0.01,0.05,0.5,0.95,0.99]).to_dict())
print("lgd_hat range:", float(out["lgd_hat"].min()), float(out["lgd_hat"].max()))
