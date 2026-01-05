import pandas as pd
from pathlib import Path

PD_SCORED = "data_raw/model/pd_scored.parquet"
LGD_SCORED = "data_raw/model/lgd_scored.parquet"

OUT_DIR = Path("reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PARQUET = OUT_DIR / "el_scored.parquet"
OUT_TOP = OUT_DIR / "top_risk_loans.csv"
OUT_GRADE = OUT_DIR / "el_by_grade.csv"
OUT_TERM = OUT_DIR / "el_by_term.csv"
OUT_PURPOSE = OUT_DIR / "el_by_purpose.csv"

# ========= 1) 读取 =========
pd_df = pd.read_parquet(PD_SCORED)
lgd_df = pd.read_parquet(LGD_SCORED)

# 只保留需要的列，避免重复
pd_keep = ["id","issue_d","year","y_bad","loan_status","loan_amnt","funded_amnt","EAD","pd_hat"]
lgd_keep = ["id","lgd_hat","p_recovery_hat","recovery_rate_hat_pos","recovery_rate_hat"]

pd_df = pd_df[pd_keep].copy()
lgd_df = lgd_df[lgd_keep].copy()

# ========= 2) 合并 =========
df = pd_df.merge(lgd_df, on="id", how="inner")

# ========= 3) 计算 EL =========
df["EL_rate"] = df["pd_hat"] * df["lgd_hat"]
df["EL_amount"] = df["EL_rate"] * df["EAD"]

# 简单 sanity check
print("merged rows:", df.shape)
print("pd_hat range:", df["pd_hat"].min(), df["pd_hat"].max())
print("lgd_hat range:", df["lgd_hat"].min(), df["lgd_hat"].max())
print("EL_amount describe:\n", df["EL_amount"].describe(percentiles=[.01,.05,.5,.95,.99]))

# ========= 4) 保存全量打分 =========
df.to_parquet(OUT_PARQUET, index=False)
print(f"[OK] saved -> {OUT_PARQUET}")

# ========= 5) Top 风险清单 =========
top = df.sort_values("EL_amount", ascending=False).head(1000).copy()
top_cols = ["id","issue_d","year","loan_status","EAD","pd_hat","lgd_hat","EL_rate","EL_amount",
            "grade" if "grade" in df.columns else None,
            "term" if "term" in df.columns else None,
            "purpose" if "purpose" in df.columns else None]
top_cols = [c for c in top_cols if c is not None]
top[top_cols].to_csv(OUT_TOP, index=False, encoding="utf-8-sig")
print(f"[OK] saved -> {OUT_TOP}")

# ========= 6) 分组报表函数 =========
def group_report(d: pd.DataFrame, key: str) -> pd.DataFrame:
    g = d.groupby(key, dropna=False).agg(
        n=("id","count"),
        bad_rate=("y_bad","mean"),
        mean_EAD=("EAD","mean"),
        mean_PD=("pd_hat","mean"),
        mean_LGD=("lgd_hat","mean"),
        mean_EL_rate=("EL_rate","mean"),
        sum_EL=("EL_amount","sum"),
        sum_EAD=("EAD","sum"),
    ).reset_index()
    g["EL_per_EAD"] = g["sum_EL"] / g["sum_EAD"]
    g = g.sort_values("sum_EL", ascending=False)
    return g

# 注意：pd_scored 里未必带 grade/term/purpose（取决于你最初保存的列）
# 如果缺少，我们从 pd_model.parquet 再补一次也行。这里先“有就报表，没有就跳过”。
for key, out_path in [("grade", OUT_GRADE), ("term", OUT_TERM), ("purpose", OUT_PURPOSE)]:
    if key in df.columns:
        rep = group_report(df, key)
        rep.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[OK] saved -> {out_path}")
    else:
        print(f"[WARN] column '{key}' not found in el_scored. Skip {out_path.name}.")
