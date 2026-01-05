import pandas as pd
from pathlib import Path

EL_PATH = "reports/el_scored.parquet"
PD_MODEL_PATH = "data_raw/model/pd_model.parquet"

OUT_DIR = Path("reports")
OUT_GRADE = OUT_DIR / "el_by_grade.csv"
OUT_TERM = OUT_DIR / "el_by_term.csv"
OUT_PURPOSE = OUT_DIR / "el_by_purpose.csv"

# 1) 读 EL
el = pd.read_parquet(EL_PATH)

# 2) 从 pd_model 补维度字段（只读需要列，速度快）
dims = pd.read_parquet(PD_MODEL_PATH, columns=["id", "grade", "term", "purpose"])
# 防止 id 重复导致 merge 放大（理论上不该，但加一道保险）
dims = dims.drop_duplicates(subset=["id"])

# 3) merge 回去
df = el.merge(dims, on="id", how="left")

print("rows:", df.shape)
print("missing grade:", df["grade"].isna().mean())
print("missing term :", df["term"].isna().mean())
print("missing purpose:", df["purpose"].isna().mean())

# 4) 分组报表
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

rep_grade = group_report(df, "grade")
rep_term = group_report(df, "term")
rep_purpose = group_report(df, "purpose")

rep_grade.to_csv(OUT_GRADE, index=False, encoding="utf-8-sig")
rep_term.to_csv(OUT_TERM, index=False, encoding="utf-8-sig")
rep_purpose.to_csv(OUT_PURPOSE, index=False, encoding="utf-8-sig")

print(f"[OK] saved -> {OUT_GRADE}")
print(f"[OK] saved -> {OUT_TERM}")
print(f"[OK] saved -> {OUT_PURPOSE}")
