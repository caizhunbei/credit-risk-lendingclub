import pandas as pd
from pathlib import Path
from datetime import datetime

REPORTS_DIR = Path("reports")

GRADE_CSV = REPORTS_DIR / "el_by_grade.csv"
TERM_CSV = REPORTS_DIR / "el_by_term.csv"
PURPOSE_CSV = REPORTS_DIR / "el_by_purpose.csv"
TOP_CSV = REPORTS_DIR / "top_risk_loans.csv"

OUT_MD = REPORTS_DIR / "summary.md"
OUT_SNIPPET = REPORTS_DIR / "README_snippet.md"

TOPN = 8  # 每张表输出前 N 行

def fmt_money(x):
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)

def fmt_pct(x, digits=2):
    try:
        return f"{100*float(x):.{digits}f}%"
    except Exception:
        return str(x)

def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def top_rows_as_md(df: pd.DataFrame, cols: list[str], n: int = TOPN) -> str:
    use = [c for c in cols if c in df.columns]
    out = df.loc[: , use].head(n).copy()
    return out.to_markdown(index=False)

def describe_key_findings(df: pd.DataFrame, key_col: str) -> dict:
    """
    根据 sum_EL / EL_per_EAD 给出最关键的两条结论：
    1) 总EL最大的组
    2) 风险强度(EL_per_EAD)最大的组
    """
    out = {}
    # 总EL最大
    if "sum_EL" in df.columns:
        r = df.sort_values("sum_EL", ascending=False).iloc[0]
        out["top_sum"] = (r[key_col], r["sum_EL"], r.get("n", None), r.get("EL_per_EAD", None))
    # 风险强度最大
    if "EL_per_EAD" in df.columns:
        r = df.sort_values("EL_per_EAD", ascending=False).iloc[0]
        out["top_intensity"] = (r[key_col], r.get("EL_per_EAD", None), r.get("sum_EL", None), r.get("n", None))
    return out

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    grade = safe_read_csv(GRADE_CSV)
    term = safe_read_csv(TERM_CSV)
    purpose = safe_read_csv(PURPOSE_CSV)
    top = safe_read_csv(TOP_CSV)

    # ===== 关键结论 =====
    gk = describe_key_findings(grade, "grade")
    tk = describe_key_findings(term, "term")
    pk = describe_key_findings(purpose, "purpose")

    # ===== Top 风险清单（展示列尽量通用）=====
    top_cols_pref = ["id", "issue_d", "year", "loan_status", "EAD", "pd_hat", "lgd_hat", "EL_rate", "EL_amount", "grade", "term", "purpose"]
    top_cols = [c for c in top_cols_pref if c in top.columns]
    top_show = top.loc[:, top_cols].head(20).copy()

    # 格式化展示（不影响原 CSV）
    for c in ["EAD", "EL_amount"]:
        if c in top_show.columns:
            top_show[c] = top_show[c].map(fmt_money)
    for c in ["pd_hat", "lgd_hat", "EL_rate"]:
        if c in top_show.columns:
            top_show[c] = top_show[c].map(lambda x: fmt_pct(x, 2))

    # ===== 总体信息（从 top_risk_loans 读不到全量统计，这里用分组表推一个近似总EL）=====
    total_el = None
    if "sum_EL" in grade.columns:
        total_el = grade["sum_EL"].sum()

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ===== summary.md 内容 =====
    md = []
    md.append(f"# LendingClub Credit Risk (PD/LGD/EL) — Summary\n")
    md.append(f"_Generated: {now}_\n")

    md.append("## Project Overview\n")
    md.append("- Dataset: LendingClub loan-level data (Parquet shards merged into modeling tables)\n")
    md.append("- Goal: Build an end-to-end credit risk pipeline: **PD → LGD → EL**\n")
    md.append("- Modeling:\n")
    md.append("  - **PD**: logistic regression (time-based split: train ≤2016, test ≥2017)\n")
    md.append("  - **LGD**: two-stage approach\n")
    md.append("    - Stage 1: P(recovery > 0)\n")
    md.append("    - Stage 2: recovery rate | recovery>0 (logit transform + Ridge)\n")
    md.append("  - **EL**: `EL_amount = PD_hat × LGD_hat × EAD` ; `EL_rate = PD_hat × LGD_hat`\n")

    md.append("## Key Findings (from reports)\n")
    if total_el is not None:
        md.append(f"- Estimated total expected loss (sum over loans): **{fmt_money(total_el)}**\n")

    def bullet_from_findings(tag, findings):
        out = []
        if "top_sum" in findings:
            k, sum_el, n, el_per_ead = findings["top_sum"]
            out.append(f"- **{tag}** — largest total EL: `{k}` (sum_EL={fmt_money(sum_el)})")
        if "top_intensity" in findings:
            k, el_per_ead, sum_el, n = findings["top_intensity"]
            out.append(f"- **{tag}** — highest EL intensity (EL_per_EAD): `{k}` (EL_per_EAD={fmt_pct(el_per_ead,2)})")
        return out

    md += bullet_from_findings("Grade", gk)
    md += bullet_from_findings("Term", tk)
    md += bullet_from_findings("Purpose", pk)
    md.append("")

    md.append("## Breakdown Tables (Top rows)\n")

    md.append("### EL by Grade (Top)\n")
    md.append(top_rows_as_md(
        grade,
        cols=["grade","n","bad_rate","mean_PD","mean_LGD","mean_EL_rate","sum_EL","EL_per_EAD"],
        n=TOPN
    ))
    md.append("\n")

    md.append("### EL by Term (Top)\n")
    md.append(top_rows_as_md(
        term,
        cols=["term","n","bad_rate","mean_PD","mean_LGD","mean_EL_rate","sum_EL","EL_per_EAD"],
        n=TOPN
    ))
    md.append("\n")

    md.append("### EL by Purpose (Top)\n")
    md.append(top_rows_as_md(
        purpose,
        cols=["purpose","n","bad_rate","mean_PD","mean_LGD","mean_EL_rate","sum_EL","EL_per_EAD"],
        n=TOPN
    ))
    md.append("\n")

    md.append("## Top Risk Loans (Top 20 by EL_amount)\n")
    md.append(top_show.to_markdown(index=False))
    md.append("\n")

    md.append("## Reproducibility\n")
    md.append("- Key outputs:\n")
    md.append("  - `reports/el_scored.parquet`\n")
    md.append("  - `reports/top_risk_loans.csv`\n")
    md.append("  - `reports/el_by_grade.csv`, `reports/el_by_term.csv`, `reports/el_by_purpose.csv`\n")
    md.append("- Run order:\n")
    md.append("  1. Train/score PD → `pd_scored.parquet`\n")
    md.append("  2. Train/score LGD → `lgd_scored.parquet`\n")
    md.append("  3. Merge to EL + reports → `el_scored.parquet` + CSVs\n")
    md.append("  4. Generate this summary → `summary.md`\n")

    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] wrote -> {OUT_MD}")

    # ===== README_snippet.md：给你一个更适合放 GitHub README 的结构化片段 =====
    snippet = []
    snippet.append("# LendingClub Credit Risk Modeling (PD/LGD/EL)\n")
    snippet.append("End-to-end credit risk pipeline using LendingClub loan-level data.\n")
    snippet.append("## Highlights\n")
    if "top_sum" in gk:
        snippet.append(f"- Largest EL by grade: **{gk['top_sum'][0]}** (sum_EL ≈ {fmt_money(gk['top_sum'][1])})")
    if "top_intensity" in pk:
        snippet.append(f"- Highest EL intensity by purpose: **{pk['top_intensity'][0]}** (EL_per_EAD ≈ {fmt_pct(pk['top_intensity'][1],2)})")
    if total_el is not None:
        snippet.append(f"- Total expected loss (approx): **{fmt_money(total_el)}**\n")

    snippet.append("## Method\n")
    snippet.append("- **PD**: Logistic Regression (time split: train ≤2016, test ≥2017)\n")
    snippet.append("- **LGD**: Two-stage\n")
    snippet.append("  1) P(recovery>0)\n")
    snippet.append("  2) recovery_rate | recovery>0 (logit transform + Ridge)\n")
    snippet.append("- **EL**: `PD_hat × LGD_hat × EAD`\n")

    snippet.append("## Outputs\n")
    snippet.append("- `reports/summary.md` (auto-generated)\n")
    snippet.append("- `reports/el_scored.parquet`\n")
    snippet.append("- `reports/top_risk_loans.csv`\n")
    snippet.append("- `reports/el_by_grade.csv`, `reports/el_by_term.csv`, `reports/el_by_purpose.csv`\n")

    snippet.append("## How to run\n")
    snippet.append("```bash\npython scripts/05_el_merge_and_report.py\npython scripts/06_el_add_dims_and_group_reports.py\npython scripts/07_make_readme_summary.py\n```\n")

    OUT_SNIPPET.write_text("\n".join(snippet), encoding="utf-8")
    print(f"[OK] wrote -> {OUT_SNIPPET}")

if __name__ == "__main__":
    main()
