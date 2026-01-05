from pathlib import Path
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.parquet as pq

PARTS_DIR = Path("data_raw/processed")   # 你现在分片就在这
OUT_DIR = Path("data_raw/model")         # 你已经有这个文件夹
OUT_DIR.mkdir(parents=True, exist_ok=True)

dataset = ds.dataset(str(PARTS_DIR), format="parquet")
cols_all = set(dataset.schema.names)

good_status = ["Fully Paid"]
bad_status  = ["Charged Off", "Default"]

# ========= PD：只保留（Fully Paid vs Charged Off/Default），并生成 y_bad =========
pd_cols = [
    "id", "issue_d",
    "loan_amnt", "funded_amnt", "term", "int_rate",
    "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "purpose", "addr_state", "dti",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "fico_range_low", "fico_range_high",
    "loan_status",
]
pd_cols = [c for c in pd_cols if c in cols_all]

pd_filter = ds.field("loan_status").isin(good_status + bad_status)
pd_scanner = dataset.scanner(columns=pd_cols, filter=pd_filter, batch_size=200_000)

pd_out = OUT_DIR / "pd_model.parquet"
pd_writer = None
pd_rows = 0

for batch in pd_scanner.to_batches():
    tbl = pa.Table.from_batches([batch])
    y_bad = pc.is_in(tbl["loan_status"], value_set=pa.array(bad_status))
    tbl = tbl.append_column("y_bad", pc.cast(y_bad, pa.int8()))
    if pd_writer is None:
        pd_writer = pq.ParquetWriter(str(pd_out), tbl.schema, compression="snappy")
    pd_writer.write_table(tbl)
    pd_rows += tbl.num_rows

if pd_writer:
    pd_writer.close()

print(f"[OK] PD saved -> {pd_out}  rows={pd_rows}")

# ========= LGD：只在坏账样本上，计算 recovery_rate / lgd / has_recovery =========
lgd_cols = pd_cols + ["recoveries", "collection_recovery_fee"]
lgd_cols = [c for c in lgd_cols if c in cols_all]

lgd_filter = ds.field("loan_status").isin(bad_status)
lgd_scanner = dataset.scanner(columns=lgd_cols, filter=lgd_filter, batch_size=200_000)

lgd_out = OUT_DIR / "lgd_model.parquet"
lgd_writer = None
lgd_rows = 0

for batch in lgd_scanner.to_batches():
    tbl = pa.Table.from_batches([batch])

    # EAD ≈ funded_amnt（空则用 loan_amnt）
    funded = tbl["funded_amnt"] if "funded_amnt" in tbl.column_names else None
    loan   = tbl["loan_amnt"] if "loan_amnt" in tbl.column_names else None
    ead = pc.if_else(pc.is_null(funded), loan, funded) if funded is not None else loan
    # 避免除0/负数
    ead_safe = pc.if_else(pc.less_equal(ead, 0), pa.scalar(1.0), ead)

    recoveries = tbl["recoveries"] if "recoveries" in tbl.column_names else pa.array([0.0] * tbl.num_rows)
    recoveries0 = pc.fill_null(recoveries, 0)

    rr = pc.divide(recoveries0, ead_safe)
    rr = pc.min_element_wise(pc.max_element_wise(rr, 0), 1)  # clip到[0,1]
    lgd = pc.subtract(1, rr)
    has_recovery = pc.cast(pc.greater(recoveries0, 0), pa.int8())

    tbl = tbl.append_column("EAD", pc.cast(ead_safe, pa.float64()))
    tbl = tbl.append_column("recovery_rate", pc.cast(rr, pa.float64()))
    tbl = tbl.append_column("lgd", pc.cast(lgd, pa.float64()))
    tbl = tbl.append_column("has_recovery", has_recovery)

    if lgd_writer is None:
        lgd_writer = pq.ParquetWriter(str(lgd_out), tbl.schema, compression="snappy")
    lgd_writer.write_table(tbl)
    lgd_rows += tbl.num_rows

if lgd_writer:
    lgd_writer.close()

print(f"[OK] LGD saved -> {lgd_out}  rows={lgd_rows}")
