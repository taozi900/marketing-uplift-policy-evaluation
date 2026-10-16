"""第一阶段：读取原始数据，做结构审计。直接运行本文件即可。"""
from pathlib import Path
import hashlib
import json
import pandas as pd

# 路径由脚本位置确定，在VS Code/PyCharm里不依赖当前工作目录。
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "raw" / "hillstrom.csv"
MANIFEST_FILE = PROJECT_DIR / "data" / "raw" / "source_manifest.json"
REPORT_DIR = PROJECT_DIR / "reports"

PRE_FEATURES = ["recency", "history", "history_segment", "mens", "womens",
                "zip_code", "newbie", "channel"]
OUTCOME_FIELDS = ["visit", "conversion", "spend"]
EXPECTED_FIELDS = PRE_FEATURES + ["segment"] + OUTCOME_FIELDS


def main():
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"没有找到数据，请保持起步包目录结构：{DATA_FILE}")
    original_bytes = DATA_FILE.read_bytes()
    file_hash = hashlib.sha256(original_bytes).hexdigest()
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if file_hash != manifest["csv_sha256"]:
        raise ValueError("原始CSV已变化。请保留原始数据，将清洗结果另存。")

    df = pd.read_csv(DATA_FILE)
    if set(df.columns) != set(EXPECTED_FIELDS) or df.columns.duplicated().any():
        raise ValueError("字段与预期不符，需要先查原因。")
    if len(df) != 64000:
        raise ValueError("记录数与已校验原始数据不符。")
    expected_groups = {"Mens E-Mail", "Womens E-Mail", "No E-Mail"}
    if set(df["segment"].unique()) != expected_groups:
        raise ValueError("实验分组标签与预期不符。")

    binary_fields = ["mens", "womens", "newbie", "visit", "conversion"]
    for column in binary_fields:
        if not df[column].isin([0, 1]).all():
            raise ValueError(f"{column}不全是0/1，需要审计。")

    audit = {
        "rows": len(df), "columns": len(df.columns), "csv_sha256": file_hash,
        "pandas_version": pd.__version__,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
        "segment_counts": {k: int(v) for k, v in df["segment"].value_counts().items()},
        "category_values": {c: sorted(df[c].unique().tolist())
                            for c in ["segment", "zip_code", "channel"]},
        "source_recency_range": [int(df.recency.min()), int(df.recency.max())],
        "duplicate_attribute_rows": int(df.duplicated().sum()),
        "outcome_logic_checks": {
            "conversion_1_visit_0": int(((df.conversion == 1) & (df.visit == 0)).sum()),
            "positive_spend_conversion_0": int(((df.spend > 0) & (df.conversion == 0)).sum()),
        },
        "notes": [
            "无真实客户ID，属性整行相同不等于同一个人；此阶段不去重。",
            "结构审计不是随机化有效性证明，也不是模型验证。",
            "此阶段不查看分层购买率、不选择高效果人群、不训练模型。",
        ],
    }
    REPORT_DIR.mkdir(exist_ok=True)
    report_path = REPORT_DIR / "01_data_audit.json"
    report_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"数据维度：{df.shape}，应为(64000, 12)")
    print("\n实验组人数：")
    print(df["segment"].value_counts().to_string())
    print(f"\n缺失值总数：{int(df.isna().sum().sum())}")
    print("\n字段类型：")
    print(df.dtypes.to_string())
    print(f"\n审计文件已保存：{report_path}")
    print("\n完成后，请用自己的话解释：一行是什么？T是什么？Y是什么？哪些字段不能当特征？")


if __name__ == "__main__":
    main()
