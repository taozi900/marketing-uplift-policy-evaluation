"""第二阶段：锁定客户级划分。先阅读analysis_plan.md，再运行。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
RAW_FILE = PROJECT_DIR / "data" / "raw" / "hillstrom.csv"
OUT_DIR = PROJECT_DIR / "data" / "processed"
SEED = 20261014


def preserve_or_write(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"已有不同的划分记录，先查原因，不覆盖：{path}")
    else:
        path.write_bytes(content)


def main():
    raw_bytes = RAW_FILE.read_bytes()
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    source_manifest = json.loads((RAW_FILE.parent / "source_manifest.json").read_text(encoding="utf-8"))
    if raw_hash != source_manifest["csv_sha256"]:
        raise ValueError("原始CSV校验不一致，不能生成划分。")
    raw = pd.read_csv(RAW_FILE)
    raw["source_row_proxy"] = np.arange(1, len(raw) + 1)
    # 只保留分配组与原始行号，不读取购买结果来决定谁进入哪个集合。
    data = raw.loc[raw.segment.isin(["Mens E-Mail", "No E-Mail"]),
                   ["source_row_proxy", "segment"]].copy()
    data["T"] = (data["segment"] == "Mens E-Mail").astype(int)
    if data["T"].nunique() != 2:
        raise ValueError("主分析必须同时包含处理组和对照组。")
    data["split"] = ""
    rng = np.random.default_rng(SEED)
    for treatment in [0, 1]:
        indices = data.index[data["T"] == treatment].to_numpy()
        shuffled = rng.permutation(indices)
        train_end = int(0.60 * len(shuffled))
        validation_end = train_end + int(0.20 * len(shuffled))
        data.loc[shuffled[:train_end], "split"] = "train"
        data.loc[shuffled[train_end:validation_end], "split"] = "validation"
        data.loc[shuffled[validation_end:], "split"] = "test"

    assignment = data[["source_row_proxy", "T", "split"]].sort_values("source_row_proxy")
    if assignment.source_row_proxy.duplicated().any() or (assignment.split == "").any():
        raise ValueError("存在重复或未分配客户。")
    counts = assignment.groupby(["split", "T"]).size().unstack(fill_value=0)
    encoded = assignment.to_csv(index=False, lineterminator="\n").encode("utf-8")
    record = {
        "seed": SEED, "contrast": "Mens E-Mail versus No E-Mail",
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "primary_outcome": "conversion", "primary_capacity": 0.20,
        "split_rule": "Within each T arm: floor(60%) train, floor(20%) validation, remainder test.",
        "stratification": "T only, no outcome-based selection",
        "rows": len(assignment), "source_csv_sha256": raw_hash,
        "split_csv_sha256": hashlib.sha256(encoded).hexdigest(),
        "counts": {split: {str(t): int(n) for t, n in row.items()}
                   for split, row in counts.to_dict(orient="index").items()},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preserve_or_write(OUT_DIR / "split_assignment.csv", encoded)
    preserve_or_write(OUT_DIR / "split_manifest.json",
                      json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"))
    print(f"主分析记录数：{len(assignment)}，应为42613")
    print("\n锁定划分人数（列为T）：")
    print(counts.to_string())
    print("\n后续按source_row_proxy关联这份划分，不重新随机拆分。")
    print("训练集用于探索与拟合；验证集用于选方案；测试集保留到最终评价。")
    print("本文件只生成划分记录，没有训练模型或报告效果。")


if __name__ == "__main__":
    main()
