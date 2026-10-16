"""第四阶段：检查冻结训练集两组客户的实验前特征平衡。

主问题：发男装邮件和不发邮件的客户，在营销之前是否有明显差异？
只读取实验前字段和实验分组，不读取visit、conversion、spend。
|SMD|=0.1是描述性经验参考线，不是随机化有效性的证明。
直接运行前，请把本文件放到D盘的marketing_project_starter根目录。
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

import numpy as np
import pandas as pd


PRE_FEATURES = [
    "recency", "history", "history_segment", "mens", "womens",
    "zip_code", "newbie", "channel",
]
REFERENCE = 0.1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_training_covariates(project_dir):
    """按01/02阶段的来源校验和原始行号提取训练集。"""
    raw_file = project_dir / "data" / "raw" / "hillstrom.csv"
    split_file = project_dir / "data" / "processed" / "split_assignment.csv"
    split_manifest = split_file.parent / "split_manifest.json"
    for path in (raw_file, split_file, split_manifest):
        require(path.is_file(), f"缺少文件：{path}；先完成01、02阶段。")
    source = json.loads((raw_file.parent / "source_manifest.json").read_text(encoding="utf-8-sig"))
    manifest = json.loads(split_manifest.read_text(encoding="utf-8-sig"))
    raw_hash, split_hash = digest(raw_file), digest(split_file)
    require(raw_hash == source["csv_sha256"] == manifest["source_csv_sha256"], "原始数据校验不一致。")
    require(split_hash == manifest["split_csv_sha256"], "划分校验不一致；不要重新拆分。")
    require(manifest["contrast"] == "Mens E-Mail versus No E-Mail", "主比较发生变化。")
    require(len(raw_file.read_bytes().splitlines()) == 64001, "源CSV物理行数不符合行号约定。")

    assignment = pd.read_csv(split_file)
    require(list(assignment.columns) == ["source_row_proxy", "T", "split"], "划分字段异常。")
    require(not assignment.isna().any().any(), "划分中存在缺失。")
    require(assignment["source_row_proxy"].is_unique, "原始行号重复。")
    require(len(assignment) == manifest["rows"] == 42613, "主分析人数异常。")
    require(set(assignment["T"].unique()) == {0, 1}, "T编码异常。")
    require(set(assignment["split"].unique()) == {"train", "validation", "test"}, "集合标签异常。")
    counts = assignment.groupby(["split", "T"]).size()
    for split, arms in manifest["counts"].items():
        for t, n in arms.items():
            require(int(counts.loc[(split, int(t))]) == int(n), "人数与划分记录不一致。")

    # 全体只核对随机分配组，不解析任何活动后结果字段。
    groups = pd.read_csv(raw_file, usecols=["segment"])
    require(len(groups) == 64000, "原始数据行数异常。")
    ids = assignment["source_row_proxy"].to_numpy()
    require(np.issubdtype(ids.dtype, np.integer), "原始行号应为整数。")
    require(bool(((ids >= 1) & (ids <= len(groups))).all()), "原始行号越界。")
    expected_ids = np.flatnonzero(groups["segment"].isin(["Mens E-Mail", "No E-Mail"])) + 1
    require(np.array_equal(np.sort(ids), expected_ids), "主比较的原始行覆盖范围异常。")
    expected_t = (groups.iloc[ids - 1]["segment"].to_numpy() == "Mens E-Mail").astype(int)
    require(np.array_equal(expected_t, assignment["T"].to_numpy()), "T与原始分组不对应。")

    train_assignment = assignment.loc[assignment["split"] == "train"].sort_values("source_row_proxy")
    train_ids = set(train_assignment["source_row_proxy"].astype(int))
    training = pd.read_csv(
        raw_file, usecols=PRE_FEATURES + ["segment"],
        skiprows=lambda row_number: row_number != 0 and row_number not in train_ids,
    )
    require(len(training) == len(train_assignment) == 25567, "训练集提取人数异常。")
    training["T"] = train_assignment["T"].to_numpy()
    training["source_row_proxy"] = train_assignment["source_row_proxy"].to_numpy()
    require(np.array_equal((training["segment"] == "Mens E-Mail").astype(int), training["T"]),
            "训练行顺序与T不一致。")
    require(not training[PRE_FEATURES].isna().any().any(), "训练集实验前特征存在缺失。")
    for column in ["recency", "history", "mens", "womens", "newbie"]:
        training[column] = pd.to_numeric(training[column], errors="raise")
        require(np.isfinite(training[column].to_numpy()).all(), f"{column}存在无穷值。")
    for column in ["mens", "womens", "newbie"]:
        require(training[column].isin([0, 1]).all(), f"{column}不是0/1。")
    require(training["history"].ge(0).all(), "history不能为负数。")
    return training, {"source_csv_sha256": raw_hash, "split_csv_sha256": split_hash}


def smd_record(name, kind, values0, values1, level=None):
    """连续变量用样本方差；0/1指示变量用p(1-p)。"""
    a, b = np.asarray(values0, dtype=float), np.asarray(values1, dtype=float)
    require(len(a) >= 2 and len(b) >= 2, "每组至少需要2个观测。")
    require(np.isfinite(a).all() and np.isfinite(b).all(), "SMD输入含缺失或无穷值。")
    mean0, mean1 = float(a.mean()), float(b.mean())
    if kind == "continuous":
        var0, var1 = float(a.var(ddof=1)), float(b.var(ddof=1))
    else:
        require(np.isin(a, [0, 1]).all() and np.isin(b, [0, 1]).all(), "指示变量只能取0/1。")
        var0, var1 = mean0 * (1 - mean0), mean1 * (1 - mean1)
    scale = math.sqrt((var0 + var1) / 2)
    if scale == 0:
        if mean0 == mean1:
            # 明确标记同常量；将SMD按无组间差异的惯例记为0。
            smd, status = 0.0, "constant_same"
        else:
            # 两组常量不同：SMD无有限定义，必须单独提示。
            smd, status = None, "zero_variance_different"
    else:
        smd, status = (mean1 - mean0) / scale, "computed"
    needs_review = status == "zero_variance_different" or (smd is not None and abs(smd) >= REFERENCE)
    return {
        "feature": name, "type": kind, "level": level,
        "mean_or_proportion_T0": mean0, "mean_or_proportion_T1": mean1,
        "sd_T0": math.sqrt(var0), "sd_T1": math.sqrt(var1),
        "smd": smd, "abs_smd": None if smd is None else abs(smd),
        "status": status, "needs_review": bool(needs_review),
    }


def balance_table(training):
    rows, distribution = [], {}
    # log1p是事先固定的变换，用于辅助观察消费金额长尾；不根据购买结果选变换。
    training = training.copy()
    training["log1p_history"] = np.log1p(training["history"])
    for column in ["recency", "history", "log1p_history"]:
        arms = [training.loc[training["T"] == t, column].to_numpy() for t in [0, 1]]
        rows.append(smd_record(column, "continuous", *arms))
        distribution[column] = {}
        for t, values in enumerate(arms):
            distribution[column][f"T{t}"] = {
                "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)),
                "q25": float(np.quantile(values, 0.25)), "median": float(np.median(values)),
                "q75": float(np.quantile(values, 0.75)), "q95": float(np.quantile(values, 0.95)),
            }
    for column in ["mens", "womens", "newbie"]:
        arms = [training.loc[training["T"] == t, column].to_numpy() for t in [0, 1]]
        rows.append(smd_record(column, "binary", *arms))
    for column in ["zip_code", "channel", "history_segment"]:
        for level in sorted(training[column].unique()):
            arms = [training.loc[training["T"] == t, column].eq(level).astype(int).to_numpy() for t in [0, 1]]
            rows.append(smd_record(column, "category_indicator", *arms, level=str(level)))
    return pd.DataFrame(rows), distribution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else project_dir / "reports"
    training, provenance = load_training_covariates(project_dir)
    table, distribution = balance_table(training)
    table = table.sort_values(["needs_review", "abs_smd"], ascending=[False, False], na_position="first")
    flagged = int(table["needs_review"].sum())
    calculated = table["abs_smd"].dropna()
    max_smd = float(calculated.max()) if len(calculated) else None
    report = {
        "analysis_set": "train", "train_rows": len(training),
        "arm_counts": {str(t): int((training["T"] == t).sum()) for t in [0, 1]},
        "features_checked": PRE_FEATURES + ["log1p_history (fixed auxiliary transform)"],
        "reference_abs_smd": REFERENCE, "diagnostic_rows": len(table),
        "rows_needing_review": flagged, "max_finite_abs_smd": max_smd,
        "continuous_distribution_summary": distribution,
        "source_reference": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3472075/",
        "notes": [
            "T1为男装邮件、T0为不发邮件；SMD的方向始终为T1减T0。",
            "连续变量分母使用两组ddof=1样本方差的平均开方。",
            "0/1及分类指示变量分母使用两组p(1-p)的平均开方。",
            "0.1只是描述性参考线；不做基线显著性筛选，不据此判定随机化成功。",
            "SMD诊断均值/比例差异；不能排除尾部、联合分布或未观测特征差异。",
            "原数据无客户ID，原始行号不是身份验证；没有按属性相同去重。",
            "history及history_segment同时用于诊断；后续模型不必机械地同时使用。",
            "mens/womens是历史购买品类标记，不是客户性别。",
            "同常量列单独标记并按惯例记SMD=0；不同常量列SMD留空且明确需核查。",
            "本脚本没有读取visit、conversion、spend，不更改数据或重新拆分。",
        ],
        "runtime": {"executable": sys.executable, "python": sys.version.split()[0],
                    "numpy": np.__version__, "pandas": pd.__version__},
        **provenance,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "04_train_balance.csv", index=False, encoding="utf-8-sig")
    (output_dir / "04_train_balance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"训练集：{len(training):,}人；T0={(training['T'] == 0).sum():,}，T1={(training['T'] == 1).sum():,}")
    print("只分析实验前特征；没有读取活动后结果。")
    print("\n按绝对SMD从大到小显示前10项，完整结果保存在CSV：")
    display = table.head(10).copy()
    display["item"] = display.apply(
        lambda r: r["feature"] if pd.isna(r["level"]) else f"{r['feature']}={r['level']}", axis=1
    )
    print("T0/T1列：数值特征显示均值，0/1或类别指示显示比例。")
    display = display.rename(columns={"mean_or_proportion_T0": "T0", "mean_or_proportion_T1": "T1",
                                      "smd": "SMD", "needs_review": "check"})
    print(display[["item", "T0", "T1", "SMD", "check"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))
    if max_smd is not None:
        print(f"\n可计算项中的最大绝对SMD：{max_smd:.4f}")
    print(f"超过0.1参考线或有不同常量异常的项目数：{flagged}")
    if flagged:
        print("请核查标记项并如实记录；保持原始客户和冻结划分，不能为了好看重拆或删人。")
    else:
        print("本次所检查的均值/比例差异均低于0.1参考线；仍需保留诊断局限。")
    print("这不是随机化有效或不存在混杂的证明。")
    print(f"\n结果保存到：{output_dir}")
    print("请先找出：差异最大的特征是什么？它的绝对SMD是否超过0.1？")


if __name__ == "__main__":
    main()
