"""第三阶段：按冻结划分分析训练集的总体邮件效果。

先读懂三件事：购买率、购买率差（百分点）、差异的95%置信区间。
直接运行本文件；正式使用时请把它放到 marketing_project_starter 根目录。
结果是训练阶段的初步分析，后续仍需独立评价。
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.stats import fisher_exact
import statsmodels
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportions_ztest,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_training_data(project_dir):
    """核验原始数据与划分；仅载入训练行的结果变量。"""
    raw_file = project_dir / "data" / "raw" / "hillstrom.csv"
    assignment_file = project_dir / "data" / "processed" / "split_assignment.csv"
    manifest_file = assignment_file.parent / "split_manifest.json"
    for path in (raw_file, assignment_file, manifest_file):
        require(path.is_file(), f"缺少文件：{path}；先完成01、02阶段。")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
    source = json.loads((raw_file.parent / "source_manifest.json").read_text(encoding="utf-8-sig"))
    raw_hash, assignment_hash = sha256(raw_file), sha256(assignment_file)
    require(raw_hash == source["csv_sha256"] == manifest["source_csv_sha256"],
            "原始数据校验不一致，停止分析。")
    require(assignment_hash == manifest["split_csv_sha256"],
            "划分文件校验不一致，停止分析；不要重新随机划分。")
    require(manifest["primary_outcome"] == "conversion", "主结果变量发生变化。")
    require(manifest["contrast"] == "Mens E-Mail versus No E-Mail", "主比较发生变化。")

    assignment = pd.read_csv(assignment_file)
    require(list(assignment.columns) == ["source_row_proxy", "T", "split"], "划分字段不符合约定。")
    require(not assignment.isna().any().any(), "划分中存在缺失值。")
    require(assignment["source_row_proxy"].is_unique, "同一个原始行被分配多次。")
    require(set(assignment["T"].unique()) == {0, 1}, "处理组编码不是0/1。")
    require(set(assignment["split"].unique()) == {"train", "validation", "test"}, "集合标签异常。")
    require(len(assignment) == manifest["rows"] == 42613, "主分析记录数异常。")
    counts = assignment.groupby(["split", "T"]).size()
    for split, arms in manifest["counts"].items():
        for treatment, n in arms.items():
            require(int(counts.loc[(split, int(treatment))]) == int(n), "分组人数与划分记录不一致。")

    # 全体数据这里只核对实验分配，不读取验证/测试集的购买结果。
    all_groups = pd.read_csv(raw_file, usecols=["segment"])
    require(len(all_groups) == 64000, "原始数据行数异常。")
    ids = assignment["source_row_proxy"].to_numpy()
    require(np.issubdtype(ids.dtype, np.integer), "原始行号必须为整数。")
    require(bool(((ids >= 1) & (ids <= len(all_groups))).all()), "原始行号越界。")
    primary_ids = np.flatnonzero(all_groups["segment"].isin(["Mens E-Mail", "No E-Mail"])) + 1
    require(np.array_equal(np.sort(ids), primary_ids), "主比较客户覆盖范围不一致。")
    original_groups = all_groups.iloc[ids - 1]["segment"].to_numpy()
    expected_t = (original_groups == "Mens E-Mail").astype(int)
    require(np.array_equal(expected_t, assignment["T"].to_numpy()), "T与原始邮件组不一致。")

    train_assignment = assignment.loc[assignment["split"] == "train"].sort_values("source_row_proxy")
    training_ids = set(train_assignment["source_row_proxy"].astype(int))
    # 这个已校验的CSV每条客户记录占一行：0为表头，1开始是客户原始行号。
    # skiprows保留训练行，因此验证/测试行的conversion不进入分析表。
    training = pd.read_csv(
        raw_file,
        usecols=["segment", "conversion"],
        skiprows=lambda row_number: row_number != 0 and row_number not in training_ids,
    )
    require(len(training) == len(train_assignment), "训练行提取数量不一致。")
    training["source_row_proxy"] = train_assignment["source_row_proxy"].to_numpy()
    training["T"] = train_assignment["T"].to_numpy()
    require(np.array_equal((training["segment"] == "Mens E-Mail").astype(int), training["T"]),
            "训练行关联顺序异常。")
    require(training["conversion"].isin([0, 1]).all(), "训练集购买结果不是0/1。")
    return training, {"source_csv_sha256": raw_hash, "split_csv_sha256": assignment_hash}


def estimate_effect(training):
    """T=1男装邮件，T=0不发邮件；Y为两周内是否购买。"""
    rows = []
    for treatment, label in [(0, "No E-Mail"), (1, "Mens E-Mail")]:
        y = training.loc[training["T"] == treatment, "conversion"]
        n, buyers = len(y), int(y.sum())
        require(n > 0, "训练集缺少处理组或对照组。")
        low, high = proportion_confint(buyers, n, alpha=0.05, method="wilson")
        rows.append({
            "T": treatment, "group": label, "customers": n, "buyers": buyers,
            "conversion_rate": buyers / n,
            "rate_ci_low": float(low), "rate_ci_high": float(high),
        })
    summary = pd.DataFrame(rows)
    c, t = rows
    p0, p1 = c["conversion_rate"], t["conversion_rate"]
    difference = p1 - p0
    low, high = confint_proportions_2indep(
        t["buyers"], t["customers"], c["buyers"], c["customers"],
        compare="diff", method="newcomb", alpha=0.05,
    )

    pooled = (c["buyers"] + t["buyers"]) / (c["customers"] + t["customers"])
    expected = [n * p for n in [c["customers"], t["customers"]] for p in [pooled, 1 - pooled]]
    if min(expected) >= 5:
        _, p_value = proportions_ztest(
            [t["buyers"], c["buyers"]], [t["customers"], c["customers"]],
            value=0, alternative="two-sided",
        )
        test_name = "two-sided pooled two-proportion z-test"
    else:
        _, p_value = fisher_exact([
            [t["buyers"], t["customers"] - t["buyers"]],
            [c["buyers"], c["customers"] - c["buyers"]],
        ], alternative="two-sided")
        test_name = "two-sided Fisher exact test (small expected cell count)"

    effect = {
        "analysis_set": "train", "train_rows": len(training),
        "outcome": "14-day conversion", "contrast": "Mens E-Mail minus No E-Mail",
        "risk_difference": difference, "risk_difference_pp": 100 * difference,
        "difference_ci_low": float(low), "difference_ci_high": float(high),
        "difference_ci_low_pp": 100 * float(low), "difference_ci_high_pp": 100 * float(high),
        "relative_lift_point_estimate": None if p0 == 0 else difference / p0,
        "p_value": float(p_value), "test": test_name,
        "rate_interval": "Wilson 95%", "difference_interval": "Newcombe 95%",
        "notes": [
            "这是预先指定主比较在训练集上的初步分析，不是独立测试集结论。",
            "因果解释依赖原始随机分配描述；结构审计本身不能证明随机化有效。",
            "区间和检验将每行视为独立客户；无客户ID，无法完全核验重复客户问题。",
            "没有按字段相同去重，也没有查看验证集或测试集的购买率。",
            "主指标为购买率差；百分点与相对提升率不是同一个单位。",
            "不据此承诺线上提升、利润或个人客户的真实增量效果。",
        ],
    }
    return summary, effect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else project_dir / "reports"
    training, provenance = load_training_data(project_dir)
    summary, effect = estimate_effect(training)
    effect.update(provenance)
    effect["runtime"] = {
        "executable": sys.executable, "python": sys.version.split()[0],
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "03_train_group_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "03_train_effect.json").write_text(
        json.dumps(effect, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("本阶段仅分析冻结的训练集；没有重新随机划分。")
    print(f"训练集客户数：{len(training):,}")
    print("\n两组的14天购买率：")
    for row in summary.to_dict(orient="records"):
        print(f"T={row['T']} {row['group']}: {row['buyers']}/{row['customers']}，"
              f"购买率 {row['conversion_rate']:.3%}，"
              f"95%区间 [{row['rate_ci_low']:.3%}, {row['rate_ci_high']:.3%}]")
    print(f"\n购买率差（男装邮件 - 不发邮件）：{effect['risk_difference_pp']:+.3f} 个百分点")
    print(f"差异的95%区间：[{effect['difference_ci_low_pp']:+.3f}, "
          f"{effect['difference_ci_high_pp']:+.3f}] 个百分点")
    if effect["relative_lift_point_estimate"] is not None:
        print(f"相对提升点估计：{effect['relative_lift_point_estimate']:+.1%}（与百分点不同）")
    print(f"双侧检验 p 值：{effect['p_value']:.3g}")
    if effect["difference_ci_low"] > 0:
        print("训练集区间位于0以上；这是初步正向证据，后续仍需独立评价。")
    elif effect["difference_ci_high"] < 0:
        print("训练集区间位于0以下；这是初步负向证据，后续仍需独立评价。")
    else:
        print("训练集区间包含0；目前方向仍有不确定性，不能据此证明有效或无效。")
    print(f"\n结果已保存到：{output_dir}")
    print("请先解释：购买率差是多少个百分点？95%区间是否包含0？")


if __name__ == "__main__":
    main()
