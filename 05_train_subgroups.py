"""第五阶段：在冻结训练集中探索客户分层的邮件效果。

本轮固定两个维度：newbie的0/1，以及history_segment原有七档。
它们是看过03总体训练效果后新增的探索，不能称为预注册验证。
先记录本轮分层设置，再读取训练行的conversion；不读取验证/测试Y。
直接运行前，把本文件放到D盘marketing_project_starter根目录。
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import statsmodels
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
)


SPECIFICATION = {
    "version": 1,
    "analysis_set": "train",
    "contrast": "Mens E-Mail minus No E-Mail",
    "outcome": "conversion",
    "effect_scale": "risk difference, T1 minus T0; displayed in percentage points",
    "selection_context": (
        "Stage-5 exploratory choices fixed before computing this round's subgroup effects; "
        "the stage-3 overall training result has already been inspected. Not preregistered."
    ),
    "dimensions": [
        {
            "feature": "newbie",
            "levels": [
                {"value": 0, "label": "非过去12个月新获取的客户"},
                {"value": 1, "label": "过去12个月新获取的客户"},
            ],
        },
        {
            "feature": "history_segment",
            "levels": [
                {"value": "1) $0 - $100", "label": "历史消费第1档"},
                {"value": "2) $100 - $200", "label": "历史消费第2档"},
                {"value": "3) $200 - $350", "label": "历史消费第3档"},
                {"value": "4) $350 - $500", "label": "历史消费第4档"},
                {"value": "5) $500 - $750", "label": "历史消费第5档"},
                {"value": "6) $750 - $1,000", "label": "历史消费第6档"},
                {"value": "7) $1,000 +", "label": "历史消费第7档"},
            ],
        },
    ],
    "intervals": {
        "alpha": 0.05,
        "arm_rate": "Wilson",
        "risk_difference": "Newcombe (statsmodels method='newcomb')",
        "coverage": "pointwise, not simultaneous; no formal subgroup screening",
    },
    "reporting": "Keep every planned level in original order; no outcome-driven sorting or merging.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def specification_hash(specification):
    canonical = json.dumps(specification, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_training_features(project_dir):
    """核验来源及冻结划分；本函数只解析实验前特征和分配组。"""
    raw_file = project_dir / "data" / "raw" / "hillstrom.csv"
    split_file = project_dir / "data" / "processed" / "split_assignment.csv"
    manifest_file = split_file.parent / "split_manifest.json"
    source_file = raw_file.parent / "source_manifest.json"
    for path in (raw_file, split_file, manifest_file, source_file):
        require(path.is_file(), f"缺少文件：{path}；先完成01、02阶段。")
    source = json.loads(source_file.read_text(encoding="utf-8-sig"))
    manifest = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
    raw_hash, split_hash = digest(raw_file), digest(split_file)
    require(raw_hash == source["csv_sha256"] == manifest["source_csv_sha256"], "原始数据校验不一致。")
    require(split_hash == manifest["split_csv_sha256"], "划分校验不一致；不要重新随机划分。")
    require(manifest["contrast"] == "Mens E-Mail versus No E-Mail", "主比较发生变化。")
    require(manifest["primary_outcome"] == "conversion", "主结果变量发生变化。")
    require(len(raw_file.read_bytes().splitlines()) == 64001, "源CSV不符合一条记录一物理行的约定。")

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

    # 全体数据只解析segment，核对客户覆盖和随机分配；不解析活动后字段。
    groups = pd.read_csv(raw_file, usecols=["segment"])
    require(len(groups) == 64000, "原始数据行数异常。")
    ids = assignment["source_row_proxy"].to_numpy()
    require(np.issubdtype(ids.dtype, np.integer), "原始行号应为整数。")
    require(bool(((ids >= 1) & (ids <= len(groups))).all()), "原始行号越界。")
    expected_ids = np.flatnonzero(groups["segment"].isin(["Mens E-Mail", "No E-Mail"])) + 1
    require(np.array_equal(np.sort(ids), expected_ids), "主比较原始行覆盖范围异常。")
    expected_t = (groups.iloc[ids - 1]["segment"].to_numpy() == "Mens E-Mail").astype(int)
    require(np.array_equal(expected_t, assignment["T"].to_numpy()), "T与原始分组不对应。")

    train_assignment = assignment.loc[assignment["split"] == "train"].sort_values("source_row_proxy")
    train_ids = set(train_assignment["source_row_proxy"].astype(int))
    training = pd.read_csv(
        raw_file, usecols=["segment", "newbie", "history_segment"],
        skiprows=lambda row_number: row_number != 0 and row_number not in train_ids,
    )
    require(len(training) == len(train_assignment) == 25567, "训练行提取数量异常。")
    training["T"] = train_assignment["T"].to_numpy()
    training["source_row_proxy"] = train_assignment["source_row_proxy"].to_numpy()
    require(np.array_equal((training["segment"] == "Mens E-Mail").astype(int), training["T"]),
            "训练行顺序与T不一致。")
    require(not training[["newbie", "history_segment"]].isna().any().any(), "分层字段存在缺失。")
    require(training["newbie"].isin([0, 1]).all(), "newbie不是0/1。")
    known_history = [item["value"] for item in SPECIFICATION["dimensions"][1]["levels"]]
    require(training["history_segment"].isin(known_history).all(), "历史档位与固定设置不同。")
    provenance = {"source_csv_sha256": raw_hash, "split_csv_sha256": split_hash}
    return training, raw_file, train_ids, provenance


def record_specification(output_dir, provenance):
    """在本轮读取分层结果前记录设置；重复执行时核对已有记录。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "05_subgroup_specification.json"
    spec_hash = specification_hash(SPECIFICATION)
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8-sig"))
        require(recorded["specification"] == SPECIFICATION, "分层设置发生变化；请记录新探索，保留旧记录。")
        require(recorded["specification_sha256"] == spec_hash, "分层设置记录校验不一致。")
        for key, value in provenance.items():
            require(recorded[key] == value, "分层记录的数据来源或划分发生变化。")
    else:
        recorded = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "specification": SPECIFICATION,
            "specification_sha256": spec_hash,
            **provenance,
            "note": "软件运行记录用于追踪，不等于第三方预注册或未查看过总体结果的证明。",
        }
        with path.open("x", encoding="utf-8") as handle:
            json.dump(recorded, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return {"specification_sha256": spec_hash, "specification_record_sha256": digest(path)}


def attach_training_outcomes(training, raw_file, train_ids):
    """只解析训练行Y；用原始行号和分组重新核对关联。"""
    outcomes = pd.read_csv(
        raw_file, usecols=["segment", "conversion"],
        skiprows=lambda row_number: row_number != 0 and row_number not in train_ids,
    )
    require(len(outcomes) == len(training), "训练Y提取数量异常。")
    require(np.array_equal(outcomes["segment"].to_numpy(), training["segment"].to_numpy()),
            "训练Y与特征顺序不一致。")
    require(outcomes["conversion"].isin([0, 1]).all(), "训练Y不是0/1。")
    result = training.copy()
    result["conversion"] = outcomes["conversion"].to_numpy()
    return result


def estimate_subgroup(subset):
    """返回T1-T0，保留各组分母与购买人数；不执行筛选检验。"""
    result = {}
    for t in [0, 1]:
        y = subset.loc[subset["T"] == t, "conversion"]
        n, buyers = len(y), int(y.sum())
        result[f"customers_T{t}"] = n
        result[f"buyers_T{t}"] = buyers
        if n:
            low, high = proportion_confint(buyers, n, alpha=0.05, method="wilson")
            result.update({f"rate_T{t}": buyers / n,
                           f"rate_ci_low_T{t}": float(low), f"rate_ci_high_T{t}": float(high)})
        else:
            result.update({f"rate_T{t}": None, f"rate_ci_low_T{t}": None, f"rate_ci_high_T{t}": None})
    n0, n1 = result["customers_T0"], result["customers_T1"]
    b0, b1 = result["buyers_T0"], result["buyers_T1"]
    result["sparse_buyers"] = bool(n0 and n1 and min(b0, b1) < 5)
    if n0 and n1:
        difference = result["rate_T1"] - result["rate_T0"]
        low, high = confint_proportions_2indep(b1, n1, b0, n0, compare="diff", method="newcomb", alpha=0.05)
        require(np.isfinite([difference, low, high]).all(), "区间计算出现非有限值。")
        require(low <= difference <= high, "区间未包含点估计，请检查运行环境。")
        result.update({"risk_difference": difference,
                       "risk_difference_pp": 100 * difference,
                       "difference_ci_low_pp": 100 * float(low),
                       "difference_ci_high_pp": 100 * float(high),
                       "ci_width_pp": 100 * float(high - low),
                       "status": "estimated"})
    else:
        result.update({"risk_difference": None, "risk_difference_pp": None,
                       "difference_ci_low_pp": None, "difference_ci_high_pp": None,
                       "ci_width_pp": None, "status": "missing_arm"})
    return result


def subgroup_records(training):
    records = []
    for dimension in SPECIFICATION["dimensions"]:
        feature = dimension["feature"]
        family = []
        for item in dimension["levels"]:
            subset = training.loc[training[feature].eq(item["value"])]
            family.append({"feature": feature, "level": item["value"], "label": item["label"],
                           **estimate_subgroup(subset)})
        # 每个维度各自覆盖整个训练集。两个维度是同一批客户，不能把9行再合并当新样本。
        for t in [0, 1]:
            require(sum(row[f"customers_T{t}"] for row in family) == int((training["T"] == t).sum()),
                    "分层人数没有回到对应训练组总人数。")
            require(sum(row[f"buyers_T{t}"] for row in family) == int(training.loc[training["T"] == t, "conversion"].sum()),
                    "分层购买人数汇总不一致。")
        records.extend(family)
    return records


def print_results(records):
    print("\n固定顺序展示全部分层；T0=不发邮件，T1=男装邮件：")
    for row in records:
        print(f"\n{row['feature']}={row['level']}（{row['label']}）")
        arm_text = []
        for t in [0, 1]:
            n, buyers, rate = row[f"customers_T{t}"], row[f"buyers_T{t}"], row[f"rate_T{t}"]
            arm_text.append(f"T{t}：{buyers}/{n}=" + (f"{rate:.3%}" if rate is not None else "无法估计"))
        print("  " + "；".join(arm_text))
        if row["status"] == "estimated":
            print(f"  购买率差：{row['risk_difference_pp']:+.3f}个百分点；"
                  f"95%区间 [{row['difference_ci_low_pp']:+.3f}, {row['difference_ci_high_pp']:+.3f}]")
            if row["sparse_buyers"]:
                print("  注意：至少一组购买人数少于5，保留该组并谨慎解读区间。")
        else:
            print("  缺少邮件组或对照组，购买率差无法估计。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else project_dir / "reports"
    training, raw_file, train_ids, provenance = load_training_features(project_dir)
    specification_provenance = record_specification(output_dir, provenance)
    training = attach_training_outcomes(training, raw_file, train_ids)
    records = subgroup_records(training)
    report = {
        "analysis_set": "train", "train_rows": len(training), "subgroup_rows": len(records),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm_counts": {str(t): int((training["T"] == t).sum()) for t in [0, 1]},
        "specification": SPECIFICATION,
        "results": records,
        "notes": [
            "本步为总体训练效果已查看后的新增探索，不是项目开始时预注册的假设验证。",
            "newbie=1指过去12个月新获取的客户，不能直接解释为首单客户。",
            "原有history_segment标签原样保留，不依据分层购买结果重新分箱或合并。",
            "两套分层使用同一批客户，9行不是9个独立实验；各维度的样本数分别回到25567。",
            "区间为单项95%区间，未作多重比较调整，不承诺9个区间同时覆盖。",
            "点估计不同，或一组区间跨0而另一组不跨0，均不能单独证明群体效果不同。",
            "没有进行异质性的直接检验，不据此对客户群贴有效/无效标签或选最终名单。",
            "后续如要证明异质性，需预先固定效应尺度并直接比较效应差或相应交互。",
            "区间依赖独立客户/二项样本假设；原数据无客户ID，无法完全核验身份重复。",
            "只解析训练行的conversion；不查看验证/测试Y，不去重，不重拆分，不改原始数据。",
        ],
        "method_references": [
            "https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html",
            "https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.confint_proportions_2indep.html",
        ],
        "runtime": {"executable": sys.executable, "python": sys.version.split()[0],
                    "numpy": np.__version__, "pandas": pd.__version__, "statsmodels": statsmodels.__version__},
        "script_sha256": digest(Path(__file__).resolve()),
        **provenance, **specification_provenance,
    }
    pd.DataFrame(records).to_csv(output_dir / "05_train_subgroups.csv", index=False, encoding="utf-8-sig")
    (output_dir / "05_train_subgroups.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"冻结训练集：{len(training):,}人；只探索newbie和原有历史消费档位。")
    print("本轮设置已记录；这是新增探索，未重新随机划分，未读取验证/测试Y。")
    print_results(records)
    print("\n请注意：以上为单项95%区间，不是9项同时覆盖。")
    print("一组区间跨0、另一组不跨0，不代表两组效果之间存在明确差异。")
    print("本轮用于形成待验证假设，不据此形成最终触达名单。")
    print(f"结果保存到：{output_dir}")
    print("请先向上找到newbie=0和newbie=1两段，比较人数、购买率差及区间宽度。")


if __name__ == "__main__":
    main()
