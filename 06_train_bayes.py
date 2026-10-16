"""第六阶段：训练集的独立Beta-Binomial贝叶斯评估。

回答：在所选先验、模型与训练数据下，邮件购买率差如何分布？
主先验Beta(0.5,0.5)，敏感性先验Beta(1,1)；不根据结果更换主先验。
这是看过05训练结果后的新增探索，不是预注册验证，也不是层级模型。
需要同目录已有05_train_subgroups.py；不用安装新库。
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import math
import numbers
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.stats import beta as beta_distribution


SEED = 20261015
DRAWS = 200_000
PRIORS = [
    {"name": "jeffreys", "alpha": 0.5, "beta": 0.5, "role": "primary"},
    {"name": "uniform", "alpha": 1.0, "beta": 1.0, "role": "sensitivity"},
]
CONFIGURATION = {
    "version": 1,
    "analysis_set": "train",
    "contrast": "Mens E-Mail minus No E-Mail",
    "outcome": "conversion",
    "model": "independent Beta priors and binomial likelihood in each arm of each view",
    "priors": PRIORS,
    "seed": SEED,
    "draws_per_view_and_prior": DRAWS,
    "random_stream": "numpy default_rng(SeedSequence([seed, view_index, prior_index]))",
    "interval": "pointwise equal-tail 95% posterior credible interval of p1-p0",
    "views": "overall plus the two newbie levels and seven original history_segment levels from stage 05",
    "console_focus": ["overall", "history_segment=7) $1,000 +"],
    "selection_context": (
        "Added training exploration after stage-05 results were inspected; "
        "fixed before this stage's posterior calculations, not preregistered."
    ),
    "reporting": "All views and both priors retained; no ranking, formal screening or final targeting list.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value):
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def positive_integer(value, name, minimum=1):
    require(isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_)),
            f"{name}必须是整数。")
    require(value >= minimum, f"{name}不能小于{minimum}。")
    return int(value)


def posterior_from_counts(b0, n0, b1, n1, prior, rng, draws):
    """解析均值/方差；独立抽样估计差值的区间和正向概率，不是MCMC。"""
    n0, n1 = positive_integer(n0, "n0"), positive_integer(n1, "n1")
    b0, b1 = positive_integer(b0, "b0", 0), positive_integer(b1, "b1", 0)
    draws = positive_integer(draws, "draws", 1000)
    require(b0 <= n0 and b1 <= n1, "购买人数不能超过对应总人数。")
    a, b = float(prior["alpha"]), float(prior["beta"])
    require(math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0, "Beta先验参数应为正的有限数。")
    result = {"customers_T0": n0, "buyers_T0": b0, "customers_T1": n1, "buyers_T1": b1}
    variances, samples = [], []
    for t, buyers, n in [(0, b0, n0), (1, b1, n1)]:
        alpha, beta = a + buyers, b + n - buyers
        total = alpha + beta
        mean = alpha / total
        variance = alpha * beta / (total * total * (total + 1))
        low, high = beta_distribution.ppf([0.025, 0.975], alpha, beta)
        result.update({f"posterior_alpha_T{t}": alpha, f"posterior_beta_T{t}": beta,
                       f"posterior_mean_T{t}": mean, f"posterior_variance_T{t}": variance,
                       f"rate_credible_low_T{t}": float(low), f"rate_credible_high_T{t}": float(high)})
        variances.append(variance)
        samples.append(rng.beta(alpha, beta, size=draws))
    difference = samples[1] - samples[0]
    low, high = np.quantile(difference, [0.025, 0.975])
    positive_draws = int(np.count_nonzero(difference > 0))
    probability = positive_draws / draws
    mc_se = None if positive_draws in (0, draws) else math.sqrt(probability * (1 - probability) / draws)
    result.update({
        "risk_difference_mean_pp": 100 * (result["posterior_mean_T1"] - result["posterior_mean_T0"]),
        "risk_difference_sd_pp": 100 * math.sqrt(sum(variances)),
        "risk_difference_mc_mean_pp": 100 * float(difference.mean()),
        "risk_difference_mc_sd_pp": 100 * float(difference.std(ddof=1)),
        "credible_low_pp": 100 * float(low), "credible_high_pp": 100 * float(high),
        "prob_positive_mc": probability, "positive_draws": positive_draws,
        "nonpositive_draws": draws - positive_draws, "draws": draws,
        "mc_se": mc_se, "mc_frequency_step": 1 / draws,
        "mc_boundary_note": (
            "所有模拟同号时，不把频数0/1或零经验标准误解释为真实后验概率恰好0/1；mc_se留空。"
            if mc_se is None else None
        ),
        "status": "estimated",
    })
    require(np.isfinite([result["risk_difference_mean_pp"], low, high]).all(), "后验计算出现非有限值。")
    require(-1 <= low <= high <= 1, "差值可信区间超出[-1,1]。")
    return result


def load_stage05(project_dir):
    dependency_file = project_dir / "05_train_subgroups.py"
    require(dependency_file.is_file(), "缺少05_train_subgroups.py；请把06与05放在同一个项目根目录。")
    specification = importlib.util.spec_from_file_location("hillstrom_stage05_dependency", dependency_file)
    require(specification is not None and specification.loader is not None, "无法载入第五步脚本。")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module, dependency_file


def record_configuration(output_dir, provenance):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "06_bayes_specification.json"
    config_hash = stable_hash(CONFIGURATION)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8-sig"))
        require(previous["configuration"] == CONFIGURATION, "先验或模拟设置发生变化；请保留旧记录并记录新探索。")
        require(previous["configuration_sha256"] == config_hash, "贝叶斯设置校验不一致。")
        for key, value in provenance.items():
            require(previous[key] == value, "贝叶斯记录对应的原始数据、划分或上游结果发生变化。")
    else:
        previous = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": CONFIGURATION, "configuration_sha256": config_hash,
            **provenance,
            "note": "运行记录不等于第三方预注册；此前已经查看05训练分层效果。",
        }
        with path.open("x", encoding="utf-8") as handle:
            json.dump(previous, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return {"configuration_sha256": config_hash, "configuration_record_sha256": digest(path)}


def build_views(training, subgroup_specification):
    def counts(subset):
        result = {}
        for t in [0, 1]:
            y = subset.loc[subset["T"] == t, "conversion"]
            result[f"customers_T{t}"] = len(y)
            result[f"buyers_T{t}"] = int(y.sum())
        return result

    overall = {"feature": "overall", "level": "all", "label": "总体训练集", **counts(training)}
    views = [overall]
    for dimension in subgroup_specification["dimensions"]:
        feature = dimension["feature"]
        family = []
        for item in dimension["levels"]:
            subset = training.loc[training[feature].eq(item["value"])]
            family.append({"feature": feature, "level": item["value"], "label": item["label"], **counts(subset)})
        for t in [0, 1]:
            for field in [f"customers_T{t}", f"buyers_T{t}"]:
                require(sum(row[field] for row in family) == overall[field], "分层汇总没有回到训练集总数。")
        views.extend(family)
    require(len(views) == 10, "预定视角应为总体加9个人群。")
    return views


def validate_stage05_report(report_file, specification_file, stage05, dependency_file, provenance, views):
    specification = json.loads(specification_file.read_text(encoding="utf-8-sig"))
    require(specification["specification"] == stage05.SPECIFICATION, "第五步分层设置发生变化。")
    require(specification["specification_sha256"] == stage05.specification_hash(stage05.SPECIFICATION),
            "第五步分层设置校验不一致。")
    report = json.loads(report_file.read_text(encoding="utf-8-sig"))
    require(report["analysis_set"] == "train" and report["train_rows"] == 25567, "第五步报告不是冻结训练集结果。")
    require(report["specification"] == stage05.SPECIFICATION, "第五步报告采用的分层不同。")
    require(report["script_sha256"] == digest(dependency_file), "第五步脚本与生成已有结果的版本不同。")
    require(report["specification_record_sha256"] == digest(specification_file), "第五步设置记录已改变。")
    for key in ["source_csv_sha256", "split_csv_sha256"]:
        require(report[key] == specification[key] == provenance[key], "第五步报告来源或划分不同。")
    require(len(report["results"]) == len(views) - 1, "第五步分层结果数异常。")
    for previous, current in zip(report["results"], views[1:]):
        for key in ["feature", "level", "customers_T0", "customers_T1", "buyers_T0", "buyers_T1"]:
            require(previous[key] == current[key], "第五步计数与重新核验的训练数据不一致。")


def estimate_views(views):
    results = []
    for view_index, view in enumerate(views):
        for prior_index, prior in enumerate(PRIORS):
            identity = {key: view[key] for key in ["feature", "level", "label"]}
            identity.update({"prior_name": prior["name"], "prior_role": prior["role"],
                             "prior_alpha": prior["alpha"], "prior_beta": prior["beta"],
                             "seed_words": [SEED, view_index, prior_index]})
            if not view["customers_T0"] or not view["customers_T1"]:
                results.append({**identity, **view, "status": "missing_arm",
                                "note": "缺少一组数据，未用纯先验生成有数据支持的效果估计。"})
                continue
            rng = np.random.default_rng(np.random.SeedSequence([SEED, view_index, prior_index]))
            posterior = posterior_from_counts(
                view["buyers_T0"], view["customers_T0"], view["buyers_T1"], view["customers_T1"],
                prior, rng, DRAWS,
            )
            results.append({**identity, **posterior})
    return results


def sensitivity_summary(results):
    summary = []
    primary = [row for row in results if row["prior_role"] == "primary"]
    alternate = [row for row in results if row["prior_role"] == "sensitivity"]
    for first, second in zip(primary, alternate):
        row = {key: first[key] for key in ["feature", "level", "label"]}
        if first["status"] != "estimated" or second["status"] != "estimated":
            row["status"] = "missing_arm"
        else:
            row.update({"status": "estimated",
                        "mean_change_uniform_minus_jeffreys_pp": second["risk_difference_mean_pp"] - first["risk_difference_mean_pp"],
                        "probability_change_uniform_minus_jeffreys": second["prob_positive_mc"] - first["prob_positive_mc"],
                        "probability_change_mc_se": (
                            math.sqrt(first["mc_se"] ** 2 + second["mc_se"] ** 2)
                            if first["mc_se"] is not None and second["mc_se"] is not None else None
                        )})
        summary.append(row)
    return summary


def probability_text(row):
    if row["positive_draws"] == row["draws"]:
        return f"{row['draws']:,}次模拟均为正；不能据此写100%确定"
    if row["positive_draws"] == 0:
        return f"{row['draws']:,}次模拟未见正增量；不能据此写概率恰好为0"
    return (f"{row['prob_positive_mc']:.5%}（正向{row['positive_draws']}/{row['draws']}次；"
            f"模拟标准误约{row['mc_se'] * 100:.3f}个百分点）")


def print_focus(results):
    print("\n先看这里：总体邮件效果")
    print("T0=不发邮件；T1=男装邮件。概率解释均以本模型、先验和训练数据为条件。")
    print("总体及样本较少的第7档在终端展示；全部分层和两套先验都保存在CSV/JSON。")
    focus = [row for row in results if row["feature"] == "overall" or
             (row["feature"] == "history_segment" and row["level"] == "7) $1,000 +")]
    for row in focus:
        print(f"\n{row['label']}；{row['prior_role']} Beta({row['prior_alpha']:g},{row['prior_beta']:g})")
        print(f"  T0：{row['buyers_T0']}/{row['customers_T0']}；T1：{row['buyers_T1']}/{row['customers_T1']}")
        if row["status"] != "estimated":
            print("  缺少一组，无法估计。")
            continue
        print(f"  后验平均购买率差：{row['risk_difference_mean_pp']:+.3f}个百分点")
        print(f"  95%可信区间：[{row['credible_low_pp']:+.3f}, {row['credible_high_pp']:+.3f}]个百分点")
        print(f"  正增量的后验概率：{probability_text(row)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else project_dir / "reports"
    stage05, dependency_file = load_stage05(project_dir)
    training, raw_file, train_ids, provenance = stage05.load_training_features(project_dir)
    report05 = project_dir / "reports" / "05_train_subgroups.json"
    spec05 = project_dir / "reports" / "05_subgroup_specification.json"
    for path in [report05, spec05]:
        require(path.is_file(), f"缺少第五步结果：{path}；请先完成05。")
    provenance.update({"stage05_script_sha256": digest(dependency_file),
                       "stage05_report_sha256": digest(report05),
                       "stage05_specification_sha256": digest(spec05)})
    configuration_provenance = record_configuration(output_dir, provenance)
    training = stage05.attach_training_outcomes(training, raw_file, train_ids)
    views = build_views(training, stage05.SPECIFICATION)
    validate_stage05_report(report05, spec05, stage05, dependency_file, provenance, views)
    results = estimate_views(views)
    sensitivity = sensitivity_summary(results)
    report = {
        "analysis_set": "train", "train_rows": len(training),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": CONFIGURATION, "view_count": len(views), "posterior_rows": len(results),
        "results": results, "sensitivity": sensitivity,
        "notes": [
            "后验为Beta(a+购买人数,b+未购买人数)，两臂先验相同且独立；Delta=p1-p0。",
            "主Jeffreys先验Beta(0.5,0.5)；Beta(1,1)仅用于另一常见先验的敏感性比较。",
            "两套先验都不是从真实历史业务数据估计，不能说对所有先验都稳健。",
            "后验均值/方差解析计算；差值区间与正向概率由20万次独立直接抽样近似，没有使用MCMC。",
            "mc_se表示后验概率数值模拟的标准误，不是营销效果的后验不确定性。",
            "模拟全部同号不等于真实概率恰好0或1；边界mc_se留空，保留命中次数。",
            "这是逐项等尾95%可信区间，不是频率学置信区间，不承诺跨所有分层同时覆盖。",
            "独立Beta-Binomial没有人群间借力，不是层级模型，不称为向总体效果收缩。",
            "总体与两套分层重复使用同一批客户，是不同汇总视角，不能当作独立重复证据或相乘概率。",
            "已看过05训练效果，本步为新增探索；未读取验证/测试Y，不按结果改分层、先验或容量。",
            "因果解释依赖来源的随机分配与独立客户等假设；无客户ID，身份重复仍不能完全核验。",
            "正增量概率高不等于盈利或值得按此选人；本步不生成最终名单，后续仍需独立评价。",
        ],
        "method_references": [
            "https://www.bayesrulesbook.com/chapter-3.html",
            "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html",
        ],
        "runtime": {"executable": sys.executable, "python": sys.version.split()[0],
                    "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "script_sha256": digest(Path(__file__).resolve()),
        **provenance, **configuration_provenance,
    }
    pd.DataFrame(results).to_csv(output_dir / "06_train_bayes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sensitivity).to_csv(output_dir / "06_prior_sensitivity.csv", index=False, encoding="utf-8-sig")
    (output_dir / "06_train_bayes.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"冻结训练集：{len(training):,}人；先验和模拟设置已记录。")
    print("只使用训练集；总体加9个人群，每个视角保留两套先验。")
    print_focus(results)
    print("\n提示：可信区间与前一步置信区间的解释不同；先读总体主分析，再看第7档。")
    print("两套先验的比较范围有限，不能据此宣称所有设定下都稳定。")
    print(f"结果保存到：{output_dir}")
    print("请向上找到“总体训练集；primary Beta(0.5,0.5)”一段，截图后一起解读。")


if __name__ == "__main__":
    main()
