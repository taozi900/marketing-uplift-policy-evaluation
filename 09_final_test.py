"""第九阶段：按第八步冻结规则完成保留测试集评价。

不重新训练模型，不按验证/测试表现更换容量、排序方式或随机种子。
先保存测试分数、名单与评价协议，再解析测试行conversion。
总体邮件效果为辅助确认；主策略比较为20%容量下增量排序减购买概率排序。
本文件放项目根目录，同目录需已有04、07、08及其已运行产物。
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
import statsmodels
from statsmodels.stats.proportion import proportion_confint, confint_proportions_2indep


def require(condition, message):
    if not condition:
        raise ValueError(message)


def overall_test_effect(frame):
    require({"T", "conversion"}.issubset(frame.columns), "总体效果缺少T/Y。")
    require(frame["T"].isin([0, 1]).all() and set(frame["T"].unique()) == {0, 1}, "总体效果需要两个实验臂。")
    require(frame["conversion"].isin([0, 1]).all(), "购买结果应为0/1。")
    groups = []
    for t, label in [(0, "No E-Mail"), (1, "Mens E-Mail")]:
        values = frame.loc[frame["T"] == t, "conversion"]
        n, buyers = len(values), int(values.sum())
        require(n > 0, "总体效果缺少一个实验臂。")
        low, high = proportion_confint(buyers, n, alpha=0.05, method="wilson")
        groups.append({"T": t, "group": label, "customers": n, "buyers": buyers, "rate": buyers / n,
                       "rate_ci_low": float(low), "rate_ci_high": float(high)})
    control, treated = groups
    low, high = confint_proportions_2indep(treated["buyers"], treated["customers"],
                                         control["buyers"], control["customers"],
                                         compare="diff", method="newcomb", alpha=0.05)
    difference = treated["rate"] - control["rate"]
    require(np.isfinite([difference, low, high]).all(), "总体效果或区间不是有限数。")
    return {"analysis_set": "test", "role": "planned auxiliary overall effect confirmation",
            "contrast": "Mens E-Mail minus No E-Mail", "groups": groups,
            "risk_difference_pp": 100 * difference, "difference_ci_low_pp": 100 * float(low),
            "difference_ci_high_pp": 100 * float(high), "interval": "pointwise Newcombe 95%"}


def attach_test_y(frame, raw_file, ids):
    outcomes = pd.read_csv(raw_file, usecols=["segment", "conversion"],
                           skiprows=lambda row_number: row_number != 0 and row_number not in ids)
    require(len(outcomes) == len(frame) == len(ids), "测试Y提取人数异常。")
    require(np.array_equal(outcomes["segment"].to_numpy(), frame["segment"].to_numpy()), "测试Y关联顺序异常。")
    require(np.array_equal((outcomes["segment"] == "Mens E-Mail").astype(int), frame["T"]), "测试Y与T不对应。")
    require(outcomes["conversion"].isin([0, 1]).all(), "测试购买结果不是0/1。")
    result = frame.copy()
    result["conversion"] = outcomes["conversion"].to_numpy()
    return result


def load_frozen_assets(project_dir):
    """核验原始数据、模型和08冻结记录；本函数不提取测试行X/Y。"""
    import hashlib
    import importlib.util

    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    reports = project_dir / "reports"
    paths = {name: reports / name for name in [
        "08_frozen_policy_plan.json", "08_validation_specification.json", "08_validation_summary.json",
        "08_validation_scores.csv", "08_validation_selection.csv", "07_training_summary.json",
        "07_model_specification.json", "07_ranker.joblib",
    ]}
    for path in paths.values():
        require(path.is_file(), f"缺少已冻结产物：{path}；先完成07、08。")
    plan = json.loads(paths["08_frozen_policy_plan.json"].read_text(encoding="utf-8-sig"))["payload"]
    specification08 = json.loads(paths["08_validation_specification.json"].read_text(encoding="utf-8-sig"))["payload"]
    summary08 = json.loads(paths["08_validation_summary.json"].read_text(encoding="utf-8-sig"))
    path08 = project_dir / "08_validate_policies.py"
    require(path08.is_file() and digest(path08) == plan["stage08_script_sha256"] == summary08["stage08_script_sha256"],
            "第八步代码与冻结时不同，停止最终评价。")
    module_spec = importlib.util.spec_from_file_location("hillstrom_stage08_final_dependency", path08)
    require(module_spec is not None and module_spec.loader is not None, "无法载入第八步纯评价函数。")
    stage08 = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(stage08)
    require(plan["configuration"] == specification08["configuration"] == summary08["configuration"] == stage08.CONFIGURATION,
            "冻结策略与当前第八步规则不同。")
    require(plan["configuration_sha256"] == stage08.stable_hash(stage08.CONFIGURATION), "冻结配置校验异常。")
    require(summary08["analysis_set"] == "validation" and summary08["validation_rows"] == 8522, "第八步结果集合异常。")
    for key, name in [("frozen_policy_plan_sha256", "08_frozen_policy_plan.json"),
                      ("configuration_record_sha256", "08_validation_specification.json"),
                      ("validation_scores_sha256", "08_validation_scores.csv"),
                      ("validation_selection_sha256", "08_validation_selection.csv")]:
        require(summary08[key] == digest(paths[name]), f"08产物哈希不一致：{name}")
    for key, name in [("validation_scores_sha256", "08_validation_scores.csv"),
                      ("validation_selection_sha256", "08_validation_selection.csv")]:
        require(plan[key] == digest(paths[name]), "验证分数或名单与冻结计划不同。")

    path07 = project_dir / "07_train_ranker.py"
    require(path07.is_file() and digest(path07) == plan["stage07_script_sha256"], "模型打分实现与冻结时不同。")
    ranker = stage08.load_module(path07, "hillstrom_stage07_final_dependency")
    stage04, path04 = ranker.load_dependency(project_dir, "04_train_balance.py", "hillstrom_stage04_final_guard")
    # 复用来源/原始行号审计，仅训练协变量及全体segment，不读取任何测试购买结果。
    _, provenance = stage04.load_training_covariates(project_dir)
    report07 = json.loads(paths["07_training_summary.json"].read_text(encoding="utf-8-sig"))
    specification07 = json.loads(paths["07_model_specification.json"].read_text(encoding="utf-8-sig"))
    require(report07["analysis_set"] == "train" and report07["train_rows"] == 25567, "模型训练集合异常。")
    require(report07["configuration"] == specification07["configuration"] == ranker.CONFIGURATION, "07模型配置已变。")
    require(report07["configuration_record_sha256"] == digest(paths["07_model_specification.json"]), "07设置记录已变。")
    require(report07["script_sha256"] == digest(path07), "07代码版本不一致。")
    require(report07["stage04_script_sha256"] == digest(path04), "来源审计依赖已变。")
    require(report07["stage05_script_sha256"] == digest(project_dir / "05_train_subgroups.py"), "训练Y读取依赖已变。")
    for key in ["source_csv_sha256", "split_csv_sha256"]:
        require(provenance[key] == plan[key] == summary08[key] == report07[key] == specification07[key],
                "原始数据或划分与冻结记录不一致。")
    for key, name in [("model_sha256", "07_ranker.joblib"), ("stage07_report_sha256", "07_training_summary.json"),
                      ("stage07_specification_sha256", "07_model_specification.json")]:
        require(plan[key] == summary08[key] == digest(paths[name]), "冻结模型或07训练记录已变。")
    require(report07["model_sha256"] == digest(paths["07_ranker.joblib"]), "模型文件哈希不一致。")
    require(report07["runtime"]["scikit_learn"] == sklearn.__version__, "模型训练与当前scikit-learn版本不同。")
    bundle = joblib.load(paths["07_ranker.joblib"])
    require(bundle["configuration"] == ranker.CONFIGURATION, "模型包配置异常。")
    for key in ["source_csv_sha256", "split_csv_sha256"]:
        require(bundle["provenance"][key] == provenance[key], "模型包的来源不同。")
    provenance.update({"model_sha256": digest(paths["07_ranker.joblib"]), "stage07_script_sha256": digest(path07),
                       "stage07_report_sha256": digest(paths["07_training_summary.json"]),
                       "stage08_script_sha256": digest(path08),
                       "stage08_frozen_plan_sha256": digest(paths["08_frozen_policy_plan.json"]),
                       "stage08_summary_sha256": digest(paths["08_validation_summary.json"])})
    return stage08, ranker, bundle, provenance


def load_test_features(project_dir, ranker):
    assignment = pd.read_csv(project_dir / "data" / "processed" / "split_assignment.csv")
    test_assignment = assignment.loc[assignment["split"] == "test"].sort_values("source_row_proxy")
    require(len(test_assignment) == 8524, "保留测试集人数异常。")
    require(test_assignment.groupby("T").size().to_dict() == {0: 4262, 1: 4262}, "测试集两臂人数异常。")
    ids = set(test_assignment["source_row_proxy"].astype(int))
    raw_file = project_dir / "data" / "raw" / "hillstrom.csv"
    frame = pd.read_csv(raw_file, usecols=ranker.RAW_FEATURES + ["segment"],
                        skiprows=lambda row_number: row_number != 0 and row_number not in ids)
    require(len(frame) == len(test_assignment), "测试特征提取数量异常。")
    frame["source_row_proxy"] = test_assignment["source_row_proxy"].to_numpy()
    frame["T"] = test_assignment["T"].to_numpy()
    require(np.array_equal((frame["segment"] == "Mens E-Mail").astype(int), frame["T"]), "测试行顺序与T不同。")
    return frame, raw_file, ids


def primary_interpretation(primary):
    if primary["lists_identical"]:
        return "这批测试客户的两份名单相同，固定名单差值为0；不能推出未来名单始终相同。"
    if primary["bootstrap_ci_degenerate"]:
        return "差值Bootstrap区间退化，需结合购买事件数解读；不能认定真实效果确定或相同。"
    if primary["ci_low_per_10000_eligible"] > 0:
        return "本次离线测试支持增量排序优于购买概率排序；结论范围限于当前冻结模型、容量与评价假设。"
    if primary["ci_high_per_10000_eligible"] < 0:
        return "本次离线测试支持购买概率排序优于增量排序；应保留并如实报告该结果。"
    return "主差值区间包含0，证据不足以确认两种策略的优劣；不能将其解释为效果相同。"


def findings_markdown(overall, rows, primary, interpretation):
    control, treated = overall["groups"]
    lines = [
        "# 营销活动增量评估：最终测试结果", "",
        "本项目使用Hillstrom公开随机邮件实验，是离线研究。主比较为男装邮件与不发邮件，结果为14天购买。", "",
        "## 总体邮件效果（辅助确认）", "",
        f"保留测试集8,524人：不发邮件组{control['buyers']}/{control['customers']}购买（{control['rate']:.3%}）；"
        f"男装邮件组{treated['buyers']}/{treated['customers']}购买（{treated['rate']:.3%}）。",
        f"购买率差为{overall['risk_difference_pp']:+.3f}个百分点，Newcombe单项95%区间"
        f"[{overall['difference_ci_low_pp']:+.3f}, {overall['difference_ci_high_pp']:+.3f}]个百分点。", "",
        "## 约20%触达容量下的冻结策略比较", "",
        "四条规则各选1,704人。指标是相对所有人都不发邮件，每万名合格客户的新增购买离线估计。", "",
        "| 策略 | 每万名合格客户新增购买估计 | 逐项95% Bootstrap区间 |", "|---|---:|---:|",
    ]
    for row in rows:
        if row["budget_fraction"] == 0.20:
            lines.append(f"| {row['label']} | {row['incremental_per_10000_eligible']:+.2f} | "
                         f"[{row['ci_low_per_10000_eligible']:+.2f}, {row['ci_high_per_10000_eligible']:+.2f}] |")
    lines.extend([
        "", "## 事前固定的主要策略差值", "",
        f"增量排序减购买概率排序：每万名合格客户{primary['difference_per_10000_eligible']:+.2f}次购买，"
        f"成对95%区间[{primary['ci_low_per_10000_eligible']:+.2f}, {primary['ci_high_per_10000_eligible']:+.2f}]。",
        f"两份名单不同的客户中，共观察到{primary['buyers_among_different_memberships']}次购买；这些记录支撑本次策略差值估计。",
        interpretation, "",
        "总体邮件效果与客户排序优势是两个问题，分别报告，不能相互替代。", "",
        "## 解释边界与业务建议", "",
        "- 模型、三项交互、L2强度、容量、随机种子、四策略及主要差值沿用08冻结方案，未根据测试结果改动。",
        "- 区间为固定模型、分数与名单下的近似评价，未重训、重排，不包含模型训练不确定性；各区间不承诺同时覆盖。",
        "- 购买事件稀疏时，非参Bootstrap区间可能不稳定或退化；增加模拟次数不能补足未观察到的事件。相同名单的差值为0仅表示本批客户决策相同。",
        "- 原始数据缺少真实客户ID，CSV行号只是代理；未按相同属性去重，独立客户假设仍有局限。",
        "- 随机策略是一份固定随机名单。辅助基线与其他容量仅作探索，不凭点估计挑选赢家。",
        "- 离线候选文件用于研究复核，没有真实CRM身份，也没有成本、利润或真实触达信息。",
        "- 如需业务试点，应先明确成本和有意义的最小增量、检查实际客户身份与邮件送达，并在新样本中随机比较候选策略。",
        "- 不再用本次测试集调参；未来新增方法需标为后续探索，并收集新的独立评价数据。", "",
        "数据说明：https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html", "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else project_dir / "reports"
    stage08, ranker, bundle, provenance = load_frozen_assets(project_dir)
    provenance["stage09_script_sha256"] = stage08.digest(Path(__file__).resolve())
    protocol = {
        "analysis_set": "test", "inherited_rules": stage08.CONFIGURATION,
        "inherited_rules_note": "继承配置中的validation表示规则来源；本次评价数据及输出集合均为test。",
        "overall_effect": "planned auxiliary T1-T0 14-day conversion difference; Wilson arms/Newcombe pointwise 95%",
        "no_new_search": "No model/feature/subgroup/prior/capacity search; no refit on train+validation.",
        "evaluation_scope": "Final held-out evaluation under frozen rules; identical rerun is not a new experiment.",
        **provenance,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_hash = stage08.record_json(output_dir / "09_test_specification.json", protocol)
    frame, raw_file, ids = load_test_features(project_dir, ranker)
    scores = ranker.score_ranker(bundle, frame)
    scored = frame[["source_row_proxy", "history"]].join(scores)
    masks_by_capacity = {q: stage08.make_policy_masks(scored, q) for q in stage08.CAPACITIES}
    selection = frame[["source_row_proxy"]].copy()
    for q in stage08.CAPACITIES:
        for policy, mask in masks_by_capacity[q].items():
            selection[f"q{int(q*100):02d}__{policy}"] = mask
    score_hash = stage08.freeze_csv(output_dir / "09_test_scores.csv", scored)
    selection_hash = stage08.freeze_csv(output_dir / "09_test_selection.csv", selection)
    # 事前主候选策略的离线名单示例；是否有优势仍由下方独立评价决定。
    candidate_mask = masks_by_capacity[0.20]["predicted_uplift_descending"].astype(bool)
    candidates = scored.loc[candidate_mask, ["source_row_proxy", "p0", "p1", "predicted_uplift", "predicted_uplift_pp"]].copy()
    candidates["negative_predicted_uplift"] = candidates["predicted_uplift"].lt(0)
    candidates["policy"] = "frozen uplift Top-20% offline example"
    candidates["identity_note"] = "CSV row proxy only; not a verified customer/CRM ID"
    candidates["use_note"] = "Research review only; no actual contact or deployment"
    candidate_hash = stage08.freeze_csv(output_dir / "09_offline_candidate_uplift20.csv", candidates)
    snapshot_hash = stage08.record_json(output_dir / "09_pre_outcome_snapshot.json", {
        "protocol_sha256": protocol_hash, "test_scores_sha256": score_hash,
        "test_selection_sha256": selection_hash, "offline_candidate_sha256": candidate_hash,
        "test_rows": len(frame), "primary_selected_count": int(candidate_mask.sum()), **provenance,
    })
    # 到这里协议、测试分数和名单已记录；之后才解析测试Y并进行冻结规则的最终效果计算。
    frame = attach_test_y(frame.join(scores), raw_file, ids)
    overall = overall_test_effect(frame)
    rows, comparisons = stage08.evaluate_policies(frame, masks_by_capacity)
    for row in rows + comparisons:
        row["analysis_set"] = "test"
    primary = next(row for row in comparisons if row["comparison_role"] == "primary")
    interpretation = primary_interpretation(primary)
    pd.DataFrame(rows).to_csv(output_dir / "09_test_policy_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comparisons).to_csv(output_dir / "09_test_policy_differences.csv", index=False, encoding="utf-8-sig")
    (output_dir / "09_test_overall_effect.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    report = {
        "analysis_set": "test", "test_rows": len(frame), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inherited_frozen_rules": stage08.CONFIGURATION, "overall_effect": overall,
        "policy_results": rows, "paired_differences": comparisons, "primary_interpretation": interpretation,
        "notes": [
            "测试集仅用于冻结规则的最终效果评价，没有因成绩不理想再改模型、阈值、容量或子组。",
            "主策略差值为约20%容量下uplift-p1；总体邮件效果是计划内辅助确认，其他基线/容量为探索。",
            "继承配置中的analysis_set=validation只表示规则来源，当前数据及结果集合为test。",
            "总体效应分母为各实验组客户；策略增量分母为全体合格客户，两个单位不要混用。",
            "Bootstrap范围、稀疏事件与逐项区间局限继承08；不代表整个模型训练流程的无条件区间。",
            "原始行号不是身份验证；潜在重复客户、随机化与独立性假设仍然适用。",
            "离线候选名单在解析测试Y前由冻结分数生成，是研究示例，不表示已获证实的推荐名单或线上部署。",
            "相同代码/规则的操作性重跑可核验复现；重复运行不产生新的独立证据。",
            "后续新增模型需要新的评价数据，不能继续使用本次测试集调参并宣称保留测试。",
        ],
        "runtime": {"executable": sys.executable, "python": sys.version.split()[0],
                    "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                    "statsmodels": statsmodels.__version__},
        "protocol_record_sha256": protocol_hash, "pre_outcome_snapshot_sha256": snapshot_hash,
        "test_scores_sha256": score_hash, "test_selection_sha256": selection_hash,
        "offline_candidate_sha256": candidate_hash, **provenance,
    }
    (output_dir / "09_test_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "09_final_findings.md").write_text(
        findings_markdown(overall, rows, primary, interpretation), encoding="utf-8")

    print("第九步完成：保留测试集最终评价。")
    print(f"测试集：{len(frame):,}人；T0=4,262，T1=4,262；主20%每种策略选{int(candidate_mask.sum()):,}人。")
    print("\n总体邮件效果（辅助确认）：")
    for row in overall["groups"]:
        print(f"T={row['T']}：{row['buyers']}/{row['customers']}购买，购买率{row['rate']:.3%}")
    print(f"购买率差：{overall['risk_difference_pp']:+.3f}个百分点，95%区间"
          f"[{overall['difference_ci_low_pp']:+.3f}, {overall['difference_ci_high_pp']:+.3f}]")
    print("\n主容量20%的策略效果：每万名合格客户的新增购买离线估计")
    for row in rows:
        if row["budget_fraction"] == 0.20:
            print(f"{row['label']}：{row['incremental_per_10000_eligible']:+.2f}，"
                  f"区间[{row['ci_low_per_10000_eligible']:+.2f}, {row['ci_high_per_10000_eligible']:+.2f}]")
            if row["sparse_selected_buyers"] or row["bootstrap_ci_degenerate"]:
                print("  注意：事件稀疏或区间退化，需保留不确定性。")
    print("\n最终主要差值：预测增量排序 - 购买概率排序")
    print(f"每万名合格客户差值：{primary['difference_per_10000_eligible']:+.2f}；成对95%区间"
          f"[{primary['ci_low_per_10000_eligible']:+.2f}, {primary['ci_high_per_10000_eligible']:+.2f}]")
    print(f"两份名单不同的客户中，共观察到{primary['buyers_among_different_memberships']}次购买。")
    print(interpretation)
    print("模型、容量与选人规则保持冻结；不根据本次测试成绩继续调参。")
    print(f"结果及09_final_findings.md保存到：{output_dir}")
    print("请截图总体邮件效果、四种策略结果和最终主要差值，我们再整理项目结论。")


if __name__ == "__main__":
    main()
