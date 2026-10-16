"""第八阶段：冻结验证集上的等容量营销策略比较。

只复用07已训练的模型，不重新训练或调参；首次进行validation策略效果评估。
四策略均选floor(q*N)人；主容量20%，其余容量仅探索。
使用随机实验的IPW估计相对全部不发邮件的增量，并进行成对Bootstrap。
不读取测试集特征或结果。放在项目根目录，同目录需已有04、07脚本。
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

import joblib
import numpy as np
import pandas as pd
import sklearn


POLICIES = ["random", "history_descending", "p1_descending", "predicted_uplift_descending"]
LABELS = {"random": "固定随机名单", "history_descending": "历史消费排序",
          "p1_descending": "购买概率排序", "predicted_uplift_descending": "预测增量排序"}
CAPACITIES = [0.20, 0.05, 0.10, 0.30]
RANDOM_SEED = 20261014
BOOTSTRAP_SEED = 20261016
BOOTSTRAPS = 5000
CONFIGURATION = {
    "version": 1, "analysis_set": "validation", "outcome": "conversion", "propensity": 0.5,
    "policies": POLICIES, "capacities": CAPACITIES, "primary_capacity": 0.20,
    "capacity_rule": "K=floor(q*N), actual 0/1 lists for every policy",
    "ranking_ties": "score descending, source_row_proxy ascending",
    "random_rule": "sort by source_row_proxy, fixed permutation, first K; same permutation across capacities",
    "random_seed": RANDOM_SEED,
    "estimand": "E[a(X)*(Y(1)-Y(0))], relative to sending no email to all eligible customers",
    "estimator": "mean(a*(T*Y/0.5-(1-T)*Y/0.5))",
    "display_unit": "incremental purchases per 10000 eligible customers, not per 10000 selected customers",
    "bootstrap": {"repeats": BOOTSTRAPS, "seed": BOOTSTRAP_SEED, "interval": "pointwise percentile 95%",
                  "resampling": "paired customer rows, separately within each T arm, preserving arm sizes",
                  "conditioning": "fixed fitted model, scores and original selection indicators; no refit or reranking"},
    "primary_contrast": {"candidate": "predicted_uplift_descending", "baseline": "p1_descending", "capacity": 0.20},
    "auxiliary_contrasts": ["uplift minus random", "uplift minus history"],
    "final_test_rule": (
        "Keep all four rules, the stage-07 model and all capacities fixed. No validation winner selection. "
        "Final test primary contrast is uplift minus p1 at 20%; other contrasts/capacities are auxiliary exploration."
    ),
    "selection_context": "Recorded before first validation-Y analysis in this stage; earlier training analyses have been inspected.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_policy_masks(frame, q, seed=RANDOM_SEED):
    require(len(frame) > 0, "合格客户集合不能为空。")
    require(isinstance(q, numbers.Real) and not isinstance(q, bool) and math.isfinite(q) and 0 <= q <= 1,
            "容量比例应在0到1之间。")
    required = ["source_row_proxy", "history", "p1", "predicted_uplift"]
    require(set(required).issubset(frame.columns), "名单生成缺少特征或分数。")
    proxy = frame["source_row_proxy"].to_numpy()
    require(np.issubdtype(proxy.dtype, np.integer) and (proxy > 0).all(), "代理行号应为正整数。")
    require(len(np.unique(proxy)) == len(proxy), "代理行号不唯一。")
    for column in ["history", "p1", "predicted_uplift"]:
        require(np.isfinite(frame[column].to_numpy(dtype=float)).all(), f"{column}包含非有限分数。")
    require(frame["history"].ge(0).all() and frame["p1"].between(0, 1).all(), "历史金额或购买概率范围异常。")
    require(frame["predicted_uplift"].between(-1, 1).all(), "增量分数超出[-1,1]。")
    n, k = len(frame), int(np.floor(q * len(frame)))
    sorted_positions = np.argsort(proxy, kind="stable")
    random_order = sorted_positions[np.random.default_rng(seed).permutation(n)]
    orders = {"random": random_order}
    for policy, column in [("history_descending", "history"), ("p1_descending", "p1"),
                           ("predicted_uplift_descending", "predicted_uplift")]:
        orders[policy] = np.lexsort((proxy, -frame[column].to_numpy(dtype=float)))
    result = {}
    for policy in POLICIES:
        mask = np.zeros(n, dtype=np.uint8)
        mask[orders[policy][:k]] = 1
        require(int(mask.sum()) == k, "选中人数不等于固定容量。")
        result[policy] = mask
    return result


def ipw_signal(t, y, e=0.5):
    t, y = np.asarray(t), np.asarray(y)
    require(t.ndim == y.ndim == 1 and len(t) == len(y) and len(t) > 0, "T和Y应为非空等长一维数组。")
    require(np.isin(t, [0, 1]).all() and np.isin(y, [0, 1]).all(), "T/Y必须为0/1。")
    require(isinstance(e, numbers.Real) and math.isfinite(e) and 0 < e < 1, "处理概率应在0到1之间。")
    return t * y / e - (1 - t) * y / (1 - e)


def paired_stratified_bootstrap(values, t, repeats=BOOTSTRAPS, seed=BOOTSTRAP_SEED):
    values, t = np.asarray(values, dtype=float), np.asarray(t)
    require(values.ndim == 2 and values.shape[1] > 0 and len(values) == len(t) and t.ndim == 1,
            "Bootstrap应有至少一个策略列，T应与客户行等长。")
    require(np.isfinite(values).all() and np.isin(t, [0, 1]).all(), "Bootstrap含非法数值或T。")
    require(isinstance(repeats, numbers.Integral) and not isinstance(repeats, bool) and repeats >= 100,
            "Bootstrap次数应为至少100的整数。")
    arms = [np.flatnonzero(t == arm) for arm in [0, 1]]
    require(all(len(arm) >= 2 for arm in arms), "Bootstrap每个实验臂至少需要2人。")
    rng = np.random.default_rng(seed)
    samples = np.empty((repeats, values.shape[1]))
    for b in range(repeats):
        indices0 = rng.choice(arms[0], size=len(arms[0]), replace=True)
        indices1 = rng.choice(arms[1], size=len(arms[1]), replace=True)
        samples[b] = (values[indices0].sum(axis=0) + values[indices1].sum(axis=0)) / len(t)
    return samples


def load_module(path, name):
    require(path.is_file(), f"缺少文件：{path}")
    specification = importlib.util.spec_from_file_location(name, path)
    require(specification is not None and specification.loader is not None, "无法载入依赖脚本。")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_validation_inputs(project_dir):
    path07 = project_dir / "07_train_ranker.py"
    ranker = load_module(path07, "hillstrom_stage07_validation_dependency")
    stage04, path04 = ranker.load_dependency(project_dir, "04_train_balance.py", "hillstrom_stage04_validation_guard")
    # 复用既有来源/划分审计；只会读取训练协变量及全体segment，不读取任何Y。
    _, provenance = stage04.load_training_covariates(project_dir)
    reports = project_dir / "reports"
    report07_file, spec07_file, model_file = [reports / name for name in
                                           ["07_training_summary.json", "07_model_specification.json", "07_ranker.joblib"]]
    for path in [report07_file, spec07_file, model_file]:
        require(path.is_file(), f"缺少第七步产物：{path}")
    report07 = json.loads(report07_file.read_text(encoding="utf-8-sig"))
    spec07 = json.loads(spec07_file.read_text(encoding="utf-8-sig"))
    require(report07["analysis_set"] == "train" and report07["train_rows"] == 25567, "模型不是冻结训练集产物。")
    require(report07["configuration"] == spec07["configuration"] == ranker.CONFIGURATION, "第七步模型配置发生变化。")
    require(report07["configuration_sha256"] == ranker.stable_hash(ranker.CONFIGURATION), "模型配置校验异常。")
    require(report07["configuration_record_sha256"] == digest(spec07_file), "第七步设置记录已变。")
    require(report07["script_sha256"] == digest(path07), "第七步代码版本已变。")
    require(report07["model_sha256"] == digest(model_file), "模型文件校验异常。")
    require(report07["stage04_script_sha256"] == digest(path04), "来源审计依赖已变。")
    require(report07["stage05_script_sha256"] == digest(project_dir / "05_train_subgroups.py"), "训练Y读取依赖已变。")
    for key in ["source_csv_sha256", "split_csv_sha256"]:
        require(report07[key] == spec07[key] == provenance[key], "模型与当前数据/划分不一致。")
    bundle = joblib.load(model_file)
    require(bundle["configuration"] == ranker.CONFIGURATION, "模型包中的配置不一致。")
    require(bundle["runtime"]["scikit_learn"] == sklearn.__version__, "scikit-learn版本与训练时不同。")
    for key in ["source_csv_sha256", "split_csv_sha256"]:
        require(bundle["provenance"][key] == provenance[key], "模型包的来源记录不一致。")
    expected = ranker.CONFIGURATION["proposed_policy_comparison"]
    require(expected["strategies"] == POLICIES and expected["primary_capacity"] == 0.20, "策略与第七步提议不同。")

    assignment = pd.read_csv(project_dir / "data" / "processed" / "split_assignment.csv")
    validation_assignment = assignment.loc[assignment["split"] == "validation"].sort_values("source_row_proxy")
    require(len(validation_assignment) == 8522, "验证集人数异常。")
    require(validation_assignment.groupby("T").size().to_dict() == {0: 4261, 1: 4261}, "验证集两臂人数异常。")
    ids = set(validation_assignment["source_row_proxy"].astype(int))
    raw_file = project_dir / "data" / "raw" / "hillstrom.csv"
    frame = pd.read_csv(raw_file, usecols=ranker.RAW_FEATURES + ["segment"],
                        skiprows=lambda row_number: row_number != 0 and row_number not in ids)
    require(len(frame) == len(validation_assignment), "验证特征提取数量异常。")
    frame["source_row_proxy"] = validation_assignment["source_row_proxy"].to_numpy()
    frame["T"] = validation_assignment["T"].to_numpy()
    require(np.array_equal((frame["segment"] == "Mens E-Mail").astype(int), frame["T"]), "验证行对应顺序异常。")
    provenance.update({"stage07_script_sha256": digest(path07), "model_sha256": digest(model_file),
                       "stage07_report_sha256": digest(report07_file), "stage07_specification_sha256": digest(spec07_file)})
    return frame, raw_file, ids, ranker, bundle, provenance


def record_json(path, payload):
    """保留首次记录；重复执行须内容相同，时间不用于改变规范。"""
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8-sig"))
        require(previous["payload"] == payload, f"既有冻结记录与本次不同：{path.name}；请保留旧版本核查。")
    else:
        with path.open("x", encoding="utf-8") as handle:
            json.dump({"created_at_utc": datetime.now(timezone.utc).isoformat(), "payload": payload},
                      handle, ensure_ascii=False, indent=2, allow_nan=False)
    return digest(path)


def freeze_csv(path, table):
    payload = table.to_csv(index=False).encode("utf-8-sig")
    if path.exists():
        require(path.read_bytes() == payload, f"已冻结的分数/名单发生变化：{path.name}")
    else:
        with path.open("xb") as handle:
            handle.write(payload)
    return digest(path)


def attach_validation_y(frame, raw_file, ids):
    outcomes = pd.read_csv(raw_file, usecols=["segment", "conversion"],
                           skiprows=lambda row_number: row_number != 0 and row_number not in ids)
    require(len(outcomes) == len(frame), "验证Y人数异常。")
    require(np.array_equal(outcomes["segment"].to_numpy(), frame["segment"].to_numpy()), "验证Y关联顺序异常。")
    require(outcomes["conversion"].isin([0, 1]).all(), "验证购买结果不是0/1。")
    result = frame.copy()
    result["conversion"] = outcomes["conversion"].to_numpy()
    return result


def evaluate_policies(frame, masks_by_capacity):
    t, y = frame["T"].to_numpy(), frame["conversion"].to_numpy()
    signal = ipw_signal(t, y)
    rows, values, positions = [], [], {}
    for q in CAPACITIES:
        for policy in POLICIES:
            mask = masks_by_capacity[q][policy]
            n0, n1 = [int(((mask == 1) & (t == arm)).sum()) for arm in [0, 1]]
            b0, b1 = [int(y[(mask == 1) & (t == arm)].sum()) for arm in [0, 1]]
            positions[(q, policy)] = len(rows)
            values.append(mask * signal)
            rows.append({"budget_fraction": q, "analysis_role": "primary_capacity" if q == 0.2 else "exploratory_capacity",
                         "policy": policy, "label": LABELS[policy], "eligible_count": len(frame),
                         "selected_count": int(mask.sum()), "actual_selected_fraction": float(mask.mean()),
                         "selected_T0": n0, "selected_T1": n1, "selected_buyers_T0": b0, "selected_buyers_T1": b1,
                         "selected_negative_predicted_uplift": int(((mask == 1) & frame["predicted_uplift"].lt(0)).sum()),
                         "sparse_selected_buyers": min(b0, b1) < 5})
    values = np.column_stack(values)
    bootstrap = paired_stratified_bootstrap(values, t)
    low, high = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    for j, row in enumerate(rows):
        row.update({"incremental_per_10000_eligible": float(values[:, j].mean() * 10000),
                    "ci_low_per_10000_eligible": float(low[j] * 10000),
                    "ci_high_per_10000_eligible": float(high[j] * 10000),
                    "bootstrap_ci_degenerate": bool(np.isclose(low[j], high[j], atol=1e-15, rtol=0))})
    comparisons = []
    for q in CAPACITIES:
        candidate = "predicted_uplift_descending"
        a = positions[(q, candidate)]
        for baseline in ["p1_descending", "random", "history_descending"]:
            b = positions[(q, baseline)]
            difference = bootstrap[:, a] - bootstrap[:, b]
            low_diff, high_diff = np.quantile(difference, [0.025, 0.975])
            changed = masks_by_capacity[q][candidate] != masks_by_capacity[q][baseline]
            contributions = values[:, a] - values[:, b]
            comparisons.append({"budget_fraction": q, "candidate": candidate, "baseline": baseline,
                                "comparison_role": "primary" if q == 0.20 and baseline == "p1_descending" else "auxiliary_exploratory",
                                "difference_per_10000_eligible": float(contributions.mean() * 10000),
                                "ci_low_per_10000_eligible": float(low_diff * 10000),
                                "ci_high_per_10000_eligible": float(high_diff * 10000),
                                "lists_identical": bool(not changed.any()), "different_memberships": int(changed.sum()),
                                "buyers_among_different_memberships": int(y[changed].sum()),
                                "positive_difference_contribution_rows": int((contributions > 0).sum()),
                                "negative_difference_contribution_rows": int((contributions < 0).sum()),
                                "bootstrap_ci_degenerate": bool(np.isclose(low_diff, high_diff, atol=1e-15, rtol=0))})
    return rows, comparisons


def print_results(rows, comparisons):
    print("第八步完成：验证集策略比较。")
    print("验证集8,522人；主容量20%，四种策略各选1,704人。")
    print("指标：相对所有人不发邮件，每万名合格客户的新增购买估计。")
    print("\n主容量20%的结果（估计值与逐项95% Bootstrap区间）：")
    for row in rows:
        if row["budget_fraction"] == 0.20:
            print(f"{row['label']}：{row['incremental_per_10000_eligible']:+.2f}，"
                  f"区间[{row['ci_low_per_10000_eligible']:+.2f}, {row['ci_high_per_10000_eligible']:+.2f}]")
            if row["sparse_selected_buyers"] or row["bootstrap_ci_degenerate"]:
                print("  提示：选中购买事件稀少或区间退化，不能据此确定真实效果。")
    primary = next(row for row in comparisons if row["comparison_role"] == "primary")
    print("\n主要差值：预测增量排序 - 购买概率排序")
    print(f"每万名合格客户差值：{primary['difference_per_10000_eligible']:+.2f}")
    print(f"成对95%区间：[{primary['ci_low_per_10000_eligible']:+.2f}, {primary['ci_high_per_10000_eligible']:+.2f}]")
    print(f"两份名单不同的客户中，共观察到{primary['buyers_among_different_memberships']}次购买。")
    if primary["lists_identical"]:
        print("这批验证客户中两份名单相同，固定名单比较的差值因此为0。")
    elif primary["bootstrap_ci_degenerate"]:
        print("差值区间退化，需要结合不同名单中的购买事件数解读，不能视为真实效果确定。")
    elif primary["ci_low_per_10000_eligible"] > 0:
        print("验证集初步支持增量排序优于购买概率排序；仍需最终测试集评价。")
    elif primary["ci_high_per_10000_eligible"] < 0:
        print("验证集初步支持购买概率排序更好；保持规则，最终如实评价。")
    else:
        print("差值区间包含0；目前不足以确认增量排序优于购买概率排序。")
    print("\n本阶段未提取测试数据用于评分或效果评价；四策略及最终主比较规则已记录。")
    print("区间仅针对固定模型/名单的近似评价，不包含重训、重排或选择赢家的不确定性。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else project_dir / "reports"
    frame, raw_file, ids, ranker, bundle, provenance = load_validation_inputs(project_dir)
    provenance["stage08_script_sha256"] = digest(Path(__file__).resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration_hash = record_json(output_dir / "08_validation_specification.json",
                                     {"configuration": CONFIGURATION, "configuration_sha256": stable_hash(CONFIGURATION), **provenance})
    scores = ranker.score_ranker(bundle, frame)
    scored = frame[["source_row_proxy", "history"]].join(scores)
    masks_by_capacity = {q: make_policy_masks(scored, q) for q in CAPACITIES}
    selection = frame[["source_row_proxy"]].copy()
    for q in CAPACITIES:
        for policy, mask in masks_by_capacity[q].items():
            selection[f"q{int(q*100):02d}__{policy}"] = mask
    prediction_hash = freeze_csv(output_dir / "08_validation_scores.csv", scored)
    selection_hash = freeze_csv(output_dir / "08_validation_selection.csv", selection)
    frozen_plan = {"configuration": CONFIGURATION, "configuration_sha256": stable_hash(CONFIGURATION),
                   "validation_scores_sha256": prediction_hash, "validation_selection_sha256": selection_hash,
                   "note": "四规则与主比较在本阶段读取验证Y前固定，未按验证成绩另选赢家；09阶段才读取测试数据。",
                   **provenance}
    plan_hash = record_json(output_dir / "08_frozen_policy_plan.json", frozen_plan)
    # 分数、名单和后续评价规则都已记录后，才解析验证集Y；测试行始终跳过。
    frame = attach_validation_y(frame.join(scores), raw_file, ids)
    rows, comparisons = evaluate_policies(frame, masks_by_capacity)
    pd.DataFrame(rows).to_csv(output_dir / "08_validation_policy_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comparisons).to_csv(output_dir / "08_validation_policy_differences.csv", index=False, encoding="utf-8-sig")
    report = {
        "analysis_set": "validation", "validation_rows": len(frame),
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "configuration": CONFIGURATION,
        "arm_counts": {str(t): int((frame["T"] == t).sum()) for t in [0, 1]},
        "arm_buyers": {str(t): int(frame.loc[frame["T"] == t, "conversion"].sum()) for t in [0, 1]},
        "policy_results": rows, "paired_differences": comparisons,
        "notes": [
            "这是验证集开发阶段比较，不是最终测试成绩；没有按验证结果重训、改特征、容量或随机种子。",
            "主指标的分母为全部合格客户N，选中比例约20%；不是选中人群的原始购买率。",
            "IPW使用原始随机实验条件下已知e=0.5，没有用模型预测增量替代实际实验结果。",
            "随机基线是一份固定种子的实际名单，本次估计不是对所有随机名单平均表现的无噪声描述。",
            "同一次按T分层的重抽索引用于全部策略；差值区间直接用成对差构造，不通过单策略区间重叠判断。",
            "近似区间条件于固定模型、分数与名单，未重训或重排，不包含模型拟合和选择不确定性。",
            "Bootstrap重抽已有记录的选中标记，复制样本的选中比例可以波动，不重新取前20%。",
            "所有区间为逐项95%而非同时覆盖；最终主比较固定为20%下uplift-p1，其他为辅助探索。",
            "稀疏购买或全0贡献可令非参Bootstrap区间退化，增加模拟次数不能补足未观察到的事件。",
            "相同名单的固定名单差值为0；这不能推出两种排序对所有未来客户都相同。",
            "因果解释仍依赖随机分配、无干扰与独立客户等假设；原始行代理不能验证客户身份重复。",
            "没有成本/利润信息，不声称线上ROI、真实部署或最终业务触达名单。",
        ],
        "method_reference": "https://arxiv.org/abs/1702.02896",
        "runtime": {"executable": sys.executable, "python": sys.version.split()[0],
                    "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "script_sha256": digest(Path(__file__).resolve()), "configuration_record_sha256": configuration_hash,
        "validation_scores_sha256": prediction_hash, "validation_selection_sha256": selection_hash,
        "frozen_policy_plan_sha256": plan_hash, **provenance,
    }
    (output_dir / "08_validation_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print_results(rows, comparisons)
    print(f"结果保存到：{output_dir}")
    print("请截图主容量20%的四策略结果，以及下方主要差值和成对区间。")


if __name__ == "__main__":
    main()
