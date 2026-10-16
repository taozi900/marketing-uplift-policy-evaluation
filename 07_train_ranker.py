"""第七阶段：只用冻结训练集拟合客户排序模型。

同一个带三项交互的L2逻辑回归，分别预测p0和p1。
购买概率排序使用p1，增量排序使用p1-p0；都只是模型分数。
本步不读取验证/测试特征或结果，不比较策略收益，不产生最终触达名单。
把本文件放到项目根目录，同目录需要已有04和05脚本。
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAW_FEATURES = ["recency", "history", "newbie", "mens", "womens", "zip_code", "channel"]
NUMERIC_FEATURES = ["recency", "log_history"]
BINARY_FEATURES = ["newbie", "mens", "womens"]
CATEGORICAL_FEATURES = ["zip_code", "channel"]
INTERACTION_FEATURES = ["num__recency", "num__log_history", "binary__newbie"]
MODEL_PARAMETERS = {
    "C": 1.0, "solver": "lbfgs", "max_iter": 5000, "tol": 1e-8,
    "class_weight": None, "fit_intercept": True,
}
CONFIGURATION = {
    "version": 1,
    "analysis_set": "train",
    "contrast": "Mens E-Mail minus No E-Mail",
    "outcome": "conversion",
    "raw_feature_whitelist": RAW_FEATURES,
    "excluded_from_customer_covariates_Z": ["history_segment", "T", "segment", "source_row_proxy", "visit", "conversion", "spend"],
    "numeric": "log_history=log1p(history); StandardScaler fitted on the full training covariates only",
    "binary": BINARY_FEATURES,
    "categorical": {"columns": CATEGORICAL_FEATURES, "encoding": "training-only OneHotEncoder",
                    "drop": "first", "handle_unknown": "error", "raw_spelling": "preserved"},
    "model": "single S-learner logistic regression, L2 penalty",
    "design": "[Z, T, T*Z_selected]; Z includes all allowed main effects",
    "treatment_interactions": INTERACTION_FEATURES,
    "parameters": MODEL_PARAMETERS,
    "penalty_api": "Use LogisticRegression's L2 defaults; omit the deprecated penalty keyword.",
    "counterfactual_scores": {"p0": "set T=0", "p1": "set T=1", "predicted_uplift": "p1-p0"},
    "proposed_policy_comparison": {
        "strategies": ["random", "history_descending", "p1_descending", "predicted_uplift_descending"],
        "primary_capacity": 0.20, "exploratory_capacities": [0.05, 0.10, 0.30],
        "capacity_rule": "floor(q*N), same eligible rows and capacity for every strategy",
        "score_tie_rule": "source_row_proxy ascending", "random_seed": 20261014,
        "negative_uplift": "Report selected negative-score count; no automatic live contact.",
    },
    "selection_context": (
        "Training design recorded after stages 03-06 training analyses and before model fitting; "
        "no held-out outcome-based tuning. Three interaction variables follow the original reference formula; "
        "fixed L2 regularization is a new implementation choice."
    ),
    "reporting": "Training predictions demonstrate score calculation only; validation and test reserved for later stages.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value):
    value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prepare_features(frame):
    """显式白名单：T、原始行号、活动后结果均不进入Z。"""
    require(set(RAW_FEATURES).issubset(frame.columns), "缺少模型要求的实验前特征。")
    features = frame[RAW_FEATURES].copy()
    require(not features.isna().any().any(), "模型特征存在缺失，停止处理。")
    for column in ["recency", "history"] + BINARY_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="raise")
        require(np.isfinite(features[column].to_numpy()).all(), f"{column}含无穷值。")
    require(features["history"].ge(0).all(), "history不能为负数。")
    require(features["recency"].between(1, 12).all(), "recency应在1到12月之间。")
    require(features["recency"].eq(np.floor(features["recency"])).all(), "recency应为整数月份。")
    for column in BINARY_FEATURES:
        require(features[column].isin([0, 1]).all(), f"{column}必须为0/1。")
    for column in CATEGORICAL_FEATURES:
        require(features[column].map(lambda value: isinstance(value, str) and len(value) > 0).all(),
                f"{column}应为非空文本类别。")
    features["log_history"] = np.log1p(features["history"])
    return features[NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES]


def make_design(z, t, interaction_indices):
    """Z不含T；设定情景后同时更新T和全部预定交互项。"""
    z = np.asarray(z, dtype=float)
    require(z.ndim == 2 and np.isfinite(z).all(), "Z应为有限数值二维矩阵。")
    indices = np.asarray(interaction_indices)
    require(indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer), "交互项索引应为一维整数。")
    require(len(set(indices.tolist())) == len(indices), "交互项索引重复。")
    require(bool(((indices >= 0) & (indices < z.shape[1])).all()), "交互项索引越界。")
    treatment = np.asarray(t, dtype=float)
    if treatment.ndim == 0:
        treatment = np.full(len(z), float(treatment))
    require(treatment.ndim == 1 and len(treatment) == len(z), "T长度与Z不同。")
    require(np.isin(treatment, [0, 1]).all(), "T只能为0/1。")
    return np.column_stack([z, treatment, z[:, indices] * treatment[:, None]])


def fit_ranker(training):
    require({"T", "conversion"}.issubset(training.columns), "训练数据缺少T或购买结果。")
    require(training["T"].isin([0, 1]).all() and set(training["T"].unique()) == {0, 1}, "训练T编码异常。")
    require(training["conversion"].isin([0, 1]).all() and training["conversion"].nunique() == 2,
            "训练购买结果必须同时包含0和1。")
    features = prepare_features(training)
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("binary", "passthrough", BINARY_FEATURES),
        ("cat", OneHotEncoder(drop="first", handle_unknown="error", sparse_output=False), CATEGORICAL_FEATURES),
    ], remainder="drop", verbose_feature_names_out=True)
    z = preprocessor.fit_transform(features)
    names = preprocessor.get_feature_names_out().tolist()
    require(all(name in names for name in INTERACTION_FEATURES), "交互项与变换后的特征不对应。")
    indices = [names.index(name) for name in INTERACTION_FEATURES]
    design_names = names + ["T"] + [f"T:{name}" for name in INTERACTION_FEATURES]
    design = make_design(z, training["T"].to_numpy(), indices)
    model = LogisticRegression(**MODEL_PARAMETERS)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(design, training["conversion"].to_numpy(dtype=int))
    except ConvergenceWarning as error:
        raise ValueError("优化未收敛；本次停止，不自动修改参数或尝试多个模型。") from error
    require(np.array_equal(model.classes_, [0, 1]), "模型类别顺序异常。")
    require(np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all(), "拟合系数非有限数。")
    require(int(model.n_iter_.max()) < MODEL_PARAMETERS["max_iter"], "模型迭代达到上限。")
    return {
        "bundle_version": 1, "configuration": CONFIGURATION,
        "preprocessor": preprocessor, "model": model,
        "feature_names": names, "interaction_indices": indices,
        "design_feature_names": design_names,
        "training_rows": len(training), "training_buyers": int(training["conversion"].sum()),
    }


def score_ranker(bundle, frame):
    """同一客户X分别设为不发/发邮件情景；实际T和结果不会影响打分。"""
    features = prepare_features(frame)
    z = bundle["preprocessor"].transform(features)
    model, indices = bundle["model"], bundle["interaction_indices"]
    require(np.array_equal(model.classes_, [0, 1]), "模型购买类别索引异常。")
    p0 = model.predict_proba(make_design(z, 0, indices))[:, 1]
    p1 = model.predict_proba(make_design(z, 1, indices))[:, 1]
    require(np.isfinite(p0).all() and np.isfinite(p1).all(), "预测概率非有限数。")
    require(((p0 >= 0) & (p0 <= 1) & (p1 >= 0) & (p1 <= 1)).all(), "预测概率越界。")
    return pd.DataFrame({"p0": p0, "p1": p1, "predicted_uplift": p1 - p0,
                         "predicted_uplift_pp": 100 * (p1 - p0)}, index=frame.index)


def coefficient_table(bundle):
    names = ["intercept"] + bundle["design_feature_names"]
    coefficients = np.r_[bundle["model"].intercept_, bundle["model"].coef_[0]]
    require(len(names) == len(coefficients), "系数与特征名数量不同。")
    return pd.DataFrame({"term": names, "coefficient_log_odds": coefficients,
                         "note": "coefficient from L2 logistic model; no formal significance test"})


def fitted_preprocessing_summary(bundle):
    preprocessor = bundle["preprocessor"]
    scaler = preprocessor.named_transformers_["num"]
    encoder = preprocessor.named_transformers_["cat"]
    return {
        "numeric": {name: {"training_mean": float(scaler.mean_[i]), "training_scale": float(scaler.scale_[i])}
                    for i, name in enumerate(NUMERIC_FEATURES)},
        "binary_passthrough": BINARY_FEATURES,
        "categorical": {name: {"training_categories": encoder.categories_[i].tolist(),
                               "reference_category": str(encoder.categories_[i][int(encoder.drop_idx_[i])])}
                        for i, name in enumerate(CATEGORICAL_FEATURES)},
        "transformed_feature_names": bundle["feature_names"],
        "design_feature_names": bundle["design_feature_names"],
        "interaction_indices": bundle["interaction_indices"],
    }


def load_dependency(project_dir, name, import_name):
    path = project_dir / name
    require(path.is_file(), f"缺少{name}；请把07与已有04、05脚本放在项目根目录。")
    specification = importlib.util.spec_from_file_location(import_name, path)
    require(specification is not None and specification.loader is not None, f"无法载入{name}。")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module, path


def record_configuration(output_dir, provenance):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "07_model_specification.json"
    config_hash = stable_hash(CONFIGURATION)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8-sig"))
        require(existing["configuration"] == CONFIGURATION, "模型设置已改变，请保留旧版本并记录新尝试。")
        require(existing["configuration_sha256"] == config_hash, "模型设置记录校验不一致。")
        for key, value in provenance.items():
            require(existing[key] == value, "模型记录对应的数据、划分或依赖脚本发生变化。")
    else:
        existing = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "configuration": CONFIGURATION, "configuration_sha256": config_hash,
                    **provenance,
                    "note": "已查看此前训练阶段结果；本轮新增模型设计，不冒充未看数据的预注册。"}
        with path.open("x", encoding="utf-8") as handle:
            json.dump(existing, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return {"configuration_sha256": config_hash, "configuration_record_sha256": digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else project_dir / "reports"
    stage04, path04 = load_dependency(project_dir, "04_train_balance.py", "hillstrom_stage04_ranker_dependency")
    stage05, path05 = load_dependency(project_dir, "05_train_subgroups.py", "hillstrom_stage05_ranker_dependency")
    training, provenance = stage04.load_training_covariates(project_dir)
    provenance.update({"stage04_script_sha256": digest(path04), "stage05_script_sha256": digest(path05)})
    configuration_provenance = record_configuration(output_dir, provenance)
    train_ids = set(training["source_row_proxy"].astype(int))
    raw_file = project_dir / "data" / "raw" / "hillstrom.csv"
    training = stage05.attach_training_outcomes(training, raw_file, train_ids)
    bundle = fit_ranker(training)
    bundle["provenance"] = {**provenance, **configuration_provenance}
    bundle["runtime"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                         "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}
    scores = score_ranker(bundle, training)
    demo = training[["source_row_proxy"]].join(scores)
    demo["analysis_set"] = "train"
    demo["use"] = "in-sample score demonstration; not a final targeting list"

    model_file = output_dir / "07_ranker.joblib"
    joblib.dump(bundle, model_file, compress=3)
    # 校验刚保存的本地模型可重新载入且产生相同分数。
    restored = joblib.load(model_file)
    restored_scores = score_ranker(restored, training)
    require(np.array_equal(scores.to_numpy(), restored_scores.to_numpy()), "保存/重新载入后预测发生变化。")
    demo_file = output_dir / "07_train_scores.csv"
    demo.to_csv(demo_file, index=False, encoding="utf-8-sig")
    coefficient_table(bundle).to_csv(output_dir / "07_model_coefficients.csv", index=False, encoding="utf-8-sig")
    report = {
        "analysis_set": "train", "train_rows": len(training),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": CONFIGURATION,
        "arm_counts": {str(t): int((training["T"] == t).sum()) for t in [0, 1]},
        "arm_buyers": {str(t): int(training.loc[training["T"] == t, "conversion"].sum()) for t in [0, 1]},
        "transformed_feature_count": len(bundle["feature_names"]),
        "design_column_count": len(bundle["design_feature_names"]),
        "fitted_parameter_count_including_intercept": len(bundle["design_feature_names"]) + 1,
        "fit_iterations": int(bundle["model"].n_iter_.max()),
        "fitted_preprocessing": fitted_preprocessing_summary(bundle),
        "training_score_summary": {
            "predicted_uplift_mean_pp": float(scores["predicted_uplift_pp"].mean()),
            "negative_predicted_uplift_rows": int(scores["predicted_uplift"].lt(0).sum()),
            "note": "拟合样本分数的描述，不是留出集效果或模型质量评价。",
        },
        "notes": [
            "一个共同的概率模型给出p0和p1；购买概率排序用p1，增量排序用p1-p0。",
            "只用训练X拟合标准化和类别编码，只用训练Y拟合逻辑回归；未读取验证/测试特征或结果。",
            "全体segment仅读取用于来源与冻结划分核验，不作为模型特征。",
            "原始行号只是离线代理，不是CRM身份；未按相同属性去重。",
            "固定三项处理交互及L2强度，没有训练内搜参，没有根据验证/测试成绩调整。",
            "未做类别平衡加权、过采样或以0.5阈值分类，保留购买概率建模含义。",
            "预测增量是模型估计的条件平均效应分数，不是已观测的个人真实因果效应。",
            "惩罚回归系数不附未经说明的显著性检验；交互系数位于log-odds尺度。",
            "训练分数只用于学习与检查，不能据此声称增量排序优于购买概率或随机触达。",
            "后续等容量比较为floor(q*N)；同分按原始行号升序，不把负增量分数自动当作触达许可。",
            "joblib保存的是本地训练的模型；可重复加载需保留相应代码与依赖版本。",
        ],
        "method_references": [
            "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
            "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html",
        ],
        "runtime": {"executable": sys.executable, **bundle["runtime"]},
        "script_sha256": digest(Path(__file__).resolve()),
        "model_sha256": digest(model_file), "training_scores_sha256": digest(demo_file),
        **provenance, **configuration_provenance,
    }
    (output_dir / "07_training_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("第七步完成：客户排序模型已训练并保存。")
    print(f"训练集：{len(training):,}人；T0={(training['T'] == 0).sum():,}，T1={(training['T'] == 1).sum():,}")
    print(f"模型：L2逻辑回归，预定3项交互；优化迭代{report['fit_iterations']}次。")
    print("模型保存及重新载入校验通过；没有读取验证/测试特征或购买结果。")
    print("\n先看下面3条训练样本，按原始行号展示，未按分数挑选：")
    for row in demo.head(3).to_dict(orient="records"):
        print(f"\n原始行代理 {row['source_row_proxy']}：")
        print(f"  不发邮件预测购买率p0：{row['p0']:.3%}")
        print(f"  发邮件预测购买率p1：{row['p1']:.3%}")
        print(f"  预测增量p1-p0：{row['predicted_uplift_pp']:+.3f}个百分点")
    print("\n购买概率排序按p1；增量排序按p1-p0。")
    print("以上是训练分数演示；策略价值留到验证集比较，本步没有选出最终名单。")
    print(f"结果保存到：{output_dir}")
    print("请先截图“第七步完成”及第一条样本的p0、p1、预测增量，我们再解读。")


if __name__ == "__main__":
    main()
