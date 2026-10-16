r"""第四张图：四种策略在5%、10%、20%、30%预设容量下的增量与区间。

放到项目根目录，选择已有.venv，点击VS Code右上角运行按钮即可。
只读取阶段09已经保存的汇总CSV和协议，不重新训练或计算新效果。
沿用matplotlib依赖。默认输出到 reports/figures。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import warnings


CAPACITIES = [.05, .10, .20, .30]
POLICIES = [
    ("random", "固定随机名单", "#7B8F9A"),
    ("history_descending", "历史消费排序", "#9A7450"),
    ("p1_descending", "购买概率排序", "#426D96"),
    ("predicted_uplift_descending", "预测增量排序", "#147D73"),
]
UNIT = "每万名合格客户的新增购买人数估计（离线估计，人）"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"报告包含非有限值：{key}")
    return value


def load_inputs(project: Path):
    files = {
        "policy_results": project / "reports/09_test_policy_results.csv",
        "specification": project / "reports/09_test_specification.json",
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(
                f"缺少阶段09报告：{path}。请将脚本放在项目根目录或指定--project-dir。")
    hashes = {key: sha256(path) for key, path in files.items()}
    with files["policy_results"].open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(q, policy) for q in CAPACITIES for policy, _, _ in POLICIES}
    actual = {(float(r["budget_fraction"]), r["policy"]) for r in rows}
    if len(rows) != 16 or actual != expected:
        raise ValueError("需恰好包含四种策略、四个预设容量的16项结果。")
    if any(r["analysis_set"] != "test" for r in rows):
        raise ValueError("绘图输入必须都是保留测试集结果。")
    by_key = {(float(r["budget_fraction"]), r["policy"]): r for r in rows}
    sizes = {int(r["eligible_count"]) for r in rows}
    if len(sizes) != 1:
        raise ValueError("各容量和策略必须使用同一大小的合格客户集合。")
    n = sizes.pop()
    if n <= 0:
        raise ValueError("合格客户记录数必须为正。")
    for row in rows:
        q = float(row["budget_fraction"])
        role = "primary_capacity" if q == .2 else "exploratory_capacity"
        if row["analysis_role"] != role:
            raise ValueError("主要容量与辅助探索的标记不符合冻结方案。")
        k = int(row["selected_count"])
        if k != math.floor(q * n):
            raise ValueError("所选记录数不符合floor(q*N)规则。")
        if int(row["selected_T0"]) + int(row["selected_T1"]) != k:
            raise ValueError("实验两组所选记录数之和与总人数不一致。")
        point = number(row, "incremental_per_10000_eligible")
        lo = number(row, "ci_low_per_10000_eligible")
        hi = number(row, "ci_high_per_10000_eligible")
        if not lo <= point <= hi:
            raise ValueError("本绘图方式要求区间包含点估计，请先检查原报告。")
        count_estimate = 10000 * (2 * int(row["selected_buyers_T1"])
                                  - 2 * int(row["selected_buyers_T0"])) / n
        if not math.isclose(point, count_estimate, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("增量估计与报告内购买计数不一致。")
    payload = json.loads(files["specification"].read_text(encoding="utf-8"))["payload"]
    rules = payload["inherited_rules"]
    if (payload["analysis_set"] != "test"
        or rules["primary_capacity"] != .2
        or rules["propensity"] != .5
        or set(rules["capacities"]) != set(CAPACITIES)
        or set(rules["policies"]) != {p for p, _, _ in POLICIES}):
        raise ValueError("评估协议与四策略、四容量的展示方案不一致。")
    return rows, by_key, n, files, hashes, rules


def setup_plotting():
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt, font_manager
    except ImportError as exc:
        raise SystemExit("请使用已经安装matplotlib的项目.venv运行。") from exc
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    ]
    chosen = next((path for path in candidates if path.is_file()), None)
    if chosen is None:
        raise FileNotFoundError("请在setup_plotting()中设置中文字体路径。")
    font_manager.fontManager.addfont(str(chosen))
    family = font_manager.FontProperties(fname=str(chosen)).get_name()
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False,
                         "svg.fonttype": "path", "font.size": 11,
                         "savefig.facecolor": "white"})
    warnings.filterwarnings("error", message=r"Glyph .* missing from font")
    return matplotlib, plt


def draw(by_key, n, output: Path, dpi: int):
    matplotlib, plt = setup_plotting()
    from matplotlib.patches import FancyBboxPatch

    ink, muted = "#19313E", "#60727D"
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 10.4), sharex=True, sharey=True,
                             facecolor="white")
    fig.subplots_adjust(left=.11, right=.955, bottom=.205, top=.755,
                        wspace=.18, hspace=.45)
    fig.text(.055, .957, "不同触达比例下，四种策略的增量估计", fontsize=23,
             weight="bold", color=ink)
    fig.text(.055, .911,
             f"冻结测试集 {n:,} 条客户记录  |  同一14天购买窗口  |  四个小图使用相同坐标范围",
             fontsize=11.5, color=muted)
    fig.text(.055, .872, "每策略入选", fontsize=10.8, color=muted, va="center")
    for i, q in enumerate(CAPACITIES):
        x = .20 + i * .19
        main = q == .2
        fig.add_artist(FancyBboxPatch(
            (x, .854), .165, .036,
            boxstyle="round,pad=0.005,rounding_size=.009",
            transform=fig.transFigure,
            facecolor="#E9F5EF" if main else "#F0F4F6", edgecolor="none", zorder=-1))
        fig.text(x + .0825, .872, f"{q:.0%}：{math.floor(q * n):,} 条",
                 ha="center", va="center", fontsize=11.4,
                 color="#147D73" if main else ink,
                 weight="bold" if main else "normal")
    fig.text(.11, .815,
             "菱形与浅色带：20%主要容量　　圆点：其余探索容量　　竖向误差棒：逐项95%区间",
             fontsize=10.3, color=muted)

    all_rows = list(by_key.values())
    low = min(number(r, "ci_low_per_10000_eligible") for r in all_rows)
    high = max(number(r, "ci_high_per_10000_eligible") for r in all_rows)
    ymin = min(-10, math.floor((low - 5) / 10) * 10)
    ymax = max(10, math.ceil((high + 5) / 10) * 10)
    x_values = [q * 100 for q in CAPACITIES]
    for ax, (policy, label, color) in zip(axes.flat, POLICIES):
        values = [number(by_key[(q, policy)], "incremental_per_10000_eligible")
                  for q in CAPACITIES]
        lows = [number(by_key[(q, policy)], "ci_low_per_10000_eligible")
                for q in CAPACITIES]
        highs = [number(by_key[(q, policy)], "ci_high_per_10000_eligible")
                 for q in CAPACITIES]
        ax.axvspan(18.9, 21.1, color="#EDF6F1", zorder=0)
        ax.axvline(20, color="#B6CCC0", lw=1.0, ls=(0, (3, 3)), zorder=1)
        ax.axhline(0, color="#899AA5", lw=1.0, ls=(0, (4, 4)), zorder=1)
        ax.plot(x_values, values, color=color, lw=1.8, alpha=.8, zorder=2)
        for q, x, point, lo, hi in zip(CAPACITIES, x_values, values, lows, highs):
            ax.errorbar(x, point, yerr=[[point - lo], [hi - point]],
                        fmt="D" if q == .2 else "o", markersize=7 if q == .2 else 6,
                        color=color, ecolor=color, elinewidth=1.7,
                        capsize=4, capthick=1.4, zorder=3)
            ax.annotate(f"{point:+.2f}", (x, point), xytext=(6, 7),
                        textcoords="offset points", ha="left", va="bottom",
                        fontsize=10, color=color, weight="bold" if q == .2 else "normal",
                        bbox={"facecolor": "white", "alpha": .88,
                              "edgecolor": "none", "pad": .6})
        ax.set_title(label, fontsize=14, color=color, loc="left", weight="bold", pad=12)
        ax.set_xlim(2, 35)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(x_values, [f"{q:.0%}" for q in CAPACITIES])
        ax.set_yticks(list(range(int(ymin), int(ymax) + 1, 20)))
        ax.tick_params(axis="both", length=0, colors=muted, labelsize=10, pad=7,
                       labelbottom=True)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#E7EDF0", lw=.8)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#CFDAE0")
    fig.text(.035, .48, "每万名合格客户的新增购买人数估计（人）",
             rotation=90, va="center", ha="center", fontsize=11.7, color=muted)
    fig.text(.5325, .139, "预设触达比例（人数按比例向下取整）",
             ha="center", fontsize=11.5, color=muted)
    fig.text(.055, .091,
             "指标相对“所有人不发邮件”，按全部合格客户计；点旁数字为点估计，连线仅帮助阅读四个已评价比例。",
             fontsize=9.4, color=muted)
    fig.text(.055, .061,
             "区间条件于固定模型与名单，不承诺同时覆盖；小容量下购买事件稀少，区间需谨慎解释。",
             fontsize=9.4, color=muted)
    fig.text(.055, .031,
             "未评价容量间的成对差值，也未纳入成本和利润，不能据最高点认定最优比例。来源：阶段09冻结测试报告。",
             fontsize=9.4, color=muted)

    output.mkdir(parents=True, exist_ok=True)
    png = output / "14_capacity_comparison.png"
    svg = output / "14_capacity_comparison.svg"
    fig.savefig(png, dpi=dpi)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg, matplotlib.__version__


def write_notes(output: Path, by_key, n):
    lines = [
        "# 不同触达比例下的策略比较", "",
        "这张图将四种策略分别放入四个小图，使用相同横纵坐标，展示阶段09已经保存的点估计与逐项95% Bootstrap区间。", "",
        "## 怎么读", "",
        "- 横轴是预先设定的5%、10%、20%、30%触达比例；20%是主要容量，其余为辅助探索。实际入选人数使用floor(q*N)。",
        "- 纵轴是相对所有人不发邮件、每万名合格客户的新增购买人数估计；分母包括所有合格客户，并非只有被触达的客户。",
        "- 圆点/菱形旁的数字为点估计，竖线是区间；连线仅帮助阅读，不能当成完整Qini曲线或其他容量的实测结果。",
        "- 小容量下事件稀少；例如随机5%所选客户中，T0为0次购买、T1为1次购买，非参Bootstrap区间的零端点不能证明没有负向可能。", "",
        "## 点估计对照表", "",
        "单位：每万名合格客户的新增购买人数（人）；各数均为离线估计。", "",
        "| 预设比例 | 每策略入选数 | 随机名单 | 历史消费排序 | 购买概率排序 | 预测增量排序 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for q in CAPACITIES:
        values = [number(by_key[(q, p)], "incremental_per_10000_eligible")
                  for p, _, _ in POLICIES]
        lines.append(f"| {q:.0%} | {math.floor(q * n):,} | "
                     + " | ".join(f"{v:+.2f}" for v in values) + " |")
    lines.extend([
        "", "## 可以得出的观察", "",
        "本次增量排序在5%到30%容量下的总增量点估计逐步升高；小容量下历史消费规则的点估计也较突出。这是本次样本中的观察，并非普遍规律。", "",
        "## 不能由图直接推断的结论", "",
        "- 没有成本和利润信息，不能确定利润最优容量或声称ROI提升。",
        "- 现有报告没有容量间的成对差值区间；不能凭单项区间或点的高低确认扩大容量一定更好。",
        "- 各容量下同一策略的名单有重叠，不能把四个容量当作四次独立实验。",
        "- 逐项95%区间不等于多项同时覆盖，也不包含模型训练、重新排序等全部流程的不确定性。",
        "- 保持原有20%主要比较；作图后不据测试结果重选模型、种子或最优容量。", "",
        "## 复现", "",
        "将14_plot_capacity_comparison.py放到项目根目录，使用已有.venv环境和matplotlib，点击VS Code右上角运行即可。图像默认保存到reports/figures。脚本只读取已保存的汇总报告，未重新训练或评价逐条测试结果。", "",
    ])
    path = output / "14_触达比例比较图说明.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if not 120 <= args.dpi <= 600:
        raise ValueError("DPI应在120到600之间。")
    project = args.project_dir.resolve()
    output = (args.output_dir or project / "reports/figures").resolve()
    rows, by_key, n, files, hashes, rules = load_inputs(project)
    png, svg, version = draw(by_key, n, output, args.dpi)
    for key, path in files.items():
        if sha256(path) != hashes[key]:
            raise RuntimeError(f"来源报告在作图期间发生变化：{path}")
    notes = write_notes(output, by_key, n)
    csv_path = output / "14_capacity_plot_data.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(by_key[(q, policy)] for q in CAPACITIES for policy, _, _ in POLICIES)
    manifest = {
        "figure_type": "four_prespecified_capacities_by_four_fixed_policies",
        "analysis_set": "test", "capacities": CAPACITIES, "primary_capacity": .2,
        "eligible_records": n, "unit": UNIT,
        "selected_counts": {f"{q:.0%}": math.floor(q * n) for q in CAPACITIES},
        "no_new_model_or_capacity_selection": True,
        "role": "Display saved point estimates and pointwise intervals; no comparisons between capacities estimated.",
        "interval": rules["bootstrap"],
        "sources": {key: {"path": str(path), "sha256": hashes[key]}
                    for key, path in files.items()},
        "script_sha256": sha256(Path(__file__).resolve()),
        "runtime": {"python": sys.version.split()[0], "matplotlib": version},
        "outputs": {path.name: sha256(path) for path in (png, svg, notes, csv_path)},
    }
    (output / "14_plot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"png": str(png), "svg": str(svg), "readme": str(notes),
                      "source_reports_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
