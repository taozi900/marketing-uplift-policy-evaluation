r"""第二张图：20%容量下，增量排序与三种基线的成对差值森林图。

把脚本放到marketing_project_starter根目录，使用项目.venv运行即可。
沿用已安装的matplotlib；仅读取阶段09汇总报告，不重新训练或估计。
默认图片保存至 reports/figures。
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


BASELINES = [
    ("p1_descending", "购买概率排序", "主要比较"),
    ("random", "固定随机名单", "辅助探索"),
    ("history_descending", "历史消费排序", "辅助探索"),
]
UNIT = "每万名合格客户的新增购买人数之差（离线估计，人）"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"报告包含非有限数值：{key}")
    return value


def load_inputs(project: Path):
    files = {
        "differences": project / "reports/09_test_policy_differences.csv",
        "results": project / "reports/09_test_policy_results.csv",
        "specification": project / "reports/09_test_specification.json",
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(
                f"缺少报告：{path}。请将脚本放在项目根目录，或指定--project-dir。")
    hashes = {key: sha256(path) for key, path in files.items()}
    rows = [r for r in read_rows(files["differences"])
            if math.isclose(float(r["budget_fraction"]), .2, abs_tol=1e-12)]
    expected = {key for key, _, _ in BASELINES}
    if len(rows) != 3 or {r["baseline"] for r in rows} != expected:
        raise ValueError("20%容量应包含增量排序与三个基线的成对比较。")
    if any(r["analysis_set"] != "test"
           or r["candidate"] != "predicted_uplift_descending" for r in rows):
        raise ValueError("必须使用保留测试集的增量排序差值报告。")
    by_baseline = {r["baseline"]: r for r in rows}
    ordered = [by_baseline[key] for key, _, _ in BASELINES]
    for row, (key, _, _) in zip(ordered, BASELINES):
        required_role = "primary" if key == "p1_descending" else "auxiliary_exploratory"
        if row["comparison_role"] != required_role:
            raise ValueError("主要比较与辅助探索的角色不符合冻结方案。")
        point = number(row, "difference_per_10000_eligible")
        lo = number(row, "ci_low_per_10000_eligible")
        hi = number(row, "ci_high_per_10000_eligible")
        if not lo <= point <= hi:
            raise ValueError("区间不包含点估计，请先核对原报告。")
    policy_rows = [r for r in read_rows(files["results"])
                   if math.isclose(float(r["budget_fraction"]), .2, abs_tol=1e-12)]
    if (len(policy_rows) != 4
        or {r["policy"] for r in policy_rows} != expected | {"predicted_uplift_descending"}
        or any(r["analysis_set"] != "test" for r in policy_rows)):
        raise ValueError("四种策略的测试集结果不完整。")
    counts = {int(r["eligible_count"]) for r in policy_rows}
    selected = {int(r["selected_count"]) for r in policy_rows}
    if len(counts) != 1 or len(selected) != 1:
        raise ValueError("四种策略没有相同的评价记录数量和触达容量。")
    n, k = counts.pop(), selected.pop()
    if n <= 0 or k != math.floor(.2 * n):
        raise ValueError("容量不是floor(20% * N)。")
    policy_values = {r["policy"]: number(r, "incremental_per_10000_eligible")
                     for r in policy_rows}
    for row in ordered:
        expected_diff = (policy_values["predicted_uplift_descending"]
                         - policy_values[row["baseline"]])
        if not math.isclose(number(row, "difference_per_10000_eligible"),
                            expected_diff, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("报告差值与两种策略点估计之差不一致。")
    spec = json.loads(files["specification"].read_text(encoding="utf-8"))["payload"]
    rules = spec["inherited_rules"]
    if (spec["analysis_set"] != "test"
        or rules["primary_capacity"] != .2
        or rules["primary_contrast"]["candidate"] != "predicted_uplift_descending"
        or rules["primary_contrast"]["baseline"] != "p1_descending"):
        raise ValueError("评估协议与绘图方案不一致。")
    return ordered, n, k, files, hashes, rules


def setup_plotting():
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt, font_manager
    except ImportError as exc:
        raise SystemExit("请在项目虚拟环境安装matplotlib后再运行。") from exc
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    ]
    chosen = next((p for p in candidates if p.is_file()), None)
    if chosen is None:
        raise FileNotFoundError("请在setup_plotting()中配置支持中文的字体路径。")
    font_manager.fontManager.addfont(str(chosen))
    family = font_manager.FontProperties(fname=str(chosen)).get_name()
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False,
                         "svg.fonttype": "path", "font.size": 12,
                         "savefig.facecolor": "white"})
    warnings.filterwarnings("error", message=r"Glyph .* missing from font")
    return matplotlib, plt


def draw(rows, n, k, rules, output, dpi):
    matplotlib, plt = setup_plotting()
    from matplotlib.patches import FancyBboxPatch

    ink, muted = "#19313E", "#5D707B"
    primary_color, auxiliary_color = "#147D73", "#788D9A"
    fig = plt.figure(figsize=(13.6, 8.2), facecolor="white")
    fig.text(.055, .931, "增量排序与三种策略的差值", fontsize=23,
             weight="bold", color=ink)
    fig.text(.055, .875,
             f"冻结测试集 {n:,} 条记录  |  相同20%触达容量，每策略选 {k:,} 条  |  结果：14天是否购买",
             fontsize=11.5, color=muted)
    cross_zero = all(number(r, "ci_low_per_10000_eligible") <= 0
                     <= number(r, "ci_high_per_10000_eligible") for r in rows)
    if cross_zero:
        conclusion = "三条差值区间均跨过0，尚不足以确认增量排序优于这些基线。"
    else:
        conclusion = "逐项比较点估计与区间；主要比较和辅助探索分别报告。"
    fig.add_artist(FancyBboxPatch((.05, .772), .90, .062,
                                  boxstyle="round,pad=0.009,rounding_size=.012",
                                  transform=fig.transFigure, facecolor="#EEF6F5",
                                  edgecolor="none", zorder=-1))
    fig.text(.07, .794, conclusion, fontsize=12, color=primary_color)
    fig.text(.055, .72, "比较方式：增量排序减去相应基线", fontsize=10.5, color=muted)
    fig.text(.758, .72, "差值估计", fontsize=10.5, color=muted, ha="center")
    fig.text(.886, .72, "成对95%区间", fontsize=10.5, color=muted, ha="center")

    ax = fig.add_axes([.345, .315, .355, .365])
    ax.set_xlim(-25, 45)
    ax.set_ylim(-.6, 2.6)
    ax.axvline(0, color="#6B7E89", lw=1.5, ls=(0, (4, 4)), zorder=1)
    ax.axhspan(1.62, 2.38, color="#F0F7F5", zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#E3EBEE", lw=.8)
    for i, (row, (_, label, role)) in enumerate(zip(rows, BASELINES)):
        y = 2 - i
        point = number(row, "difference_per_10000_eligible")
        lo = number(row, "ci_low_per_10000_eligible")
        hi = number(row, "ci_high_per_10000_eligible")
        color = primary_color if i == 0 else auxiliary_color
        ax.errorbar(point, y, xerr=[[point - lo], [hi - point]],
                    fmt="D" if i == 0 else "o", markersize=9, color=color,
                    ecolor=color, capsize=5, elinewidth=2.5, capthick=1.7, zorder=3)
        fy = fig.transFigure.inverted().transform(ax.transData.transform((0, y)))[1]
        fig.text(.055, fy + .012, "增量排序 - " + label, fontsize=12.2, color=ink,
                 va="center", weight="bold" if i == 0 else "normal")
        fig.text(.055, fy - .023, role, fontsize=10, color=color, va="center")
        fig.text(.758, fy, f"{point:+.2f}", fontsize=14, ha="center", va="center",
                 color=color, weight="bold" if i == 0 else "normal")
        fig.text(.886, fy, f"[{lo:+.2f}, {hi:+.2f}]", fontsize=11.5,
                 ha="center", va="center", color=ink)
    ax.set_yticks([])
    ax.set_xticks([-20, -10, 0, 10, 20, 30, 40])
    ax.tick_params(axis="x", length=0, colors=muted, labelsize=11, pad=9)
    for side in ["top", "left", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#CAD7DD")
    ax.set_xlabel("每万名合格客户的新增购买人数之差\n（离线估计，人）",
                  fontsize=11.5, color=muted, labelpad=13)
    fig.text(.345, .195, "负值：更有利于相应基线", fontsize=9.8, color=muted)
    fig.text(.70, .195, "正值：更有利于增量排序", fontsize=9.8,
             color=muted, ha="right")
    # Direction labels sit below the x label; reserve enough breathing space.
    fig.text(.055, .153, "读图：圆点或菱形表示差值估计；横线表示成对区间；竖虚线表示两策略差值为0。",
             fontsize=10.5, color=muted)
    repeats = rules["bootstrap"]["repeats"]
    fig.text(.055, .112,
             f"来源：阶段09已保存的差值报告；{repeats:,}次成对分层Bootstrap；逐项95%区间，辅助比较未作多重比较调整。",
             fontsize=9.1, color=muted)
    fig.text(.055, .079,
             "固定模型与名单下的近似评价，未包含重训和重排的不确定性；区间跨0不代表效果相同。",
             fontsize=9.1, color=muted)
    fig.text(.055, .046,
             "购买按客户在活动后14天内是否购买记录；这里统计购买人数之差，而非订单笔数。",
             fontsize=9.1, color=muted)
    output.mkdir(parents=True, exist_ok=True)
    png = output / "12_20pct_policy_differences.png"
    svg = output / "12_20pct_policy_differences.svg"
    fig.savefig(png, dpi=dpi)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg, matplotlib.__version__


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
    rows, n, k, files, hashes, rules = load_inputs(project)
    png, svg, version = draw(rows, n, k, rules, output, args.dpi)
    for key, path in files.items():
        if sha256(path) != hashes[key]:
            raise RuntimeError(f"来源报告在作图期间发生变化：{path}")
    manifest = {
        "analysis_set": "test", "capacity": .2, "eligible_records": n,
        "selected_per_policy": k, "unit": UNIT,
        "role": "Display of saved paired comparisons, without re-estimation or new policy selection.",
        "rows": rows, "interval": rules["bootstrap"],
        "sources": {key: {"path": str(path), "sha256": hashes[key]}
                    for key, path in files.items()},
        "script_sha256": sha256(Path(__file__).resolve()),
        "runtime": {"python": sys.version.split()[0], "matplotlib": version},
        "outputs": {"png": {"file": png.name, "sha256": sha256(png)},
                    "svg": {"file": svg.name, "sha256": sha256(svg)}},
    }
    (output / "12_plot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"png": str(png), "svg": str(svg),
                      "source_reports_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
