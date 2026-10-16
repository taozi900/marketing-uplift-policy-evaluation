r"""第三张图：四类潜在响应人群的概念示意，适合作品集中的业务解释页。

使用项目已有的matplotlib即可，不需要安装新库。
放在项目根目录后，选择.venv环境，点击VS Code右上角运行按钮。
默认输出 reports/figures/13_customer_response_types.png 和同名SVG。

重要：图中为四种理论潜在响应组合，未对真实客户分类或计算人数。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings


SOURCES = [
    "https://causalml.readthedocs.io/en/latest/methodology.html#value-optimization-methods",
    "https://www.uplift-modeling.com/en/v0.3.0/user_guide/introduction/clients.html",
]

TYPES = [
    {
        "key": "persuadables", "title": "可促成购买", "english": "Persuadables",
        "Y0": 0, "Y1": 1,
        "statement": "不发不买，发了才买",
        "effect": "营销作用：增加购买",
        "implication": "正增量类型，是营销触达的理论关注对象",
        "accent": "#117A63", "background": "#EAF6F0",
    },
    {
        "key": "sure_things", "title": "自然购买", "english": "Sure Things",
        "Y0": 1, "Y1": 1,
        "statement": "不发也买，发了仍买",
        "effect": "营销作用：不改变是否购买",
        "implication": "客户会购买，但本次营销没有增加购买人数",
        "accent": "#476F8D", "background": "#EEF4F9",
    },
    {
        "key": "lost_causes", "title": "本次均不购买", "english": "Lost Causes",
        "Y0": 0, "Y1": 0,
        "statement": "不发不买，发了也不买",
        "effect": "营销作用：不改变是否购买",
        "implication": "仅针对这次活动与观察窗口，并非永久标签",
        "accent": "#63747F", "background": "#F1F4F6",
    },
    {
        "key": "sleeping_dogs", "title": "负向响应", "english": "Sleeping Dogs",
        "Y0": 1, "Y1": 0,
        "statement": "不发会买，发了反而不买",
        "effect": "营销作用：减少购买",
        "implication": "理论上的负增量情形，具体原因不能由此确定",
        "accent": "#AB5746", "background": "#FFF0EA",
    },
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt, font_manager
    except ImportError as exc:
        raise SystemExit("请使用已经安装matplotlib的项目.venv环境运行。") from exc
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    ]
    chosen = next((p for p in candidates if p.is_file()), None)
    if chosen is None:
        raise FileNotFoundError("请在prepare_matplotlib()中配置中文字体路径。")
    font_manager.fontManager.addfont(str(chosen))
    family = font_manager.FontProperties(fname=str(chosen)).get_name()
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False,
                         "svg.fonttype": "path", "font.size": 12,
                         "savefig.facecolor": "white"})
    warnings.filterwarnings("error", message=r"Glyph .* missing from font")
    return matplotlib, plt


def draw(output: Path, dpi: int):
    if {(r["Y0"], r["Y1"]) for r in TYPES} != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError("四格应完整覆盖两个二元潜在结果的全部组合。")
    matplotlib, plt = prepare_matplotlib()
    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ink, muted, teal = "#19313E", "#5F707B", "#117A63"
    ax.text(.055, .936, "营销增量的四种潜在响应情形", fontsize=26,
            color=ink, weight="bold", va="center")
    ax.text(.055, .883,
            "先区分“本来就会买”和“因营销而购买”，再讨论营销资源应该给谁。",
            fontsize=13, color=muted, va="center")
    ax.add_patch(FancyBboxPatch((.834, .914), .115, .041,
                                boxstyle="round,pad=0.007,rounding_size=.013",
                                facecolor="#F0F5F7", edgecolor="none"))
    ax.text(.8915, .9345, "概念示意", fontsize=12, color=muted,
            ha="center", va="center")

    x_positions = {0: .235, 1: .603}
    y_positions = {1: .513, 0: .232}
    width, height = .342, .236
    for outcome, x in x_positions.items():
        ax.add_patch(FancyBboxPatch((x, .779), width, .049,
                                    boxstyle="round,pad=0.004,rounding_size=.01",
                                    facecolor="#F3F7F8", edgecolor="none"))
        ax.text(x + width / 2, .8035,
                "不发邮件时：" + ("购买" if outcome else "不购买"),
                fontsize=14, color=ink, ha="center", va="center", weight="bold")
    for outcome, y in y_positions.items():
        ax.text(.135, y + height / 2 + .018, "发邮件时", fontsize=13,
                color=muted, ha="center", va="center")
        ax.text(.135, y + height / 2 - .019,
                "购买" if outcome else "不购买", fontsize=19,
                color=ink, ha="center", va="center", weight="bold")

    for row in TYPES:
        x, y = x_positions[row["Y0"]], y_positions[row["Y1"]]
        accent, background = row["accent"], row["background"]
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                                    boxstyle="round,pad=0.009,rounding_size=.014",
                                    facecolor=background, edgecolor="none"))
        ax.plot([x + .018, x + .018], [y + .028, y + height - .028],
                color=accent, lw=3.2, solid_capstyle="round")
        tx = x + .037
        ax.text(tx, y + height - .039, row["title"], fontsize=20,
                color=accent, weight="bold", va="center")
        ax.text(tx, y + height - .070, row["english"], fontsize=10.5,
                color=muted, va="center")
        ax.text(tx, y + height - .112, row["statement"], fontsize=15,
                color=ink, va="center")
        ax.text(tx, y + height - .156, row["effect"], fontsize=12.7,
                color=accent, weight="bold", va="center")
        ax.text(tx, y + .031, row["implication"], fontsize=9.6,
                color=muted, va="center")

    ax.add_patch(FancyBboxPatch((.055, .133), .89, .054,
                                boxstyle="round,pad=0.008,rounding_size=.011",
                                facecolor="#EEF6F3", edgecolor="none"))
    ax.text(.075, .16,
            "本项目做法：预测发／不发邮件的购买概率差，再在保留测试集比较选人策略。",
            fontsize=12.3, color=teal, va="center")
    ax.text(.055, .091,
            "这是理论分类，非真实客户分群结果；同一人只能观察一种情景，模型分数不能验证个人所属类型。",
            fontsize=10.3, color=ink, va="center")
    ax.text(.055, .061,
            "同一次活动、同一14天购买窗口；不展示真实人数或占比，也不据此宣称已找到盈利人群。",
            fontsize=9.5, color=muted, va="center")
    ax.text(.055, .032, "概念来源：CausalML、scikit-uplift 官方说明。", fontsize=9,
            color=muted, va="center")

    output.mkdir(parents=True, exist_ok=True)
    png = output / "13_customer_response_types.png"
    svg = output / "13_customer_response_types.svg"
    fig.savefig(png, dpi=dpi)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg, matplotlib.__version__


def write_notes(output: Path):
    notes = """# 四类潜在响应人群：概念说明与面试讲述

本图用于解释项目的业务动机，适合放在作品集或面试材料的前面。后面再展示真实策略比较图。

## 可以这样讲（约20秒）

“同样是购买客户，有人本来就会买，也有人是因为营销才购买。只预测谁会买，不能直接回答营销带来了多少新增购买。因此我用这个公开实验项目比较购买概率排序和增量排序，并用保留测试集检验它们的策略效果。”

## 四格怎么读

列表示同一客户在不发邮件时的潜在购买结果，行表示发邮件时的潜在购买结果。购买限定为同一次活动后的14天窗口。

- 可促成购买：不发不买、发了才买，正增量。
- 自然购买：发或不发都会买，对是否购买没有增量。
- 本次均不购买：两种情景都不买，仅针对本次活动和窗口。
- 负向响应：不发会买、发了不买。分类本身不能说明反感、打扰等具体原因。

## 需要说明的边界

这是一张理论概念图，没有给真实客户贴标签或计算人数。每位客户在一次实验中只实际处于一种情景；p0和p1是预测概率，不能当作同一客户两份已观测的结果。不能用任意概率阈值把人群划成四类后宣称这些就是真实类别。

本项目的测试集没有确认增量排序优于购买概率排序。概念图仅说明为什么值得研究增量，不能作为模型已经有效识别人群或已经创造利润的证明。

## 运行方法

使用已有项目虚拟环境与matplotlib即可。把13_plot_customer_types.py放到D盘项目根目录，VS Code选择.venv环境后点击右上角运行。图片保存到reports/figures。

来源：

- https://causalml.readthedocs.io/en/latest/methodology.html#value-optimization-methods
- https://www.uplift-modeling.com/en/v0.3.0/user_guide/introduction/clients.html
"""
    path = output / "13_人群概念图说明.md"
    path.write_text(notes, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if not 120 <= args.dpi <= 600:
        raise ValueError("DPI应在120到600之间。")
    output = (args.output_dir or args.project_dir / "reports/figures").resolve()
    png, svg, version = draw(output, args.dpi)
    notes = write_notes(output)
    manifest = {
        "figure_type": "conceptual_potential_response_types",
        "contains_customer_records": False,
        "contains_estimated_group_counts": False,
        "classifies_real_customers": False,
        "refits_model": False,
        "window": "Same campaign and same 14-day binary purchase window",
        "types": TYPES, "sources": SOURCES,
        "script_sha256": sha256(Path(__file__).resolve()),
        "runtime": {"python": sys.version.split()[0], "matplotlib": version},
        "outputs": {path.name: sha256(path) for path in (png, svg, notes)},
    }
    (output / "13_plot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"png": str(png), "svg": str(svg), "readme": str(notes),
                      "figure_type": "概念示意，非真实客户分群结果"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
