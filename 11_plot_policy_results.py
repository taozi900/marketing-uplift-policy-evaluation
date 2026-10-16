r"""从阶段09已保存的汇总报告绘制第一张图；不训练、不重新评价测试集。

放到项目根目录后可运行：
    .\.venv\Scripts\python.exe 11_plot_policy_results.py

仅绘图新增依赖 matplotlib；若尚未安装，可在项目环境中安装：
    .\.venv\Scripts\python.exe -m pip install matplotlib

也可指定路径：
    python 11_plot_policy_results.py --project-dir <项目目录> --output-dir <作图目录>
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


POLICIES = [
    ("random", "固定随机名单"),
    ("history_descending", "历史消费排序"),
    ("p1_descending", "购买概率排序"),
    ("predicted_uplift_descending", "预测增量排序"),
]
UNIT = "每万名合格客户的新增购买数（次）"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"报告包含非有限数值：{key}")
    return value


def load_plot_inputs(project_dir: Path):
    files = {
        "policy_results": project_dir / "reports/09_test_policy_results.csv",
        "policy_differences": project_dir / "reports/09_test_policy_differences.csv",
        "specification": project_dir / "reports/09_test_specification.json",
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(
                f"缺少阶段09报告：{path}。请把本脚本放到项目根目录，或指定--project-dir。"
            )
    hashes = {key: sha256(path) for key, path in files.items()}
    rows = [row for row in read_csv(files["policy_results"])
            if math.isclose(float(row["budget_fraction"]), .2, abs_tol=1e-12)]
    if len(rows) != 4 or {row["policy"] for row in rows} != {p for p, _ in POLICIES}:
        raise ValueError("20%容量必须恰好包含预定的四种策略。")
    by_policy = {row["policy"]: row for row in rows}
    ordered = [by_policy[key] for key, _ in POLICIES]
    if any(row["analysis_set"] != "test" or row["analysis_role"] != "primary_capacity"
           for row in ordered):
        raise ValueError("作图输入必须为保留测试集、20%主容量的结果。")
    n_values = {int(row["eligible_count"]) for row in ordered}
    k_values = {int(row["selected_count"]) for row in ordered}
    if len(n_values) != 1 or len(k_values) != 1:
        raise ValueError("四策略必须使用相同客户集合大小和选中数量。")
    n, k = n_values.pop(), k_values.pop()
    if n <= 0 or k != math.floor(.2 * n):
        raise ValueError("触达数量不符合floor(20% * N)规则。")
    for row in ordered:
        estimate = number(row, "incremental_per_10000_eligible")
        lo = number(row, "ci_low_per_10000_eligible")
        hi = number(row, "ci_high_per_10000_eligible")
        if not lo <= estimate <= hi:
            raise ValueError("本绘图方式要求区间上下限包含点估计，请检查原始报告。")
        manual = 10000 * (2 * int(row["selected_buyers_T1"])
                          - 2 * int(row["selected_buyers_T0"])) / n
        if not math.isclose(estimate, manual, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("策略估计与报告内购买计数的IPW计算不一致。")
    contrasts = [row for row in read_csv(files["policy_differences"])
                 if row["comparison_role"] == "primary"]
    if len(contrasts) != 1:
        raise ValueError("报告必须有且只有一个预定主要比较。")
    primary = contrasts[0]
    if (primary["analysis_set"] != "test"
        or primary["candidate"] != "predicted_uplift_descending"
        or primary["baseline"] != "p1_descending"
        or not math.isclose(float(primary["budget_fraction"]), .2)):
        raise ValueError("主要比较必须为测试集20%容量下的增量排序减购买概率排序。")
    primary_value = number(primary, "difference_per_10000_eligible")
    expected_value = (number(by_policy["predicted_uplift_descending"],
                             "incremental_per_10000_eligible")
                      - number(by_policy["p1_descending"],
                               "incremental_per_10000_eligible"))
    if not math.isclose(primary_value, expected_value, abs_tol=1e-10):
        raise ValueError("主要差值与两策略点估计不一致。")
    lo = number(primary, "ci_low_per_10000_eligible")
    hi = number(primary, "ci_high_per_10000_eligible")
    if not lo <= primary_value <= hi:
        raise ValueError("主要差值区间不包含点估计，请检查原报告。")
    spec = json.loads(files["specification"].read_text(encoding="utf-8"))
    payload = spec["payload"]
    if payload["analysis_set"] != "test":
        raise ValueError("评估协议不是测试集。")
    rules = payload["inherited_rules"]
    if rules["propensity"] != .5 or rules["primary_capacity"] != .2:
        raise ValueError("绘图协议与现有IPW/容量定义不一致。")
    return ordered, primary, n, k, files, hashes, rules


def font_path() -> Path:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("未找到支持中文的字体，请在font_path()中配置本机中文字体路径。")


def draw_figure(rows, primary, n: int, k: int, out_dir: Path, dpi: int):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager, patches
    except ImportError as exc:
        raise SystemExit(
            "绘图需要matplotlib。请在项目环境安装："
            ".\\.venv\\Scripts\\python.exe -m pip install matplotlib"
        ) from exc

    font_manager.fontManager.addfont(str(font_path()))
    family = font_manager.FontProperties(fname=str(font_path())).get_name()
    plt.rcParams.update({
        "font.family": family,
        "axes.unicode_minus": False,
        "svg.fonttype": "path",
        "font.size": 12,
        "savefig.facecolor": "white",
    })
    warnings.filterwarnings("error", message=r"Glyph .* missing from font")
    ink, muted = "#18313E", "#5C6F7B"
    teal, blue = "#147D73", "#426D96"
    base = "#80929C"
    fig = plt.figure(figsize=(13.6, 9.5), facecolor="white")
    fig.text(.06, .94, "同样触达20%，四种选人策略的增量估计", fontsize=21,
             weight="bold", color=ink)
    fig.text(.06, .898,
             f"冻结测试集 {n:,} 条客户记录  |  各策略选 {k:,} 条  |  14天是否购买",
             fontsize=11.5, color=muted)
    fig.text(.06, .845, "各策略相对“所有人不发邮件”的增量", fontsize=13.5,
             weight="bold", color=ink)
    fig.text(.728, .802, "点估计", fontsize=10.5, color=muted)
    fig.text(.818, .802, "逐项95%区间", fontsize=10.5, color=muted)
    ax = fig.add_axes([.205, .452, .475, .335])
    ax.axvline(0, color="#7B8C98", lw=1.3, ls=(0, (4, 4)), zorder=1)
    colors = [base, base, blue, teal]
    for i, ((_, label), row, color) in enumerate(zip(POLICIES, rows, colors)):
        y = 3 - i
        point = number(row, "incremental_per_10000_eligible")
        lo = number(row, "ci_low_per_10000_eligible")
        hi = number(row, "ci_high_per_10000_eligible")
        if i == 3:
            ax.axhspan(y - .40, y + .40, color="#EEF7F4", zorder=0)
        ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o",
                    color=color, ecolor=color, markersize=9, capsize=5,
                    capthick=1.7, elinewidth=2.4, zorder=3)
    ax.set_xlim(-15, 55)
    ax.set_ylim(-.65, 3.65)
    ax.set_yticks([3, 2, 1, 0], [label for _, label in POLICIES], color=ink)
    ax.tick_params(axis="y", length=0, pad=17, labelsize=12)
    ax.set_xticks([-10, 0, 10, 20, 30, 40, 50])
    ax.tick_params(axis="x", colors=muted, labelsize=10.5, length=0, pad=8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#E4ECEF", lw=.8)
    for side in ["left", "right", "top"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#CAD8DE")
    ax.set_xlabel(UNIT, fontsize=11, color=muted, labelpad=12)
    for i, row in enumerate(rows):
        y = 3 - i
        fig_y = fig.transFigure.inverted().transform(ax.transData.transform((0, y)))[1]
        color = colors[i]
        fig.text(.743, fig_y, f"{number(row, 'incremental_per_10000_eligible'):+.2f}",
                 ha="center", va="center", fontsize=13, color=color,
                 weight="bold" if i == 3 else "normal")
        fig.text(.88, fig_y,
                 f"[{number(row, 'ci_low_per_10000_eligible'):+.2f}, "
                 f"{number(row, 'ci_high_per_10000_eligible'):+.2f}]",
                 ha="center", va="center", fontsize=11.5, color=ink)

    fig.add_artist(patches.FancyBboxPatch(
        (.055, .153), .89, .185, boxstyle="round,pad=0.012,rounding_size=0.012",
        transform=fig.transFigure, facecolor="#F0F5F7", edgecolor="none", zorder=-1))
    fig.text(.075, .302, "主要成对差值：增量排序 - 购买概率排序",
             fontsize=13, weight="bold", color=ink)
    p = number(primary, "difference_per_10000_eligible")
    lo = number(primary, "ci_low_per_10000_eligible")
    hi = number(primary, "ci_high_per_10000_eligible")
    main = fig.add_axes([.205, .215, .475, .062], facecolor="none")
    main.axvline(0, color="#7B8C98", lw=1.3, ls=(0, (4, 4)), zorder=1)
    main.errorbar(p, 0, xerr=[[p - lo], [hi - p]], fmt="D", color=teal,
                  markersize=8, capsize=5, capthick=1.7, elinewidth=2.4, zorder=3)
    main.set_xlim(ax.get_xlim())
    main.set_ylim(-1, 1)
    main.set_yticks([])
    main.set_xticks([-10, 0, 10, 20, 30, 40, 50])
    main.tick_params(axis="x", colors=muted, labelsize=9, length=0)
    for spine in main.spines.values():
        spine.set_visible(False)
    fig.text(.743, .246, f"{p:+.2f}", ha="center", va="center", fontsize=14,
             weight="bold", color=teal)
    fig.text(.88, .246, f"[{lo:+.2f}, {hi:+.2f}]", ha="center", va="center",
             fontsize=11.5, color=ink)
    if lo <= 0 <= hi:
        interpretation = "差值区间包含0：尚不足以确认增量排序更好，也不表示两者效果相同。"
    elif lo > 0:
        interpretation = "主差值区间全在0以上：本次近似评价支持增量排序有优势。"
    else:
        interpretation = "主差值区间全在0以下：本次近似评价支持购买概率排序有优势。"
    fig.text(.075, .169, interpretation, fontsize=11, color=ink)

    fig.text(.06, .105, "读图：圆点为增量估计，横线为区间；策略间优势看成对差值，不能只看圆点高低。",
             fontsize=10, color=muted)
    fig.text(.06, .075, "公开实验离线估计；5,000次成对分层Bootstrap；逐项95%区间，不承诺同时覆盖。",
             fontsize=9.1, color=muted)
    fig.text(.06, .049, "区间条件于固定模型与名单，未包含重训和重排不确定性；购买事件稀少时需谨慎解释。",
             fontsize=9.1, color=muted)
    fig.text(.06, .024, "来源：阶段09冻结测试报告  |  原始报告的策略顺序与数值均保留",
             fontsize=8.7, color=muted)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "11_20pct_policy_intervals.png"
    svg = out_dir / "11_20pct_policy_intervals.svg"
    fig.savefig(png, dpi=dpi)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg, matplotlib.__version__


def write_readme(out_dir: Path):
    text = """# 第一张图：20%容量下四种策略的增量与区间

## 先看什么

1. 上半图每个圆点表示该策略相对“所有人都不发邮件”的新增购买离线估计；横线是逐项95% Bootstrap区间，虚线为0。
2. 单位为“每万名合格客户的新增购买数”，分母包括全部合格客户。测试集8,524条记录，每种策略选1,704条。
3. 下半图为预定主要成对比较：增量排序减购买概率排序，点估计+7.04，95%区间[-9.39,+25.81]。区间跨0，尚不足以确认谁更好，也不能据此说两者效果相同。
4. 上图某策略区间不跨0，不等于该策略优于其他策略；不能从两条单策略区间的重叠判断差异。

## 这次作图做了什么

只读取阶段09已经保存的汇总CSV与评估协议，保留策略顺序和所有数值；未读取逐条测试结果、未重新训练、未重新拆分、未计算新的效果估计或选择新容量。主要差值区间直接使用原报告的成对Bootstrap结果，没有用两条单策略区间相减。

区间为固定模型和名单下的近似评价；并非多项同时覆盖，也不包含重训与重排的不确定性。随机名单所选客户中，T0组只有4次购买，稀疏事件使该区间尤其需要谨慎解释。

## 你说的四类客户

这是二元购买结果下的四种理论潜在响应类型，和四种选人策略是两个概念。下表描述的是同一次活动、同一观察窗口下的假设结果，不是已经识别出的真实客户标签。

| 理论类型 | 不发邮件时的潜在结果 | 发邮件时的潜在结果 |
|---|---|---|
| 可被营销促成购买（Persuadables） | 不购买 | 购买 |
| 自然购买（Sure Things） | 购买 | 购买 |
| 两种情况下均不购买（Lost Causes） | 不购买 | 不购买 |
| 被营销劝退（Sleeping Dogs） | 购买 | 不购买 |

同一人在同一实验中只能实际处于发邮件或不发邮件的一种状态，不能同时观察两种结果。当前p0和p1是模型预测概率，不是两份已观测的个人结果；不能凭本项目给客户确定地贴这四类标签、统计四类真实人数或宣称找到了所有可被说服者。

概念来源：[CausalML方法说明](https://causalml.readthedocs.io/en/latest/methodology.html#value-optimization-methods)、[scikit-uplift四类客户说明](https://www.uplift-modeling.com/en/v0.3.0/user_guide/introduction/clients.html)。

## 如何复现图片

已提供可直接保存的PNG与SVG。PNG适合放入作品集/PPT；SVG为矢量格式，可无损放大。

在自己的项目环境运行脚本时，绘图新增依赖为matplotlib。首次需要在项目虚拟环境安装：

```powershell
.\\.venv\\Scripts\\python.exe -m pip install matplotlib
```

把11_plot_policy_results.py放在项目根目录后运行：

```powershell
.\\.venv\\Scripts\\python.exe .\\11_plot_policy_results.py
```

默认图片保存在项目的reports/figures文件夹。助手本次用独立的现成绘图环境生成输出文件，没有修改D盘项目环境。
"""
    path = out_dir / "11_读图与复现说明.md"
    path.write_text(text, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi < 120 or args.dpi > 600:
        raise ValueError("导出分辨率应在120到600 DPI之间。")
    project = args.project_dir.resolve()
    out_dir = (args.output_dir or project / "reports/figures").resolve()
    rows, primary, n, k, files, hashes, rules = load_plot_inputs(project)
    png, svg, version = draw_figure(rows, primary, n, k, out_dir, args.dpi)
    for key, path in files.items():
        if sha256(path) != hashes[key]:
            raise RuntimeError(f"作图期间来源报告发生变化：{path}")
    manifest = {
        "analysis_set": "test", "capacity": .2, "eligible_count": n,
        "selected_count_per_policy": k, "unit": UNIT,
        "role": "Display of saved stage-09 estimates only; no model or policy changes.",
        "policy_order": [key for key, _ in POLICIES], "data": rows,
        "primary_paired_comparison": primary,
        "interval": rules["bootstrap"],
        "sources": {key: {"path": str(path), "sha256": hashes[key]}
                    for key, path in files.items()},
        "script_sha256": sha256(Path(__file__).resolve()),
        "runtime": {"python": sys.version.split()[0], "matplotlib": version},
        "outputs": {"png": {"file": png.name, "sha256": sha256(png)},
                    "svg": {"file": svg.name, "sha256": sha256(svg)}},
    }
    (out_dir / "11_plot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = write_readme(out_dir)
    print(json.dumps({"png": str(png), "svg": str(svg), "readme": str(readme),
                      "source_reports_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
