# 查看结果、复画图表与重新运行

## 无需安装即可阅读

根目录README展示业务问题和结果。`reports/figures`附PNG、SVG；`reports`附聚合结果CSV与原运行设置。可先运行 `python tools/check_archive.py` 核对本次展示快照的文件哈希，该命令只需要Python标准库。

## 运行环境

本次实际使用Windows、Python3.12.0，直接依赖版本已写入requirements.txt：numpy2.5.4、pandas3.0.6、scikit-learn1.9.1、scipy1.18.1、statsmodels0.15.0、joblib1.6.0、matplotlib3.10.8。requirements-lock.txt保留原环境完整pip freeze快照，可能含项目未直接调用的依赖。

新电脑先在项目目录创建环境并安装直接依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

以下命令均在项目根目录执行。直接指定解释器即可，不需要激活环境或修改PowerShell执行策略。VS Code点击运行时，选择此项目的`.venv`解释器。

## 只复画图表

不需要下载原始数据，也不需要重训。脚本11/12/14读取随包附带的09汇总CSV与协议，13独立绘制概念图。

```powershell
.\.venv\Scripts\python.exe 11_plot_policy_results.py
.\.venv\Scripts\python.exe 12_plot_policy_differences.py
.\.venv\Scripts\python.exe 13_plot_customer_types.py
.\.venv\Scripts\python.exe 14_plot_capacity_comparison.py
```

默认保存至`reports/figures`，不会自动弹出图片。想保留已附图片和作图清单，可在每条命令末尾加 `--output-dir runs/figures`。脚本需要中文字体；Windows默认寻找微软雅黑/黑体，其他系统须配置相应中文字体。字体与渲染环境不同会影响图片字节，不应影响来源数值。

## 全新目录重新运行01–09

公开展示快照未附原始CSV、模型、逐行分数与名单。先通过来源获取并校验原始数据：

```powershell
.\.venv\Scripts\python.exe tools/fetch_data.py
```

若已有原始压缩包，也可离线恢复：

```powershell
.\.venv\Scripts\python.exe tools/fetch_data.py --archive "D:\营销项目\marketing_project_starter\data\raw\hillstrom_original.csv.gz"
```

上例路径仅供已有本地文件的原作者使用，其他读者改为自己的原始压缩包路径。工具只接受与来源清单匹配的gzip/CSV，不覆盖不同内容。

然后运行：

```powershell
.\.venv\Scripts\python.exe tools/reproduce.py
```

工具将原始01–09脚本和校验过的CSV复制到空的`runs/reproduction`，顺序执行并保存阶段日志、完整新产物和`reproduction_check.json`。它会将新09数值与本次归档比较。该目录已存在且非空时会停止；另一次运行可指定新的空目录：

```powershell
.\.venv\Scripts\python.exe tools/reproduce.py --output-dir runs/reproduction_02
```

所有新报告、模型及分数都保留在新目录，当前根目录reports作为历史结果保留。重复同一数据与规则不是新的独立实验。

## 为什么在空目录运行

原脚本对源数据、划分、代码与模型使用哈希关联，防止不同运行记录混用。部分阶段报告包含新时间戳；在已冻结目录中重跑上游可能使下游按设计拒绝继续。不要删除检查或覆盖历史记录来“解决”此类错误。

01/02由脚本位置决定目录，没有`--help`参数；03–09支持`--project-dir`和`--output-dir`，但后者只改变当阶段输出，上游仍从项目的reports读取。因此全流程由独立目录工具组织，不把每个阶段重定向到不同报告目录。

`.gitattributes`保留原始字节和行尾，避免Git自动转换破坏哈希。原始时间戳、绝对路径和旧哈希只是历史元数据；新完整运行允许生成新的自洽链，不要求所有文件哈希与旧归档相同。

## 本次实际验证范围

2026-10-16在本机原有Python3.12.0依赖环境中，把01–09及原始CSV、来源清单复制到全新目录，全部执行成功；划分CSV和最终总体效果、策略结果、配对差值文件与原归档逐字节一致。记录见`docs/evidence/reproduction_verification.json`。

本次验证使用已有本地原始压缩包与已安装依赖，未验证新电脑从零安装或重新联网下载；其他系统、版本及中文字体配置需按日志核对。固定依赖环境的复现一致性不等于模型已在新客户或线上实验验证。
