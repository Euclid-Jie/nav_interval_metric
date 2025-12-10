# nav_interval_metric

一个用于净值序列指标计算的轻量工具集，当前支持：
- 净值曲线规范化（自动识别周频/日频，校准到交易日）
- 区间收益率、区间最大回撤、区间夏普、区间卡玛等指标计算

适合以 Git submodule 的方式集成到上层项目中，也可直接作为独立脚本运行。

## 目录结构

```
nav_interval_metric/
├─ nav_metric.py          # 核心类：NavMetric、IntervalReturnETC
├─ utils.py               # 交易日生成、曲线分析、回撤工具
├─ data/                  # 示例数据（CSV）
└─ Chinese_special_holiday.txt  # 节假日表（供交易日生成使用）
```

## 环境依赖

- Python 3.8+
- numpy, pandas

安装依赖（示例）：

```powershell
python -m pip install -U numpy pandas
```

## 作为 Submodule 集成

在你的主仓库中添加本仓库为子模块（将 <repo-url> 替换为实际地址）：

```powershell
git submodule add <repo-url> nav_interval_metric
# 首次或更新子模块内容
git submodule update --init --recursive
```

更新子模块到最新提交：

```powershell
cd nav_interval_metric
git fetch; git checkout <target-commit-or-branch>
cd ../..
# 在主仓库提交子模块指针更新
git add nav_interval_metric
git commit -m "chore: bump nav_interval_metric"
```

移除子模块（可选）：

```powershell
git rm -f nav_interval_metric
# 清理.git/config和.gitmodules中相关段落（如存在）
```

## 在上层项目中的导入方式

有两种常见方式：

- 方式 A：将子模块父目录加入 `PYTHONPATH`
  - 推荐在 `nav_interval_metric/` 目录旁放置一个 `__init__.py` 使其成为包后，通过如下方式导入：
    ```python
    from nav_interval_metric.nav_metric import NavMetric, IntervalReturnETC
    ```
  - 当前仓库内 `nav_metric.py` 使用了相对导入兜底（try/except），可以在未打包前通过追加路径使用：
    ```python
    import sys
    sys.path.append("path/to/nav_interval_metric")
    from nav_metric import NavMetric, IntervalReturnETC
    ```

- 方式 B：直接在你的代码中相对路径调用（不建议长期使用）
  - 适合快速验证，长期应整理为包导入以便维护。

如果你希望我顺便补一个 `__init__.py` 来导出公共 API，告诉我即可，我可以直接加上：
```python
from .nav_metric import NavMetric, IntervalReturnETC
```
这样即可通过 `from nav_interval_metric import NavMetric, IntervalReturnETC` 直接导入。

## 数据格式要求

- CSV 至少包含以下列：
  - `日期`：能被 `pandas.to_datetime` 解析的日期
  - `累计净值`：数值型净值
- 仅支持 2020-01-01 及以后的数据；更早数据会被自动截断。
- 若存在缺失交易日的净值：
  - 默认会给出 `warnings.warn` 提示；
  - 可通过 `ffillna=True` 启用前向填充（建议数据源尽量完整）。

## 快速上手

以项目内置示例数据为例：

```python
from pathlib import Path
import pandas as pd
import numpy as np
from nav_metric import NavMetric, IntervalReturnETC  # 若作为包导入，见上文“导入方式”

nav_data_path = Path(r"data/SAHC74_合骥睿源2000增强1号.csv")
nav_df = pd.read_csv(nav_data_path)
nav_df["日期"] = pd.to_datetime(nav_df["日期"])

metric = NavMetric(
    name=nav_data_path.stem,
    nav=nav_df["累计净值"].to_numpy(dtype=np.float64),
    date=nav_df["日期"].to_numpy(dtype="datetime64[D]"),
    # freq=None 表示自动识别（日/周）；也可手动指定 "D" 或 "W"
    # ffillna=True 可在出现缺失时前向填充
)
print(metric)

intervals = [
    IntervalReturnETC("年收益", np.datetime64("2024-12-27"), np.datetime64("2025-09-30")),
    IntervalReturnETC("季度收益", np.datetime64("2025-06-27"), np.datetime64("2025-09-30")),
]
results = metric.calculate_interval_return(intervals)
for r in results:
    print(r)
```

也可以直接在仓库根目录运行示例（`nav_metric.py` 中带有 `__main__` 示例）：

```powershell
# 从 nav_interval_metric/ 目录执行
python nav_metric.py
```

## API 概览

- `class IntervalReturnETC(NamedTuple)`
  - 字段：`name: str`, `start_date: np.datetime64`, `end_date: np.datetime64`,
    `interval_return: float = np.nan`, `interval_MDD: float = np.nan`,
    `interval_sharpe: float = np.nan`, `interval_karma: float = np.nan`
  - 方法：`update(interval_return, interval_MDD, interval_sharpe=np.nan, interval_karma=np.nan)`
  - `__repr__` 会格式化输出区间名称、起止日期与指标。

- `class NavMetric`
  - 初始化：
    ```python
    NavMetric(
      name: str,
      nav: np.ndarray[np.float64],
      date: np.ndarray[np.datetime64],
      freq: Literal["W", "D"] | None = None,
      ffillna: bool = False,
    )
    ```
  - 主要属性：`name`, `nav`, `date`, `begin_date`, `end_date`, `freq`, `base_metric_dict`
  - 主要方法：
    - `calculate_interval_return(intervals: list[IntervalReturnETC]) -> list[IntervalReturnETC]`
      - 返回填充好 `interval_return / interval_MDD / interval_sharpe / interval_karma` 的列表。
  - 实现要点：
    - 自动识别频率：当与交易日重合度超过阈值（默认 0.9）视为日频，否则周频。
    - 使用交易日对齐并在必要时发出缺失提示；可选择前向填充。