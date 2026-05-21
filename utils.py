import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pathlib import Path
from typing import Tuple, Optional

__all__ = [
    "generate_trading_date",
    "drawdown_stats",
    "curve_analysis",
]

_chinese_special_holiday: Optional[NDArray[np.datetime64]] = None


def _load_holidays() -> NDArray[np.datetime64]:
    global _chinese_special_holiday
    if _chinese_special_holiday is None:
        with open(Path(__file__).resolve().parent / "Chinese_special_holiday.txt") as f:
            _chinese_special_holiday = np.array(
                [d.strip() for d in f.readlines()], dtype="datetime64[D]"
            )
    return _chinese_special_holiday

_chinese_special_holiday: Optional[NDArray[np.datetime64]] = None


def _load_holidays() -> NDArray[np.datetime64]:
    global _chinese_special_holiday
    if _chinese_special_holiday is None:
        with open(Path(__file__).resolve().parent / "Chinese_special_holiday.txt") as f:
            _chinese_special_holiday = np.array(
                [d.strip() for d in f.readlines()], dtype="datetime64[D]"
            )
    return _chinese_special_holiday


def drawdown_stats(
    nav: np.ndarray, date: np.ndarray
) -> Tuple[Optional[NDArray[np.float64]], pd.DataFrame]:
    assert len(nav) == len(date), "nav和date长度不一致, 请检查bench_data是否更新"
    cummax = np.maximum.accumulate(nav)
    drawdown = (nav - cummax) / cummax
    if drawdown.min() == 0:
        return None, pd.DataFrame()
    drawdown_infos = []
    idx = 0
    while idx < len(drawdown) - 1:
        if drawdown[idx] < 0:
            drawdown_info = {}
            drawdown_info["drawdown_start_date"] = date[idx - 1]
            drawdown_info["max_drawdown"] = drawdown[idx]
            drawdown_info["max_drawdown_date"] = date[idx]
            while drawdown[idx] < 0:
                if drawdown[idx] < drawdown_info["max_drawdown"]:
                    drawdown_info["max_drawdown"] = drawdown[idx]
                    drawdown_info["max_drawdown_date"] = date[idx]
                if idx == len(drawdown) - 1:
                    break
                idx += 1
            drawdown_info["drawdown_end_date"] = date[idx]
            drawdown_infos.append(drawdown_info)
        else:
            idx += 1
    if drawdown[-1] < 0:
        drawdown_infos[-1]["drawdown_end_date"] = np.datetime64("NaT")

    drawdown_infos = pd.DataFrame(drawdown_infos)
    drawdown_infos["max_drawdown_days"] = (
        drawdown_infos["max_drawdown_date"] - drawdown_infos["drawdown_start_date"]
    )
    drawdown_infos["drawdown_fix_days"] = (
        drawdown_infos["drawdown_end_date"] - drawdown_infos["max_drawdown_date"]
    )
    return (
        drawdown,
        drawdown_infos[
            [
                "max_drawdown",
                "drawdown_start_date",
                "max_drawdown_date",
                "max_drawdown_days",
                "drawdown_end_date",
                "drawdown_fix_days",
            ]
        ],
    )


def generate_trading_date(
    begin_date: np.datetime64 = np.datetime64("2015-01-04"),
    end_date: np.datetime64 = np.datetime64("today"),
) -> Tuple[NDArray[np.datetime64], NDArray[np.datetime64]]:
    assert begin_date >= np.datetime64(
        "2015-01-04"
    ), "系统预设起始日期仅支持2015年1月4日以后"
    chinese_special_holiday = _load_holidays()
    working_date = pd.date_range(begin_date, end_date, freq="B").values.astype(
        "datetime64[D]"
    )
    trading_date = np.setdiff1d(working_date, chinese_special_holiday)
    trading_date_df = pd.DataFrame(working_date, columns=["working_date"])
    trading_date_df["is_friday"] = trading_date_df["working_date"].apply(
        lambda x: x.weekday() == 4
    )
    is_trading = np.isin(working_date, trading_date)
    trading_date_df["trading_date"] = np.where(
        is_trading, working_date, np.datetime64("NaT")
    )
    trading_date_df["trading_date"] = pd.to_datetime(
        trading_date_df["trading_date"]
    ).ffill()
    friday_trading_dates = trading_date_df[trading_date_df["is_friday"]][
        "trading_date"
    ].to_numpy(dtype="datetime64[D]")[1:]
    return trading_date, np.unique(friday_trading_dates).astype("datetime64[D]")


def calculate_karma_ratio(annual_return: float, max_drawdown: float) -> float:
    if max_drawdown == 0:
        return np.inf
    return annual_return / (-1 * max_drawdown)


def curve_analysis(
    nav: NDArray[np.float64],
    date: NDArray[np.datetime64],
    risk_free_rate: float = 0.02,
) -> dict[str, float]:
    assert nav.ndim == 1, "nav维度不为1"
    assert np.isnan(nav).sum() == 0, "nav中有nan"
    assert len(nav) > 2, "nav不足两条, 无法计算"
    result = {"区间收益率": nav[-1] / nav[0] - 1}
    result["年化收益率"] = (1 + result["区间收益率"]) ** (
        365 / (date[-1] - date[0]).astype("timedelta64[D]").astype(int)
    ) - 1
    rtn = np.log(nav[1:] / nav[:-1])
    result["区间波动率"] = np.std(rtn, ddof=1)
    result["年化波动率"] = result["区间波动率"] * np.sqrt(
        len(rtn) * 365 / (date[-1] - date[1]).astype("timedelta64[D]").astype(int)
    )
    result["夏普比率"] = (result["年化收益率"] - risk_free_rate) / result["年化波动率"]
    cummax = np.maximum.accumulate(nav)
    result["最大回撤"] = np.min((nav - cummax) / cummax)
    result["卡玛比率"] = calculate_karma_ratio(result["年化收益率"], result["最大回撤"])
    return result
