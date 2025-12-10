import numpy as np
import pandas as pd
from typing import Literal, NamedTuple, Optional, Tuple
from numpy.typing import NDArray
from pathlib import Path

try:
    from .utils import generate_trading_date, curve_analysis, drawdown_stats
except ImportError:
    from utils import generate_trading_date, curve_analysis, drawdown_stats
import warnings


# 后续可以拓展其他指标
class IntervalReturnETC(NamedTuple):
    name: str
    start_date: np.datetime64
    end_date: np.datetime64
    interval_return: float = np.nan
    interval_MDD: float = np.nan
    interval_sharpe: float = np.nan
    interval_karma: float = np.nan
    """
    区间收益相关指标
    params:
        name: 区间名称
        start_date: 区间开始日期
        end_date: 区间结束日期
        interval_return: 区间收益率
        interval_MDD: 区间最大回撤
        interval_sharpe: 区间夏普比率
        interval_karma: 区间卡玛比率
    """

    def __repr__(self):
        return f"[IntervalReturn] {self.name}({self.start_date} - {self.end_date}) , Rtn = {self.interval_return:.2%}, MDD = {self.interval_MDD:.2%}, Sharpe = {self.interval_sharpe:.2f}, Karma = {self.interval_karma:.2f}"

    def update(
        self,
        interval_return: float,
        interval_MDD: float,
        interval_sharpe: float = np.nan,
        interval_karma: float = np.nan,
    ):
        return self._replace(
            interval_return=interval_return,
            interval_MDD=interval_MDD,
            interval_sharpe=interval_sharpe,
            interval_karma=interval_karma,
        )


class NavMetric:
    def __init__(
        self,
        name: str,
        nav: NDArray[np.float64],
        date: NDArray[np.datetime64],
        freq: Optional[Literal["W", "D"]] = None,
        ffillna: bool = False,
    ):
        """
        parmas:
            name: 净值名称
            nav: 净值序列
            date: 日期序列
            freq: 净值频率, 'W'表示周度, 'D'表示日度, 默认为None, 自动识别
            ffillna: 是否对缺失的净值进行前后填充
        仅支持2020年1月1日以后的净值数据, 早于此自动截断
        """
        assert len(nav) == len(date), "净值和日期长度不一致"
        assert freq in [None, "W", "D"], "freq仅支持'W'或'D'或None"
        begin_idx = np.where(date >= np.datetime64("2020-01-01"))[0][0]
        nav = nav[begin_idx:]
        date = date[begin_idx:]

        self.name = name
        self.nav = nav
        self.date = date
        self.freq = freq
        self.ffillna = ffillna

        self.trade_date, self.weekly_trade_date = generate_trading_date(
            self.date[0] - np.timedelta64(10, "D"),
            self.date[-1] + np.timedelta64(5, "D"),
        )
        # 经过这步后, date和nav已经被重新截断过了
        self.format_nav()
        self.base_metric_dict = curve_analysis(self.nav, self.date)

    def __repr__(self):
        return f"Nav_metric(name={self.name}, freq={self.freq} begin_date={self.begin_date.astype('M8[D]')}, end_date={self.end_date.astype('M8[D]')})"

    def format_nav(self, treshold: float = 0.9):
        """
        两种情形: 1. 用户指定freq; 2. 用户未指定freq, 自动识别
        自动识别时, 通过净值日期和交易日的重合度来判断
        当日度净值的完整度大于treshold时, 认为是日, 否则是周
        """
        if self.freq is None:
            span_date = self.trade_date[
                (self.trade_date >= self.date[0]) & (self.trade_date <= self.date[-1])
            ]
            if np.intersect1d(self.date, span_date).size / span_date.size >= treshold:
                self.freq = "D"
            else:
                self.freq = "W"
        nav_series = pd.Series(self.nav, index=self.date).reindex(
            self.weekly_trade_date if self.freq == "W" else self.trade_date,
        )
        # 如果有缺失需要提醒
        if nav_series.isna().sum() > 0:
            if not self.ffillna:
                if nav_series.isna().sum() > 0:
                    if not self.ffillna:
                        warnings.warn(
                            f"净值数据存在缺失, 共{nav_series.isna().sum()}个交易日缺失净值, 建议设置ffillna=True进行前后填充",
                            UserWarning,
                        )
                    else:
                        nav_series = nav_series.ffill()
                        warnings.warn(
                            f"净值数据存在缺失, 共{nav_series.isna().sum()}个交易日缺失净值, 已进行前后填充",
                            UserWarning,
                        )
        # 恢复净值序列区间
        nav_series = nav_series[
            (nav_series.index >= self.date[0]) & (nav_series.index <= self.date[-1])
        ]
        # 如果此时还有缺失: 1)开头缺失 2)没有使用填充, 均应该报错
        if nav_series.isna().sum() > 0:
            print(nav_series[nav_series.isna()])
            raise ValueError("净值数据开头存在缺失, 请检查数据完整性")
        self.nav = nav_series.values
        self.date = nav_series.index.values
        self.begin_date, self.end_date = self.date[0], self.date[-1]

    def drawdown_info(self) -> Tuple[Optional[NDArray[np.float64]], pd.DataFrame]:
        """
        Return (drawdown_series, drawdown_stats_dict). drawdown_series may be None or an ndarray
        depending on drawdown_stats implementation.
        """
        return drawdown_stats(self.nav, self.date)

    # 计算区间收益, 支持多个区间
    def calculate_interval_return(
        self, interval: list[IntervalReturnETC]
    ) -> list[IntervalReturnETC]:
        """
        Calculate the interval return for the given intervals.
        """
        interval_return_list = []
        for interval_item in interval:
            assert isinstance(
                interval_item, IntervalReturnETC
            ), "interval必须是IntervalReturnETC类型"
            if interval_item.start_date >= interval_item.end_date:
                raise ValueError(
                    f"interval {interval_item.name} 的 start_date 必须早于 end_date"
                )
            if (
                interval_item.start_date >= self.begin_date
                and interval_item.end_date <= self.end_date
                and interval_item.start_date in self.date
                and interval_item.end_date in self.date
            ):
                start_idx = np.where(self.date == interval_item.start_date)[0][0]
                end_idx = np.where(self.date == interval_item.end_date)[0][0]
                interval_nav = self.nav[start_idx : end_idx + 1]
                # 数据不足5条时候, 仅计算区间收益即可
                if len(interval_nav) <= 5:
                    interval_item = interval_item.update(
                        interval_return=interval_nav[-1] / interval_nav[0] - 1,
                        interval_MDD=np.nan,
                    )
                else:
                    interval_metric = curve_analysis(
                        interval_nav,
                        self.date[start_idx : end_idx + 1],
                    )
                    interval_item = interval_item.update(
                        interval_return=interval_metric["区间收益率"],
                        interval_MDD=interval_metric["最大回撤"],
                        interval_sharpe=interval_metric["夏普比率"],
                        interval_karma=interval_metric["卡玛比率"],
                    )
            interval_return_list.append(interval_item)
        return interval_return_list

    @staticmethod
    def generate_intervals(
        last_day: np.datetime64, last_week_day: np.datetime64
    ) -> list[IntervalReturnETC]:
        """
        生成预设的区间列表
        params:
            last_day: 一般为过去的最近一周的最后一个交易日
            last_week_day: last_day 的上一个last_day, 用于计算当周
        return:
            返回预设的区间列表
        """
        _, weekly_trade_date = generate_trading_date(
            last_day - np.timedelta64(380, "D"),
            last_day.astype("datetime64[D]") + np.timedelta64(10, "D"),
        )
        recent_month_day = np.datetime64(
            weekly_trade_date[weekly_trade_date >= last_day - np.timedelta64(30, "D")][
                0
            ]
        )
        year_begin_day = np.datetime64(
            weekly_trade_date[weekly_trade_date >= last_day.astype("datetime64[Y]")][0]
        )
        recent_year_day = np.datetime64(
            weekly_trade_date[weekly_trade_date >= last_day - np.timedelta64(365, "D")][
                0
            ]
        )
        intervals = [
            IntervalReturnETC("recent_week", last_week_day, last_day),
            IntervalReturnETC("recent_month", recent_month_day, last_day),
            IntervalReturnETC("ytd", year_begin_day, last_day),
            IntervalReturnETC("recent_year", recent_year_day, last_day),
            IntervalReturnETC(
                "y2024", np.datetime64("2023-12-29"), np.datetime64("2024-12-27")
            ),
            IntervalReturnETC(
                "y2023", np.datetime64("2022-12-30"), np.datetime64("2023-12-29")
            ),
            IntervalReturnETC(
                "y2022", np.datetime64("2021-12-31"), np.datetime64("2022-12-30")
            ),
        ]
        return intervals


if __name__ == "__main__":
    # Example usage
    nav_data_path = Path(r"data\SAHC74_合骥睿源2000增强1号.csv")
    nav_df = pd.read_csv(nav_data_path)
    nav_df["日期"] = pd.to_datetime(nav_df["日期"])

    metric = NavMetric(
        name=nav_data_path.stem,
        nav=nav_df["累计净值"].to_numpy(dtype=np.float64),
        date=nav_df["日期"].to_numpy(dtype="datetime64[D]"),
    )
    print(metric)
    base_interval = NavMetric.generate_intervals(
        last_week_day=np.datetime64("2025-11-28"),
        last_day=np.datetime64("2025-11-21"),
    )
    base_interval = metric.calculate_interval_return(base_interval)
    for interval in base_interval:
        print(interval)
