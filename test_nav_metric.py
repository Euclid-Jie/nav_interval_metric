import numpy as np
import pandas as pd
import pytest
from nav_metric import IntervalReturnETC, NavMetric
from utils import curve_analysis, drawdown_stats, generate_trading_date

# --- fixtures ---


def make_daily_nav(start="2021-01-04", periods=500, growth=0.0003):
    """Monotonically growing daily nav on trading-like dates."""
    dates = pd.bdate_range(start, periods=periods).values.astype("datetime64[D]")
    nav = np.cumprod(np.full(periods, 1 + growth))
    return nav, dates


def make_weekly_nav(start="2021-01-08", periods=100, growth=0.001):
    # Use actual weekly trading dates so reindex in format_nav finds matches
    _, wtd = generate_trading_date(
        np.datetime64("2021-01-01"), np.datetime64("2023-12-31")
    )
    dates = wtd[:periods]
    nav = np.cumprod(np.full(len(dates), 1 + growth))
    return nav, dates


# --- utils: curve_analysis ---


class TestCurveAnalysis:
    def test_keys_present(self):
        nav, dates = make_daily_nav(periods=50)
        result = curve_analysis(nav, dates)
        for key in [
            "区间收益率",
            "年化收益率",
            "年化波动率",
            "夏普比率",
            "最大回撤",
            "卡玛比率",
        ]:
            assert key in result

    def test_flat_nav_zero_return(self):
        dates = np.array(
            ["2021-01-04", "2021-06-01", "2021-12-31"], dtype="datetime64[D]"
        )
        nav = np.array([1.0, 1.0, 1.0])
        result = curve_analysis(nav, dates)
        assert result["区间收益率"] == pytest.approx(0.0)
        assert result["最大回撤"] == pytest.approx(0.0)

    def test_monotone_growth_no_drawdown(self):
        nav, dates = make_daily_nav(periods=100)
        result = curve_analysis(nav, dates)
        assert result["区间收益率"] > 0
        assert result["最大回撤"] == pytest.approx(0.0, abs=1e-10)

    def test_drawdown_detected(self):
        nav = np.array([1.0, 1.1, 0.9, 1.0, 1.2])
        dates = np.array(
            ["2021-01-04", "2021-03-01", "2021-06-01", "2021-09-01", "2021-12-31"],
            dtype="datetime64[D]",
        )
        result = curve_analysis(nav, dates)
        assert result["最大回撤"] < 0

    def test_invalid_nav_raises(self):
        nav = np.array([1.0, np.nan, 1.1])
        dates = np.array(
            ["2021-01-04", "2021-06-01", "2021-12-31"], dtype="datetime64[D]"
        )
        with pytest.raises(AssertionError):
            curve_analysis(nav, dates)

    def test_too_short_raises(self):
        with pytest.raises(AssertionError):
            curve_analysis(
                np.array([1.0, 1.1]),
                np.array(["2021-01-04", "2021-06-01"], dtype="datetime64[D]"),
            )


# --- utils: drawdown_stats ---


class TestDrawdownStats:
    def test_no_drawdown_returns_none(self):
        nav, dates = make_daily_nav(periods=50)
        dd_series, dd_df = drawdown_stats(nav, dates)
        assert dd_series is None
        assert dd_df.empty

    def test_single_drawdown(self):
        nav = np.array([1.0, 1.2, 0.8, 1.0, 1.3])
        dates = np.array(
            ["2021-01-04", "2021-03-01", "2021-06-01", "2021-09-01", "2021-12-31"],
            dtype="datetime64[D]",
        )
        dd_series, dd_df = drawdown_stats(nav, dates)
        assert dd_series is not None
        assert len(dd_df) == 1
        assert dd_df["max_drawdown"].iloc[0] < 0

    def test_unrecovered_drawdown_has_nat_end(self):
        nav = np.array([1.0, 1.2, 0.9, 0.8])
        dates = np.array(
            ["2021-01-04", "2021-03-01", "2021-06-01", "2021-09-01"],
            dtype="datetime64[D]",
        )
        _, dd_df = drawdown_stats(nav, dates)
        assert pd.isna(dd_df["drawdown_end_date"].iloc[-1])


# --- utils: generate_trading_date ---


class TestGenerateTradingDate:
    def test_returns_two_arrays(self):
        td, wtd = generate_trading_date(
            np.datetime64("2024-01-01"), np.datetime64("2024-03-31")
        )
        assert td.dtype == np.dtype("datetime64[D]")
        assert wtd.dtype == np.dtype("datetime64[D]")

    def test_no_weekends_in_trading_date(self):
        td, _ = generate_trading_date(
            np.datetime64("2024-01-01"), np.datetime64("2024-03-31")
        )
        weekdays = np.array(
            [np.datetime64(d, "D").astype(object).weekday() for d in td]
        )
        assert (weekdays < 5).all()

    def test_weekly_dates_are_fridays_or_adjusted(self):
        _, wtd = generate_trading_date(
            np.datetime64("2024-01-01"), np.datetime64("2024-06-30")
        )
        valid = wtd[~pd.isnull(wtd.astype(object))]
        weekdays = np.array(
            [np.datetime64(d, "D").astype(object).weekday() for d in valid]
        )
        assert (weekdays < 5).all()

    def test_begin_before_2015_raises(self):
        with pytest.raises(AssertionError):
            generate_trading_date(
                np.datetime64("2014-12-31"), np.datetime64("2015-01-10")
            )


# --- NavMetric ---


class TestNavMetric:
    def setup_method(self):
        self.nav, self.dates = make_daily_nav(start="2021-01-04", periods=500)
        self.metric = NavMetric("test", self.nav, self.dates)

    def test_repr(self):
        assert "test" in repr(self.metric)

    def test_freq_auto_detected_daily(self):
        assert self.metric.freq == "D"

    def test_freq_auto_detected_weekly(self):
        nav, dates = make_weekly_nav()
        m = NavMetric("weekly", nav, dates)
        assert m.freq == "W"

    def test_freq_explicit(self):
        nav, dates = make_weekly_nav()
        m = NavMetric("explicit", nav, dates, freq="W")
        assert m.freq == "W"

    def test_pre2020_data_truncated(self):
        nav, dates = make_daily_nav(start="2019-01-04", periods=600)
        m = NavMetric("old", nav, dates)
        assert m.begin_date >= np.datetime64("2020-01-01")

    def test_base_metric_dict_populated(self):
        assert "区间收益率" in self.metric.base_metric_dict

    def test_date_index_consistent(self):
        for d, i in self.metric._date_index.items():
            assert self.metric.date[i] == d


# --- NavMetric.calculate_interval_return ---


class TestCalculateIntervalReturn:
    def setup_method(self):
        nav, dates = make_daily_nav(start="2021-01-04", periods=500)
        self.metric = NavMetric("test", nav, dates)
        self.begin = self.metric.begin_date
        self.end = self.metric.end_date

    def _first_date_after(self, ref):
        return self.metric.date[self.metric.date > ref][0]

    def test_interval_within_range_computed(self):
        d0 = self.metric.date[10]
        d1 = self.metric.date[100]
        result = self.metric.calculate_interval_return([IntervalReturnETC("t", d0, d1)])
        assert not np.isnan(result[0].interval_return)

    def test_interval_outside_range_unchanged(self):
        item = IntervalReturnETC(
            "out", np.datetime64("2015-01-05"), np.datetime64("2015-06-01")
        )
        result = self.metric.calculate_interval_return([item])
        assert np.isnan(result[0].interval_return)

    def test_short_interval_only_return_no_mdd(self):
        d0 = self.metric.date[0]
        d1 = self.metric.date[3]  # 4 points <= 5
        result = self.metric.calculate_interval_return(
            [IntervalReturnETC("short", d0, d1)]
        )
        assert not np.isnan(result[0].interval_return)
        assert np.isnan(result[0].interval_MDD)

    def test_invalid_date_order_raises(self):
        d0 = self.metric.date[50]
        d1 = self.metric.date[10]
        with pytest.raises(ValueError):
            self.metric.calculate_interval_return([IntervalReturnETC("bad", d0, d1)])

    def test_multiple_intervals(self):
        d = self.metric.date
        items = [
            IntervalReturnETC("a", d[0], d[50]),
            IntervalReturnETC("b", d[50], d[200]),
        ]
        results = self.metric.calculate_interval_return(items)
        assert len(results) == 2
        assert all(not np.isnan(r.interval_return) for r in results)


# --- NavMetric.generate_intervals ---


class TestGenerateIntervals:
    def test_returns_seven_intervals(self):
        intervals = NavMetric.generate_intervals(
            last_day=np.datetime64("2025-01-24"),
            last_week_day=np.datetime64("2025-01-17"),
        )
        assert len(intervals) == 7

    def test_all_start_before_end(self):
        intervals = NavMetric.generate_intervals(
            last_day=np.datetime64("2025-01-24"),
            last_week_day=np.datetime64("2025-01-17"),
        )
        for iv in intervals:
            assert iv.start_date < iv.end_date
import numpy as np
import pandas as pd
import pytest
from nav_metric import IntervalReturnETC, NavMetric
from utils import curve_analysis, drawdown_stats, generate_trading_date


# --- fixtures ---

def make_daily_nav(start="2021-01-04", periods=500, growth=0.0003):
    """Monotonically growing daily nav on trading-like dates."""
    dates = pd.bdate_range(start, periods=periods).values.astype("datetime64[D]")
    nav = np.cumprod(np.full(periods, 1 + growth))
    return nav, dates


def make_weekly_nav(start="2021-01-08", periods=100, growth=0.001):
    # Use actual weekly trading dates so reindex in format_nav finds matches
    _, wtd = generate_trading_date(np.datetime64("2021-01-01"), np.datetime64("2023-12-31"))
    dates = wtd[:periods]
    nav = np.cumprod(np.full(len(dates), 1 + growth))
    return nav, dates


# --- utils: curve_analysis ---

class TestCurveAnalysis:
    def test_keys_present(self):
        nav, dates = make_daily_nav(periods=50)
        result = curve_analysis(nav, dates)
        for key in ["区间收益率", "年化收益率", "年化波动率", "夏普比率", "最大回撤", "卡玛比率"]:
            assert key in result

    def test_flat_nav_zero_return(self):
        dates = np.array(["2021-01-04", "2021-06-01", "2021-12-31"], dtype="datetime64[D]")
        nav = np.array([1.0, 1.0, 1.0])
        result = curve_analysis(nav, dates)
        assert result["区间收益率"] == pytest.approx(0.0)
        assert result["最大回撤"] == pytest.approx(0.0)

    def test_monotone_growth_no_drawdown(self):
        nav, dates = make_daily_nav(periods=100)
        result = curve_analysis(nav, dates)
        assert result["区间收益率"] > 0
        assert result["最大回撤"] == pytest.approx(0.0, abs=1e-10)

    def test_drawdown_detected(self):
        nav = np.array([1.0, 1.1, 0.9, 1.0, 1.2])
        dates = np.array(["2021-01-04", "2021-03-01", "2021-06-01", "2021-09-01", "2021-12-31"], dtype="datetime64[D]")
        result = curve_analysis(nav, dates)
        assert result["最大回撤"] < 0

    def test_invalid_nav_raises(self):
        nav = np.array([1.0, np.nan, 1.1])
        dates = np.array(["2021-01-04", "2021-06-01", "2021-12-31"], dtype="datetime64[D]")
        with pytest.raises(AssertionError):
            curve_analysis(nav, dates)

    def test_too_short_raises(self):
        with pytest.raises(AssertionError):
            curve_analysis(np.array([1.0, 1.1]), np.array(["2021-01-04", "2021-06-01"], dtype="datetime64[D]"))


# --- utils: drawdown_stats ---

class TestDrawdownStats:
    def test_no_drawdown_returns_none(self):
        nav, dates = make_daily_nav(periods=50)
        dd_series, dd_df = drawdown_stats(nav, dates)
        assert dd_series is None
        assert dd_df.empty

    def test_single_drawdown(self):
        nav = np.array([1.0, 1.2, 0.8, 1.0, 1.3])
        dates = np.array(["2021-01-04", "2021-03-01", "2021-06-01", "2021-09-01", "2021-12-31"], dtype="datetime64[D]")
        dd_series, dd_df = drawdown_stats(nav, dates)
        assert dd_series is not None
        assert len(dd_df) == 1
        assert dd_df["max_drawdown"].iloc[0] < 0

    def test_unrecovered_drawdown_has_nat_end(self):
        nav = np.array([1.0, 1.2, 0.9, 0.8])
        dates = np.array(["2021-01-04", "2021-03-01", "2021-06-01", "2021-09-01"], dtype="datetime64[D]")
        _, dd_df = drawdown_stats(nav, dates)
        assert pd.isna(dd_df["drawdown_end_date"].iloc[-1])


# --- utils: generate_trading_date ---

class TestGenerateTradingDate:
    def test_returns_two_arrays(self):
        td, wtd = generate_trading_date(np.datetime64("2024-01-01"), np.datetime64("2024-03-31"))
        assert td.dtype == np.dtype("datetime64[D]")
        assert wtd.dtype == np.dtype("datetime64[D]")

    def test_no_weekends_in_trading_date(self):
        td, _ = generate_trading_date(np.datetime64("2024-01-01"), np.datetime64("2024-03-31"))
        weekdays = np.array([np.datetime64(d, "D").astype(object).weekday() for d in td])
        assert (weekdays < 5).all()

    def test_weekly_dates_are_fridays_or_adjusted(self):
        _, wtd = generate_trading_date(np.datetime64("2024-01-01"), np.datetime64("2024-06-30"))
        valid = wtd[~pd.isnull(wtd.astype(object))]
        weekdays = np.array([np.datetime64(d, "D").astype(object).weekday() for d in valid])
        assert (weekdays < 5).all()

    def test_begin_before_2015_raises(self):
        with pytest.raises(AssertionError):
            generate_trading_date(np.datetime64("2014-12-31"), np.datetime64("2015-01-10"))


# --- NavMetric ---

class TestNavMetric:
    def setup_method(self):
        self.nav, self.dates = make_daily_nav(start="2021-01-04", periods=500)
        self.metric = NavMetric("test", self.nav, self.dates)

    def test_repr(self):
        assert "test" in repr(self.metric)

    def test_freq_auto_detected_daily(self):
        assert self.metric.freq == "D"

    def test_freq_auto_detected_weekly(self):
        nav, dates = make_weekly_nav()
        m = NavMetric("weekly", nav, dates)
        assert m.freq == "W"

    def test_freq_explicit(self):
        nav, dates = make_weekly_nav()
        m = NavMetric("explicit", nav, dates, freq="W")
        assert m.freq == "W"

    def test_pre2020_data_truncated(self):
        nav, dates = make_daily_nav(start="2019-01-04", periods=600)
        m = NavMetric("old", nav, dates)
        assert m.begin_date >= np.datetime64("2020-01-01")

    def test_base_metric_dict_populated(self):
        assert "区间收益率" in self.metric.base_metric_dict

    def test_date_index_consistent(self):
        for d, i in self.metric._date_index.items():
            assert self.metric.date[i] == d


# --- NavMetric.calculate_interval_return ---

class TestCalculateIntervalReturn:
    def setup_method(self):
        nav, dates = make_daily_nav(start="2021-01-04", periods=500)
        self.metric = NavMetric("test", nav, dates)
        self.begin = self.metric.begin_date
        self.end = self.metric.end_date

    def _first_date_after(self, ref):
        return self.metric.date[self.metric.date > ref][0]

    def test_interval_within_range_computed(self):
        d0 = self.metric.date[10]
        d1 = self.metric.date[100]
        result = self.metric.calculate_interval_return(
            [IntervalReturnETC("t", d0, d1)]
        )
        assert not np.isnan(result[0].interval_return)

    def test_interval_outside_range_unchanged(self):
        item = IntervalReturnETC("out", np.datetime64("2015-01-05"), np.datetime64("2015-06-01"))
        result = self.metric.calculate_interval_return([item])
        assert np.isnan(result[0].interval_return)

    def test_short_interval_only_return_no_mdd(self):
        d0 = self.metric.date[0]
        d1 = self.metric.date[3]  # 4 points <= 5
        result = self.metric.calculate_interval_return(
            [IntervalReturnETC("short", d0, d1)]
        )
        assert not np.isnan(result[0].interval_return)
        assert np.isnan(result[0].interval_MDD)

    def test_invalid_date_order_raises(self):
        d0 = self.metric.date[50]
        d1 = self.metric.date[10]
        with pytest.raises(ValueError):
            self.metric.calculate_interval_return([IntervalReturnETC("bad", d0, d1)])

    def test_multiple_intervals(self):
        d = self.metric.date
        items = [
            IntervalReturnETC("a", d[0], d[50]),
            IntervalReturnETC("b", d[50], d[200]),
        ]
        results = self.metric.calculate_interval_return(items)
        assert len(results) == 2
        assert all(not np.isnan(r.interval_return) for r in results)


# --- NavMetric.generate_intervals ---

class TestGenerateIntervals:
    def test_returns_seven_intervals(self):
        intervals = NavMetric.generate_intervals(
            last_day=np.datetime64("2025-01-24"),
            last_week_day=np.datetime64("2025-01-17"),
        )
        assert len(intervals) == 7

    def test_all_start_before_end(self):
        intervals = NavMetric.generate_intervals(
            last_day=np.datetime64("2025-01-24"),
            last_week_day=np.datetime64("2025-01-17"),
        )
        for iv in intervals:
            assert iv.start_date < iv.end_date
