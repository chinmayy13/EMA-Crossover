"""
tests/test_core.py
------------------
Unit tests for the EMA Crossover Backtest project.

Run with:
    pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

# Ensure the project root is on the path when running from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.indicators import ema, add_emas
from src.signals import generate_signals
from src.backtester import run_backtest
from src.metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    total_return,
    cagr,
    calmar_ratio,
    annualised_volatility,
    compute_all_metrics,
    permutation_test,
    DEFAULT_RISK_FREE_RATE,
    TRADING_DAYS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_price_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with a random-walk close price."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    high  = close * (1 + rng.uniform(0, 0.01, n))
    low   = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    vol   = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


@pytest.fixture
def price_df():
    return _make_price_df()


@pytest.fixture
def backtest_df(price_df):
    df = add_emas(price_df, short=10, long=50)
    df = generate_signals(df)
    df = run_backtest(df)
    return df


# ---------------------------------------------------------------------------
# indicators.py
# ---------------------------------------------------------------------------

class TestIndicators:

    def test_ema_length(self, price_df):
        result = ema(price_df["Close"], span=10)
        assert len(result) == len(price_df)

    def test_ema_no_nan_after_warmup(self, price_df):
        result = ema(price_df["Close"], span=10)
        # EMA with adjust=False has no NaN (starts from first value)
        assert result.isna().sum() == 0

    def test_ema_invalid_span(self, price_df):
        with pytest.raises(ValueError):
            ema(price_df["Close"], span=0)

    def test_add_emas_columns(self, price_df):
        df = add_emas(price_df, short=10, long=50)
        assert "EMA_short" in df.columns
        assert "EMA_long" in df.columns

    def test_add_emas_short_lt_long(self, price_df):
        with pytest.raises(ValueError):
            add_emas(price_df, short=50, long=10)

    def test_add_emas_equal_raises(self, price_df):
        with pytest.raises(ValueError):
            add_emas(price_df, short=20, long=20)

    def test_add_emas_does_not_mutate(self, price_df):
        original_cols = list(price_df.columns)
        add_emas(price_df, short=10, long=50)
        assert list(price_df.columns) == original_cols


# ---------------------------------------------------------------------------
# signals.py
# ---------------------------------------------------------------------------

class TestSignals:

    def test_signal_columns_exist(self, price_df):
        df = add_emas(price_df, short=10, long=50)
        df = generate_signals(df)
        assert "Position" in df.columns
        assert "Signal" in df.columns

    def test_position_values(self, price_df):
        df = add_emas(price_df, short=10, long=50)
        df = generate_signals(df)
        assert set(df["Position"].unique()).issubset({0, 1})

    def test_signal_values(self, price_df):
        df = add_emas(price_df, short=10, long=50)
        df = generate_signals(df)
        assert set(df["Signal"].unique()).issubset({-1, 0, 1})

    def test_signal_fires_on_crossover(self):
        """Signal should be +1 exactly when EMA_short crosses above EMA_long."""
        idx = pd.date_range("2020-01-01", periods=6, freq="B")
        df = pd.DataFrame({
            "Close": [100] * 6,
            "EMA_short": [1, 1, 2, 2, 1, 1],
            "EMA_long":  [2, 2, 1, 1, 2, 2],
        }, index=idx)
        df = generate_signals(df)
        # Transition at index 2: short goes from <= long to > long -> Signal=+1
        assert df["Signal"].iloc[2] == 1
        # Transition at index 4: short goes from > long to <= long -> Signal=-1
        assert df["Signal"].iloc[4] == -1

    def test_generate_signals_missing_ema_raises(self, price_df):
        with pytest.raises(ValueError):
            generate_signals(price_df)

    def test_does_not_mutate_input(self, price_df):
        df = add_emas(price_df, short=10, long=50)
        original_cols = list(df.columns)
        generate_signals(df)
        assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# backtester.py
# ---------------------------------------------------------------------------

class TestBacktester:

    def test_output_columns(self, backtest_df):
        for col in ["Market_Return", "Strategy_Return", "Gross_Return",
                    "Transaction_Cost", "Equity_Curve", "BnH_Equity", "Drawdown"]:
            assert col in backtest_df.columns, f"Missing column: {col}"

    def test_equity_starts_at_initial_capital(self, price_df):
        df = add_emas(price_df, short=10, long=50)
        df = generate_signals(df)
        bt = run_backtest(df, initial_capital=10_000)
        # First row of equity should be close to initial capital
        # (small deviation because cumprod starts from row after NaN)
        assert abs(bt["Equity_Curve"].iloc[0] - 10_000) / 10_000 < 0.05

    def test_no_nan_in_equity(self, backtest_df):
        assert backtest_df["Equity_Curve"].isna().sum() == 0

    def test_drawdown_non_positive(self, backtest_df):
        assert (backtest_df["Drawdown"] <= 0.0).all()

    def test_drawdown_bounded(self, backtest_df):
        assert (backtest_df["Drawdown"] >= -1.0).all()

    def test_look_ahead_bias_prevention(self, price_df):
        """Strategy return at bar t should use position from bar t-1."""
        df = add_emas(price_df, short=10, long=50)
        df = generate_signals(df)
        bt = run_backtest(df)
        # On a bar where position changes, the NEW position is not yet used.
        # Verify by checking that Strategy_Return == Market_Return * lagged_position
        lagged_pos = df["Position"].shift(1).reindex(bt.index)
        expected_gross = bt["Market_Return"] * lagged_pos
        diff = (bt["Gross_Return"] - expected_gross).abs().dropna()
        assert (diff < 1e-10).all()

    def test_transaction_cost_non_negative(self, backtest_df):
        assert (backtest_df["Transaction_Cost"] >= 0).all()

    def test_missing_columns_raises(self, price_df):
        with pytest.raises((ValueError, KeyError)):
            run_backtest(price_df)  # no Position column


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_sharpe_zero_variance(self):
        returns = pd.Series([0.001] * 100)
        sr = sharpe_ratio(returns)
        # Constant returns -> std is near zero -> nan
        assert np.isnan(sr) or sr > 0

    def test_sharpe_positive_for_positive_returns(self):
        returns = pd.Series([0.005] * 252)
        sr = sharpe_ratio(returns, risk_free_rate=0.0)
        assert sr > 0

    def test_sharpe_uses_risk_free_rate(self):
        returns = pd.Series([0.001] * 252)
        sr_0  = sharpe_ratio(returns, risk_free_rate=0.0)
        sr_45 = sharpe_ratio(returns, risk_free_rate=0.045)
        assert sr_0 > sr_45  # higher RFR -> lower excess return -> lower Sharpe

    def test_sortino_only_penalises_downside(self):
        # Identical upside/downside vol: Sortino should be higher than Sharpe
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0.001, 0.01, 500))
        sr = sharpe_ratio(returns, risk_free_rate=0.0)
        so = sortino_ratio(returns, risk_free_rate=0.0)
        assert so >= sr or np.isnan(so)  # Sortino >= Sharpe for positive mean

    def test_max_drawdown_is_negative(self, backtest_df):
        mdd = max_drawdown(backtest_df["Equity_Curve"])
        assert mdd <= 0

    def test_max_drawdown_flat_series(self):
        equity = pd.Series([100.0] * 100)
        assert max_drawdown(equity) == 0.0

    def test_total_return_calculation(self):
        equity = pd.Series([100.0, 150.0])
        assert abs(total_return(equity) - 0.5) < 1e-9

    def test_cagr_flat_series(self):
        equity = pd.Series([100.0] * TRADING_DAYS)
        assert abs(cagr(equity)) < 1e-9

    def test_cagr_doubles_in_one_year(self):
        equity = pd.Series(
            np.linspace(100, 200, TRADING_DAYS + 1)
        )
        c = cagr(equity)
        assert abs(c - 1.0) < 0.05  # ~100% CAGR

    def test_calmar_ratio_sign(self, backtest_df):
        cr = calmar_ratio(backtest_df["Equity_Curve"])
        # Sign of Calmar should match sign of CAGR
        c = cagr(backtest_df["Equity_Curve"])
        if not np.isnan(cr) and not np.isnan(c):
            assert np.sign(cr) == np.sign(c)

    def test_annualised_volatility_positive(self):
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(0, 0.01, 300))
        assert annualised_volatility(returns) > 0

    def test_compute_all_metrics_keys(self, backtest_df):
        metrics = compute_all_metrics(backtest_df)
        for key in ["Sharpe", "Sortino", "Calmar", "Max_Drawdown",
                    "Total_Return", "CAGR", "Volatility",
                    "BnH_Total_Return", "BnH_Sharpe"]:
            assert key in metrics, f"Missing metric: {key}"

    def test_default_risk_free_rate_positive(self):
        assert DEFAULT_RISK_FREE_RATE > 0

    def test_permutation_test_returns_valid_p_value(self, backtest_df):
        _, p_value, _ = permutation_test(
            backtest_df["Strategy_Return"],
            n_permutations=200,
            random_state=0,
        )
        assert 0.0 <= p_value <= 1.0

    def test_permutation_test_reproducible(self, backtest_df):
        returns = backtest_df["Strategy_Return"]
        _, p1, _ = permutation_test(returns, n_permutations=100, random_state=7)
        _, p2, _ = permutation_test(returns, n_permutations=100, random_state=7)
        assert p1 == p2


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_full_pipeline_runs(self, price_df):
        """End-to-end: price -> EMA -> signals -> backtest -> metrics."""
        df = add_emas(price_df, short=10, long=50)
        df = generate_signals(df)
        df = run_backtest(df)
        metrics = compute_all_metrics(df)
        assert isinstance(metrics["Sharpe"], float)
        assert isinstance(metrics["CAGR"], float)
        assert len(df) > 0

    def test_different_ema_pairs(self, price_df):
        for short, long in [(5, 20), (10, 50), (20, 100)]:
            df = add_emas(price_df, short=short, long=long)
            df = generate_signals(df)
            df = run_backtest(df)
            assert len(df) > 0, f"Empty result for EMA({short},{long})"
