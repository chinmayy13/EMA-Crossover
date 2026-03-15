# EMA Crossover Backtest — Quantitative Research Pipeline

A quantitative trading research pipeline implementing an EMA crossover strategy
with vectorised backtesting, walk-forward validation, statistical significance
testing, and an interactive Dash dashboard.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Strategy Explanation](#strategy-explanation)
- [Project Structure](#project-structure)
- [Performance Metrics](#performance-metrics)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Walk-Forward Validation](#walk-forward-validation)
- [Statistical Significance Testing](#statistical-significance-testing)
- [Interactive Dashboard](#interactive-dashboard)
- [Running Tests](#running-tests)
- [Known Limitations](#known-limitations)
- [Tech Stack](#tech-stack)
- [Future Extensions](#future-extensions)
- [Disclaimer](#disclaimer)

---

## Project Overview

| Step | Module             | Description                                                |
| ---- | ------------------ | ---------------------------------------------------------- |
| 1    | `data_loader.py`   | Downloads OHLCV data via yfinance with CSV caching         |
| 2    | `indicators.py`    | EMA and ATR calculations using pandas `ewm`                |
| 3    | `signals.py`       | BUY / SELL signal generation from EMA crossover            |
| 4    | `backtester.py`    | Vectorised backtest engine with look-ahead bias prevention |
| 5    | `metrics.py`       | Sharpe, Sortino, Calmar, CAGR, drawdown, permutation test  |
| 6    | `optimization.py`  | In-sample grid search + walk-forward validation            |
| 7    | `visualization.py` | Matplotlib/Seaborn static + Plotly interactive charts      |
| 8    | `dashboard.py`     | Dash web dashboard with live parameter controls            |

---

## What Changed (v2)

### Bug fixes

- **`validate_data` no longer mutates the input DataFrame in-place.**
  It now returns a clean copy (`dropna` without `inplace=True`).
- **Transaction cost model corrected.**
  Cost is now applied to the executed position (shifted by 1 bar) rather than
  the raw position signal, correctly reflecting the fill-day deduction.
- **`CAGR` function renamed to `cagr`** to comply with PEP 8 naming conventions.
  All callers updated.
- **yfinance MultiIndex columns handled.**
  Modern yfinance may return a MultiIndex when `auto_adjust=True`; the loader
  now flattens it automatically.
- **`__pycache__` and `.DS_Store` removed** from version control.
  A `.gitignore` file now excludes them permanently.

### New features

- **Sortino Ratio** — penalises only downside volatility (more appropriate for long-only strategies).
- **Calmar Ratio** — CAGR / |Max Drawdown|.
- **Realistic risk-free rate** — default changed from 0% to 4.5% (`DEFAULT_RISK_FREE_RATE`).
  Override with `--risk-free-rate 0.0` if needed.
- **Permutation significance test** — tests whether the observed Sharpe is
  statistically distinguishable from a random shuffle of the same returns.
  Enable with `--significance-test`.
- **Walk-forward validation** — re-optimises parameters on rolling training
  windows and reports out-of-sample performance per fold.
  Enable with `--walk-forward` in optimise mode.
- **Rolling Sharpe chart** — `plot_rolling_sharpe` function added.
- **Walk-forward results chart** — bar chart comparing in-sample vs OOS Sharpe per fold.
- **Unit tests** — `tests/test_core.py` with 30+ tests covering all core modules.
- **`src/__init__.py`** added so `src` is a proper Python package.

---

## Strategy Explanation

An **Exponential Moving Average (EMA)** weights recent prices more heavily than
older ones, making it more responsive than a Simple Moving Average.

The **crossover strategy** uses two EMAs:

| Parameter | Default | Role                        |
| --------- | ------- | --------------------------- |
| Short EMA | 10 bars | Fast, trend-sensitive line  |
| Long EMA  | 50 bars | Slow, trend-confirming line |

**Signal rules:**

```
BUY  (go long)  when  EMA_short crosses ABOVE EMA_long
SELL (go flat)  when  EMA_short crosses BELOW EMA_long
```

The strategy is fully invested (100% of capital) when the signal is active
and holds cash otherwise. No short-selling is implemented.

**Look-ahead bias prevention:**
Signals generated at bar t are executed at bar t+1 via `.shift(1)` on the
position series — standard practice in vectorised backtesting.

---

## Project Structure

```
ema_crossover_backtest/
|-- data/                        # Cached CSV price data (auto-created)
|-- notebooks/
|   `-- ema_analysis.ipynb       # Exploratory analysis notebook
|-- outputs/
|   `-- charts/                  # Saved PNG charts (--save-charts flag)
|-- src/
|   |-- __init__.py
|   |-- data_loader.py           # yfinance download + validation + caching
|   |-- indicators.py            # EMA, ATR
|   |-- signals.py               # BUY/SELL signal generation
|   |-- backtester.py            # Vectorised backtest engine
|   |-- metrics.py               # All performance metrics + significance test
|   |-- optimization.py          # Grid search + walk-forward validation
|   |-- visualization.py         # Matplotlib + Plotly charts
|   `-- dashboard.py             # Interactive Dash application
|-- tests/
|   `-- test_core.py             # Unit tests (pytest)
|-- main.py                      # CLI entry-point
|-- requirements.txt
|-- .gitignore
`-- README.md
```

---

## Performance Metrics

| Metric            | Description                                                |
| ----------------- | ---------------------------------------------------------- | ------------ | ---------------------------------- |
| **Sharpe Ratio**  | Annualised excess return / volatility (RFR = 4.5% default) |
| **Sortino Ratio** | Like Sharpe but penalises only downside volatility         |
| **Calmar Ratio**  | CAGR /                                                     | Max Drawdown | — return per unit of drawdown risk |
| **Max Drawdown**  | Largest peak-to-trough equity decline                      |
| **Total Return**  | Final portfolio value vs initial capital                   |
| **CAGR**          | Compound Annual Growth Rate                                |
| **Volatility**    | Annualised standard deviation of daily returns             |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run a backtest

```bash
python main.py --ticker SPY --short 10 --long 50
```

### 3. Run with significance test

```bash
python main.py --ticker SPY --short 10 --long 50 --significance-test
```

### 4. Launch the dashboard

```bash
python main.py --mode dashboard
# Open http://127.0.0.1:8050
```

---

## Usage

```
python main.py [OPTIONS]

Options:
  --mode              {backtest, optimize, dashboard}   (default: backtest)
  --ticker            Asset symbol, e.g. SPY, AAPL      (default: SPY)
  --start             Start date YYYY-MM-DD             (default: 2018-01-01)
  --short             Short EMA span                    (default: 10)
  --long              Long EMA span                     (default: 50)
  --transaction-cost  One-way cost fraction             (default: 0.001)
  --risk-free-rate    Annualised RFR                    (default: 0.045)
  --save-charts       Save PNG charts to outputs/charts/
  --significance-test Run permutation test on Sharpe
  --n-permutations    Permutations for significance test (default: 1000)
  --walk-forward      Run walk-forward validation (optimize mode only)
  --wf-splits         Number of walk-forward folds      (default: 5)
  --port              Dashboard port                    (default: 8050)
  --debug             Enable Dash debug mode
```

### Examples

```bash
# Default backtest
python main.py --ticker SPY --short 10 --long 50

# With significance test and saved charts
python main.py --ticker AAPL --short 12 --long 26 --significance-test --save-charts

# In-sample optimisation + walk-forward validation
python main.py --mode optimize --ticker SPY --walk-forward --save-charts

# Bitcoin backtest (zero RFR)
python main.py --ticker BTC-USD --short 20 --long 100 --start 2019-01-01 --risk-free-rate 0.0
```

---

## Walk-Forward Validation

The `--walk-forward` flag in `optimize` mode produces an **out-of-sample estimate**
of strategy performance, addressing the in-sample data-snooping problem:

1. The dataset is split into n folds (default 5).
2. For each fold, a grid search optimises parameters on the **training portion**.
3. The best in-sample parameters are evaluated on the **held-out test portion**.
4. Results are printed as a table and saved to `outputs/walk_forward_results.csv`.

```bash
python main.py --mode optimize --ticker SPY --walk-forward --wf-splits 5 --save-charts
```

---

## Statistical Significance Testing

The `--significance-test` flag runs a **permutation test** on the Sharpe Ratio:

- Daily returns are shuffled 1,000 times (configurable via `--n-permutations`).
- The fraction of shuffled Sharpes >= the observed Sharpe is the empirical p-value.
- p-value < 0.05 provides evidence that the strategy's performance is unlikely
  to be due to chance alone.

```bash
python main.py --ticker SPY --significance-test --n-permutations 2000
```

> Note: this test assumes i.i.d. returns under the null. Serial correlation in
> equity returns makes p-values approximate. Walk-forward validation is a more
> robust robustness check.

---

## Interactive Dashboard

```bash
python main.py --mode dashboard
```

Features:

- Select any ticker (SPY, AAPL, MSFT, TSLA, BTC-USD, ETH-USD, GLD, QQQ)
- Adjust Short / Long EMA and start date
- Live metrics: Sharpe, Sortino, Calmar, Max Drawdown, Total Return, CAGR, Volatility
- Interactive candlestick + EMA + signal chart
- Equity curve with drawdown sub-panel

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover: `indicators`, `signals`, `backtester`, `metrics` (including the
permutation test), and a full end-to-end integration smoke test (30+ tests).

---

## Known Limitations

| Limitation                          | Description                                                                        |
| ----------------------------------- | ---------------------------------------------------------------------------------- |
| **In-sample optimisation**          | `grid_search` results are in-sample. Use `--walk-forward` for OOS validation.      |
| **Binary position sizing**          | Strategy is fully invested (1) or fully in cash (0). No fractional sizing.         |
| **No slippage**                     | Fills assumed at close price. Real execution incurs spread and market impact.      |
| **Equity markets only**             | `trading_days=252` assumes equity market hours. Crypto (24/7) requires adjustment. |
| **No short selling**                | Strategy captures uptrends only; sits in cash during downtrends.                   |
| **Permutation test is approximate** | Serial correlation in returns means p-values are not exact.                        |

---

## Tech Stack

| Library                     | Version   | Purpose               |
| --------------------------- | --------- | --------------------- |
| `pandas`                    | >= 2.0    | Data manipulation     |
| `numpy`                     | >= 1.24   | Numerical computation |
| `yfinance`                  | >= 0.2.28 | Market data download  |
| `matplotlib`                | >= 3.7    | Static charting       |
| `seaborn`                   | >= 0.12   | Heatmap visualisation |
| `plotly`                    | >= 5.15   | Interactive charts    |
| `dash`                      | >= 2.11   | Web dashboard         |
| `dash-bootstrap-components` | >= 1.4    | Dashboard UI styling  |
| `scipy`                     | >= 1.11   | Statistical utilities |
| `pytest`                    | >= 7.4    | Unit testing          |

---

## Future Extensions

| Enhancement                 | Description                                            |
| --------------------------- | ------------------------------------------------------ |
| **Position sizing**         | Kelly criterion or ATR-normalised sizing               |
| **Stop-loss / Take-profit** | Risk management rules in signal logic                  |
| **Multi-asset portfolio**   | Basket strategy with equal or vol-weighted allocation  |
| **ML signal prediction**    | Logistic regression / XGBoost on engineered features   |
| **Regime detection**        | Hidden Markov Model to conditionally activate strategy |
| **Live paper trading**      | Connect to Alpaca or IBKR paper trading API            |
| **Slippage modelling**      | Bid-ask spread and market impact on fill prices        |

---

## Disclaimer

This project is built for **educational and portfolio demonstration purposes only**.
It does not constitute financial advice. Past performance of a backtested strategy
is not indicative of future results. Always perform thorough due diligence before
making any investment decisions.

---

_Built with Python · pandas · yfinance · Dash · Plotly_
