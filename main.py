"""
main.py
-------
CLI entry-point for the EMA Crossover Backtest project.

Modes
-----
backtest   : Run a single EMA(short, long) backtest and print metrics.
optimize   : In-sample grid search + optional walk-forward validation.
dashboard  : Launch interactive Dash web dashboard.

Usage
-----
python main.py --help
python main.py --ticker SPY --short 10 --long 50
python main.py --mode optimize --ticker AAPL --walk-forward --save-charts
python main.py --mode dashboard
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    for d in ("data", "outputs/charts"):
        Path(d).mkdir(parents=True, exist_ok=True)


def _print_metrics(metrics: dict, short: int, long: int, ticker: str) -> None:
    rfr = metrics.get("Risk_Free_Rate", 0.045)

    print("\n" + "=" * 60)
    print(f"  {ticker}  |  EMA({short}, {long})  |  RFR={rfr*100:.1f}%")
    print("=" * 60)

    rows = [
        ("Sharpe Ratio",       metrics.get("Sharpe"),          False),
        ("Sortino Ratio",      metrics.get("Sortino"),         False),
        ("Calmar Ratio",       metrics.get("Calmar"),          False),
        ("Max Drawdown",       metrics.get("Max_Drawdown"),    True),
        ("Total Return",       metrics.get("Total_Return"),    True),
        ("CAGR",               metrics.get("CAGR"),            True),
        ("Volatility (ann.)",  metrics.get("Volatility"),      True),
        ("BnH Total Return",   metrics.get("BnH_Total_Return"), True),
        ("BnH Sharpe",         metrics.get("BnH_Sharpe"),      False),
        ("N Trades",           metrics.get("N_Trades"),        None),
        ("N Bars",             metrics.get("N_Bars"),          None),
    ]

    for label, val, is_pct in rows:
        if val is None:
            formatted = "N/A"
        elif is_pct is None:
            formatted = str(int(val)) if val is not None else "N/A"
        elif is_pct:
            formatted = f"{val * 100:.2f} %"
        else:
            formatted = f"{val:.3f}"
        print(f"  {label:<24} {formatted:>12}")

    # Significance test results (optional)
    if "Permutation_P_Value" in metrics:
        pval = metrics["Permutation_P_Value"]
        sig  = metrics.get("Is_Significant_5pct", False)
        s95  = metrics.get("Null_Sharpe_95pct", float("nan"))
        print("  " + "-" * 36)
        print(f"  {'Permutation p-value':<24} {pval:>12.4f}")
        print(f"  {'Null 95th pct Sharpe':<24} {s95:>12.3f}")
        print(f"  {'Significant @ 5%':<24} {'YES' if sig else 'NO':>12}")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_backtest_mode(args) -> None:
    from src.data_loader import download_data, validate_data
    from src.indicators import add_emas
    from src.signals import generate_signals, get_trade_events
    from src.backtester import run_backtest
    from src.metrics import compute_all_metrics, DEFAULT_RISK_FREE_RATE
    from src.visualization import (
        plot_price_with_signals,
        plot_equity_curve,
        plot_rolling_sharpe,
    )

    rfr = args.risk_free_rate if args.risk_free_rate is not None else DEFAULT_RISK_FREE_RATE
    print(f"\n[Main] Backtest mode -> {args.ticker} | EMA({args.short},{args.long}) | RFR={rfr*100:.1f}%")

    raw = download_data(args.ticker, start=args.start)
    df  = validate_data(raw)

    df = add_emas(df, args.short, args.long)
    df = generate_signals(df)
    df = run_backtest(df, transaction_cost=args.transaction_cost)

    metrics = compute_all_metrics(
        df,
        risk_free_rate=rfr,
        run_permutation_test=args.significance_test,
        n_permutations=args.n_permutations,
    )

    _print_metrics(metrics, args.short, args.long, args.ticker)

    trades = get_trade_events(df)
    print(f"[Main] BUY signals:  {(trades['Type'] == 'BUY').sum()}")
    print(f"[Main] SELL signals: {(trades['Type'] == 'SELL').sum()}\n")

    price_path   = "outputs/charts/price_signals.png"  if args.save_charts else None
    equity_path  = "outputs/charts/equity_curve.png"   if args.save_charts else None
    rolling_path = "outputs/charts/rolling_sharpe.png" if args.save_charts else None

    plot_price_with_signals(df, ticker=args.ticker, short=args.short, long=args.long,
                            save_path=price_path)
    plot_equity_curve(df, ticker=args.ticker, save_path=equity_path)
    plot_rolling_sharpe(df, ticker=args.ticker, save_path=rolling_path)

    if args.save_charts:
        print("[Main] Charts saved to outputs/charts/")


def run_optimize_mode(args) -> None:
    from src.data_loader import download_data, validate_data
    from src.optimization import grid_search, best_params, build_heatmap_pivot, walk_forward_optimize
    from src.visualization import plot_optimization_heatmap, plot_walk_forward_results
    from src.metrics import DEFAULT_RISK_FREE_RATE

    rfr = args.risk_free_rate if args.risk_free_rate is not None else DEFAULT_RISK_FREE_RATE
    print(f"\n[Main] Optimisation mode -> {args.ticker} | RFR={rfr*100:.1f}%")
    print("[Main] NOTE: grid_search results are IN-SAMPLE. "
          "Use --walk-forward for out-of-sample validation.\n")

    raw = download_data(args.ticker, start=args.start)
    df  = validate_data(raw)

    opt_df = grid_search(
        df,
        short_range=range(5, 51, 5),
        long_range=range(20, 201, 10),
        risk_free_rate=rfr,
        transaction_cost=args.transaction_cost,
    )

    bs, bl = best_params(opt_df)
    print(f"\n[Main] Best params (in-sample): Short EMA={bs}, Long EMA={bl}")
    print(f"[Main] Best in-sample Sharpe : {opt_df.iloc[0]['Sharpe']:.3f}")
    print(f"[Main] Best in-sample Return : {opt_df.iloc[0]['Total_Return']*100:.2f} %\n")
    print("[Main] Top 10 combinations:")
    print(opt_df.head(10).to_string(index=False))

    pivot = build_heatmap_pivot(opt_df, "Sharpe")
    hm_path = "outputs/charts/heatmap.png" if args.save_charts else None
    plot_optimization_heatmap(pivot, "Sharpe", save_path=hm_path)

    if args.save_charts:
        opt_df.to_csv("outputs/optimization_results.csv", index=False)
        print("[Main] Results saved to outputs/optimization_results.csv")

    # Walk-forward validation
    if args.walk_forward:
        print("\n[Main] Running walk-forward validation ...")
        wf_df = walk_forward_optimize(
            df,
            short_range=range(5, 51, 5),
            long_range=range(20, 201, 10),
            n_splits=args.wf_splits,
            risk_free_rate=rfr,
            transaction_cost=args.transaction_cost,
        )
        print("\n[Main] Walk-forward results:")
        print(wf_df.to_string(index=False))

        wf_path = "outputs/charts/walk_forward.png" if args.save_charts else None
        plot_walk_forward_results(wf_df, save_path=wf_path)

        if args.save_charts:
            wf_df.to_csv("outputs/walk_forward_results.csv", index=False)
            print("[Main] Walk-forward results saved to outputs/walk_forward_results.csv")


def run_dashboard_mode(args) -> None:
    from src.dashboard import run_server
    run_server(debug=args.debug, port=args.port)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EMA Crossover Backtest — CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--mode", choices=["backtest", "optimize", "dashboard"],
                        default="backtest", help="Execution mode")
    parser.add_argument("--ticker", default="SPY",
                        help="Asset ticker symbol (e.g. SPY, AAPL, BTC-USD)")
    parser.add_argument("--start", default="2018-01-01",
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--short", type=int, default=10,
                        help="Short EMA span")
    parser.add_argument("--long",  type=int, default=50,
                        help="Long EMA span")
    parser.add_argument("--transaction-cost", type=float, default=0.001,
                        help="One-way transaction cost fraction (e.g. 0.001 = 0.1%%)")
    parser.add_argument("--risk-free-rate", type=float, default=None,
                        help="Annualised risk-free rate (default: 0.045 = 4.5%%)")
    parser.add_argument("--save-charts", action="store_true",
                        help="Save charts to outputs/charts/")

    # Significance testing
    parser.add_argument("--significance-test", action="store_true",
                        help="Run permutation significance test on Sharpe Ratio")
    parser.add_argument("--n-permutations", type=int, default=1000,
                        help="Number of permutations for the significance test")

    # Walk-forward
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward validation after grid search (optimize mode)")
    parser.add_argument("--wf-splits", type=int, default=5,
                        help="Number of walk-forward folds")

    # Dashboard
    parser.add_argument("--port",  type=int, default=8050,
                        help="Dashboard server port")
    parser.add_argument("--debug", action="store_true",
                        help="Enable Dash debug mode")

    return parser.parse_args()


def main() -> None:
    _ensure_dirs()
    args = parse_args()

    if args.mode != "dashboard" and args.short >= args.long:
        raise ValueError(
            f"--short ({args.short}) must be strictly less than --long ({args.long})"
        )

    if   args.mode == "backtest":  run_backtest_mode(args)
    elif args.mode == "optimize":  run_optimize_mode(args)
    elif args.mode == "dashboard": run_dashboard_mode(args)


if __name__ == "__main__":
    main()