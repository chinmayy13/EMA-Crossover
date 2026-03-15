"""
visualization.py
----------------
Static Matplotlib/Seaborn charts and interactive Plotly figures.

All chart functions follow the same pattern:
- Accept a backtest DataFrame and optional display/save parameters.
- Return nothing (matplotlib) or a Figure object (plotly).
- Pass save_path=None to display interactively; pass a string path to save.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Matplotlib / Seaborn (static charts)
# ---------------------------------------------------------------------------

def plot_price_with_signals(
    df: pd.DataFrame,
    ticker: str = "",
    short: int = 0,
    long: int = 0,
    save_path: Optional[str] = None,
) -> None:
    """
    Matplotlib chart: Close price + EMA lines + buy/sell markers.

    Parameters
    ----------
    df : pd.DataFrame
        Backtest DataFrame with 'Close', 'EMA_short', 'EMA_long', 'Signal'.
    ticker : str
        Ticker symbol for the chart title.
    short, long : int
        EMA spans for legend labels.
    save_path : str or None
        File path to save the PNG.  None -> display interactively.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df.index, df["Close"],     color="#1f2937", lw=1.2, label="Close",      alpha=0.9)
    ax.plot(df.index, df["EMA_short"], color="#3b82f6", lw=1.5,
            label=f"EMA Short{f' ({short})' if short else ''}", alpha=0.85)
    ax.plot(df.index, df["EMA_long"],  color="#f59e0b", lw=1.5,
            label=f"EMA Long{f' ({long})' if long else ''}",  alpha=0.85)

    buys  = df[df["Signal"] == 1]
    sells = df[df["Signal"] == -1]

    ax.scatter(buys.index,  buys["Close"],  marker="^", color="#10b981", zorder=5, s=80, label="BUY")
    ax.scatter(sells.index, sells["Close"], marker="v", color="#ef4444", zorder=5, s=80, label="SELL")

    ax.set_title(f"{ticker} — Price & EMA Crossover Signals", fontsize=14, fontweight="bold")
    ax.set_ylabel("Price ($)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30)
    plt.tight_layout()

    _save_or_show(fig, save_path)


def plot_equity_curve(
    df: pd.DataFrame,
    ticker: str = "",
    save_path: Optional[str] = None,
) -> None:
    """
    Matplotlib equity curve: Strategy vs Buy-and-Hold with drawdown sub-panel.

    Parameters
    ----------
    df : pd.DataFrame
        Backtest DataFrame with 'Equity_Curve', 'BnH_Equity', 'Drawdown'.
    ticker : str
        Ticker symbol for the chart title.
    save_path : str or None
        File path to save the PNG.  None -> display interactively.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(df.index, df["Equity_Curve"], color="#3b82f6", lw=2,       label="EMA Strategy")
    ax1.plot(df.index, df["BnH_Equity"],   color="#9ca3af", lw=1.5,
             linestyle="--", label="Buy & Hold")
    ax1.set_title(f"{ticker} — Equity Curve Comparison", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    ax2.fill_between(df.index, df["Drawdown"] * 100, 0, color="#ef4444", alpha=0.4)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    plt.xticks(rotation=30)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_rolling_sharpe(
    df: pd.DataFrame,
    ticker: str = "",
    window: int = 126,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the rolling (6-month) Sharpe Ratio over time.

    Parameters
    ----------
    df : pd.DataFrame
        Backtest DataFrame with 'Strategy_Return' column.
    ticker : str
        Ticker symbol for the chart title.
    window : int
        Rolling window in bars (default 126 ~ 6 months).
    save_path : str or None
        File path to save the PNG.  None -> display interactively.
    """
    from src.metrics import rolling_sharpe

    roll_sr = rolling_sharpe(df["Strategy_Return"], window=window)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, roll_sr, color="#8b5cf6", lw=1.5, label=f"Rolling Sharpe ({window}d)")
    ax.axhline(0, color="#9ca3af", lw=0.8, linestyle="--")
    ax.axhline(1, color="#10b981", lw=0.8, linestyle=":", label="Sharpe = 1")
    ax.fill_between(df.index, roll_sr, 0,
                    where=(roll_sr >= 0), color="#10b981", alpha=0.15)
    ax.fill_between(df.index, roll_sr, 0,
                    where=(roll_sr < 0),  color="#ef4444", alpha=0.15)
    ax.set_title(f"{ticker} — Rolling {window}-bar Sharpe Ratio", fontsize=14, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_optimization_heatmap(
    pivot: pd.DataFrame,
    metric: str = "Sharpe",
    save_path: Optional[str] = None,
) -> None:
    """
    Seaborn heatmap of in-sample optimisation results.

    .. note::
        This shows in-sample Sharpe.  Use walk-forward results for a
        realistic estimate of out-of-sample performance.

    Parameters
    ----------
    pivot : pd.DataFrame
        Output of :func:`~optimization.build_heatmap_pivot`.
    metric : str
        Metric name for the title/colorbar.
    save_path : str or None
        File path to save the PNG.  None -> display interactively.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(
        pivot,
        cmap="RdYlGn",
        annot=False,
        fmt=".2f",
        linewidths=0.3,
        ax=ax,
        cbar_kws={"label": metric},
    )
    ax.set_title(
        f"EMA Parameter Optimisation — {metric} Heatmap (In-Sample)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Long EMA")
    ax.set_ylabel("Short EMA")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_walk_forward_results(
    wf_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart comparing in-sample vs out-of-sample Sharpe per fold.

    Parameters
    ----------
    wf_df : pd.DataFrame
        Output of :func:`~optimization.walk_forward_optimize`.
    save_path : str or None
        File path to save the PNG.  None -> display interactively.
    """
    if wf_df.empty:
        print("[Visualization] Walk-forward result is empty — skipping chart.")
        return

    x = wf_df["Fold"].astype(str)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    bars_is  = ax.bar([i - width / 2 for i in range(len(x))], wf_df["InSample_Sharpe"],
                      width=width, color="#3b82f6", alpha=0.85, label="In-Sample Sharpe")
    bars_oos = ax.bar([i + width / 2 for i in range(len(x))], wf_df["OOS_Sharpe"],
                      width=width, color="#f59e0b", alpha=0.85, label="OOS Sharpe")

    ax.axhline(0, color="#9ca3af", lw=0.8)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels([f"Fold {f}" for f in wf_df["Fold"]])
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Walk-Forward Validation — In-Sample vs Out-of-Sample Sharpe",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Plotly (interactive charts — reused by the Dash dashboard)
# ---------------------------------------------------------------------------

def plotly_price_chart(
    df: pd.DataFrame,
    ticker: str = "",
    short: int = 0,
    long: int = 0,
) -> go.Figure:
    """
    Interactive Plotly candlestick chart with EMA lines and signal markers.

    Parameters
    ----------
    df : pd.DataFrame
        Backtest DataFrame with OHLCV, EMA_short/long, Signal columns.
    ticker : str
        Ticker symbol label.
    short, long : int
        EMA spans for legend labels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"],   close=df["Close"],
            name="OHLC",
            increasing_line_color="#10b981",
            decreasing_line_color="#ef4444",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df["EMA_short"],
                   line=dict(color="#3b82f6", width=1.5),
                   name=f"EMA {short}"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["EMA_long"],
                   line=dict(color="#f59e0b", width=1.5),
                   name=f"EMA {long}"),
        row=1, col=1,
    )

    buys  = df[df["Signal"] == 1]
    sells = df[df["Signal"] == -1]

    fig.add_trace(
        go.Scatter(
            x=buys.index,  y=buys["Close"],
            mode="markers",
            marker=dict(symbol="triangle-up",   size=10, color="#10b981"),
            name="BUY",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sells.index, y=sells["Close"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=10, color="#ef4444"),
            name="SELL",
        ),
        row=1, col=1,
    )

    colours = [
        "#10b981" if c >= o else "#ef4444"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"],
               marker_color=colours, name="Volume", opacity=0.6),
        row=2, col=1,
    )

    fig.update_layout(
        title=f"{ticker} — EMA Crossover Strategy",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=30, t=60, b=40),
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume",    row=2, col=1)
    return fig


def plotly_equity_chart(df: pd.DataFrame, ticker: str = "") -> go.Figure:
    """
    Interactive Plotly equity curve with drawdown sub-panel.

    Parameters
    ----------
    df : pd.DataFrame
        Backtest DataFrame with 'Equity_Curve', 'BnH_Equity', 'Drawdown'.
    ticker : str
        Ticker symbol label.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.06,
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df["Equity_Curve"],
                   line=dict(color="#3b82f6", width=2),
                   name="EMA Strategy", fill="tozeroy",
                   fillcolor="rgba(59,130,246,0.1)"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["BnH_Equity"],
                   line=dict(color="#9ca3af", width=1.5, dash="dash"),
                   name="Buy & Hold"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["Drawdown"] * 100,
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.3)",
            line=dict(color="#ef4444", width=1),
            name="Drawdown %",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title=f"{ticker} — Equity Curve & Drawdown",
        template="plotly_dark",
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=30, t=60, b=40),
    )
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)",        row=2, col=1)
    return fig


def plotly_heatmap(pivot: pd.DataFrame, metric: str = "Sharpe") -> go.Figure:
    """
    Interactive Plotly heatmap of in-sample optimisation results.

    Parameters
    ----------
    pivot : pd.DataFrame
        Output of :func:`~optimization.build_heatmap_pivot`.
    metric : str
        Metric name for the colour-axis label.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(r) for r in pivot.index],
            colorscale="RdYlGn",
            colorbar=dict(title=metric),
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title=f"EMA Parameter Optimisation — {metric} Heatmap (In-Sample)",
        xaxis_title="Long EMA",
        yaxis_title="Short EMA",
        template="plotly_dark",
        height=500,
        margin=dict(l=60, r=30, t=60, b=60),
    )
    return fig


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    """Save the figure to *save_path* or display it interactively."""
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
