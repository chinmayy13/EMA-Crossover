"""
dashboard.py
------------
Interactive Dash dashboard for the EMA Crossover Backtest system.

Run:
    python main.py --mode dashboard
or directly:
    python -m src.dashboard
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

warnings.filterwarnings("ignore")

from src.backtester import run_backtest
from src.data_loader import download_data, validate_data
from src.indicators import add_emas
from src.metrics import DEFAULT_RISK_FREE_RATE, compute_all_metrics
from src.signals import generate_signals
from src.visualization import plotly_equity_chart, plotly_price_chart


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

TICKERS = ["SPY", "AAPL", "MSFT", "TSLA", "BTC-USD", "ETH-USD", "GLD", "QQQ"]

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="EMA Crossover Dashboard",
)
server = app.server  # expose Flask server for deployment


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def metric_card(label: str, value_id: str, color: str = "#3b82f6") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.P(label, className="text-muted mb-1", style={"fontSize": "0.8rem"}),
            html.H4(id=value_id, children="—",
                    style={"color": color, "fontWeight": "700"}),
        ]),
        className="text-center",
        style={"background": "#1e293b", "border": f"1px solid {color}33"},
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

app.layout = dbc.Container(
    fluid=True,
    style={"background": "#0f172a", "minHeight": "100vh", "padding": "24px"},
    children=[
        # Header
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.H2(
                        "EMA Crossover Backtest Dashboard",
                        style={"color": "#f8fafc", "fontWeight": "800", "marginBottom": "4px"},
                    ),
                    html.P(
                        "Quantitative EMA Strategy  |  Parameter Optimisation  |  Performance Analytics",
                        style={"color": "#64748b", "fontSize": "0.9rem"},
                    ),
                ]),
                width=12,
            ),
            className="mb-3",
        ),

        # Control panel
        dbc.Card(
            dbc.CardBody(
                dbc.Row([
                    dbc.Col([
                        html.Label("Ticker", style={"color": "#94a3b8", "fontSize": "0.8rem"}),
                        dcc.Dropdown(
                            id="dd-ticker",
                            options=[{"label": t, "value": t} for t in TICKERS],
                            value="SPY",
                            clearable=False,
                            style={"background": "#1e293b"},
                        ),
                    ], md=2),

                    dbc.Col([
                        html.Label("Start Date", style={"color": "#94a3b8", "fontSize": "0.8rem"}),
                        dcc.Input(
                            id="inp-start", type="text", value="2018-01-01", debounce=True,
                            style={"width": "100%", "background": "#1e293b", "color": "#f1f5f9",
                                   "border": "1px solid #334155", "borderRadius": "4px", "padding": "6px"},
                        ),
                    ], md=2),

                    dbc.Col([
                        html.Label("Short EMA", style={"color": "#94a3b8", "fontSize": "0.8rem"}),
                        dcc.Input(
                            id="inp-short", type="number", value=10, min=2, max=100, debounce=True,
                            style={"width": "100%", "background": "#1e293b", "color": "#f1f5f9",
                                   "border": "1px solid #334155", "borderRadius": "4px", "padding": "6px"},
                        ),
                    ], md=2),

                    dbc.Col([
                        html.Label("Long EMA", style={"color": "#94a3b8", "fontSize": "0.8rem"}),
                        dcc.Input(
                            id="inp-long", type="number", value=50, min=5, max=300, debounce=True,
                            style={"width": "100%", "background": "#1e293b", "color": "#f1f5f9",
                                   "border": "1px solid #334155", "borderRadius": "4px", "padding": "6px"},
                        ),
                    ], md=2),

                    dbc.Col(
                        dbc.Button(
                            "Run Backtest", id="btn-run", color="primary",
                            n_clicks=0,
                            style={"marginTop": "22px", "width": "100%"},
                        ),
                        md=2,
                    ),
                    dbc.Col(
                        html.Div(
                            id="status-msg",
                            style={"color": "#64748b", "fontSize": "0.8rem", "marginTop": "26px"},
                        ),
                        md=2,
                    ),
                ])
            ),
            style={"background": "#1e293b", "border": "1px solid #334155", "marginBottom": "18px"},
        ),

        # Metric cards (8 metrics in two rows of 4)
        dbc.Row([
            dbc.Col(metric_card("Sharpe Ratio",   "m-sharpe",  "#3b82f6"), md=3),
            dbc.Col(metric_card("Sortino Ratio",  "m-sortino", "#8b5cf6"), md=3),
            dbc.Col(metric_card("Calmar Ratio",   "m-calmar",  "#06b6d4"), md=3),
            dbc.Col(metric_card("Max Drawdown",   "m-mdd",     "#ef4444"), md=3),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(metric_card("Total Return",   "m-tr",      "#10b981"), md=3),
            dbc.Col(metric_card("CAGR",           "m-cagr",    "#f59e0b"), md=3),
            dbc.Col(metric_card("Volatility",     "m-vol",     "#a78bfa"), md=3),
            dbc.Col(metric_card("BnH Return",     "m-bnh",     "#94a3b8"), md=3),
        ], className="mb-3"),

        # Charts
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(dcc.Graph(id="chart-price", config={"displayModeBar": True})),
                    style={"background": "#1e293b", "border": "1px solid #334155"},
                ),
                md=12, className="mb-3",
            ),
        ]),
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(dcc.Graph(id="chart-equity", config={"displayModeBar": True})),
                    style={"background": "#1e293b", "border": "1px solid #334155"},
                ),
                md=12, className="mb-3",
            ),
        ]),

        # Footer
        html.Hr(style={"borderColor": "#334155"}),
        html.P(
            "EMA Crossover Backtest Engine  |  Built with Dash + Plotly  |  "
            "For educational purposes only. Not financial advice.",
            style={"color": "#475569", "textAlign": "center", "fontSize": "0.75rem"},
        ),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("chart-price",  "figure"),
    Output("chart-equity", "figure"),
    Output("m-sharpe",  "children"),
    Output("m-sortino", "children"),
    Output("m-calmar",  "children"),
    Output("m-mdd",     "children"),
    Output("m-tr",      "children"),
    Output("m-cagr",    "children"),
    Output("m-vol",     "children"),
    Output("m-bnh",     "children"),
    Output("status-msg", "children"),
    Input("btn-run", "n_clicks"),
    State("dd-ticker",  "value"),
    State("inp-start",  "value"),
    State("inp-short",  "value"),
    State("inp-long",   "value"),
    prevent_initial_call=False,
)
def run_dashboard(n_clicks, ticker, start_date, short_ema, long_ema):
    """Main callback: download -> compute -> update all outputs."""
    empty_fig = go.Figure()
    empty_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
    )
    blank = ("—",) * 8

    if not ticker:
        return (empty_fig, empty_fig) + blank + ("Select a ticker",)

    short_ema = int(short_ema or 10)
    long_ema  = int(long_ema  or 50)

    if short_ema >= long_ema:
        return (empty_fig, empty_fig) + blank + ("Short EMA must be < Long EMA",)

    try:
        raw = download_data(ticker, start=start_date or "2018-01-01")
        df  = validate_data(raw)          # returns a clean copy
        df  = add_emas(df, short_ema, long_ema)
        df  = generate_signals(df)
        df  = run_backtest(df)
        metrics = compute_all_metrics(df, risk_free_rate=DEFAULT_RISK_FREE_RATE)

        price_fig  = plotly_price_chart(df, ticker, short_ema, long_ema)
        equity_fig = plotly_equity_chart(df, ticker)

        def fmt_ratio(val: object) -> str:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "N/A"
            return f"{val:.2f}"

        def fmt_pct(val: object) -> str:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "N/A"
            return f"{val * 100:.2f}%"

        n_trades = int((df["Signal"] != 0).sum())
        status   = f"Done: {n_trades} signals | {len(df)} bars"

        return (
            price_fig,
            equity_fig,
            fmt_ratio(metrics["Sharpe"]),
            fmt_ratio(metrics["Sortino"]),
            fmt_ratio(metrics["Calmar"]),
            fmt_pct(metrics["Max_Drawdown"]),
            fmt_pct(metrics["Total_Return"]),
            fmt_pct(metrics["CAGR"]),
            fmt_pct(metrics["Volatility"]),
            fmt_pct(metrics["BnH_Total_Return"]),
            status,
        )

    except Exception as exc:
        return (empty_fig, empty_fig) + blank + (f"Error: {exc}",)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_server(debug: bool = False, port: int = int(os.environ.get("PORT", 8050))):
    print(f"[Dashboard] Starting on port {port}")
    app.run(debug=debug, port=port, host="0.0.0.0")


if __name__ == "__main__":
    run_server(debug=True)
