# import sys
# import os
# sys.path.append(os.path.abspath('.'))

# import streamlit as st
# import pandas as pd

# from src.data_loader import download_data
# from src.indicators import add_emas
# from src.signals import generate_signals
# from src.backtester import run_backtest
# from src.metrics import compute_metrics
# from src.visualization import plot_equity_curve

# st.title("Quant Trading Strategy Backtester")

# st.write("EMA Crossover Backtesting Dashboard")

# # Inputs
# ticker = st.text_input("Ticker", "AAPL")
# short = st.slider("Short EMA", 5, 50, 12)
# long = st.slider("Long EMA", 20, 200, 26)

# if st.button("Run Backtest"):

#     df = download_data(ticker)

#     df = add_emas(df, short, long)

#     df = generate_signals(df)

#     results = run_backtest(df)

#     metrics = compute_metrics(results)

#     st.subheader("Performance Metrics")

#     st.write(metrics)

#     fig = plot_equity_curve(results)

#     st.plotly_chart(fig)
    
    
    
import streamlit as st
import subprocess
import sys

st.title("📈 Quant Trading Strategy Backtester")

st.write("EMA Crossover Strategy Research Dashboard")

# Inputs
ticker = st.text_input("Ticker", "AAPL")
short = st.slider("Short EMA", 5, 50, 12)
long = st.slider("Long EMA", 20, 200, 26)

if st.button("Run Backtest"):

    command = [
        sys.executable,
        "main.py",
        "--ticker", ticker,
        "--short", str(short),
        "--long", str(long),
        "--save-charts"
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    st.subheader("Backtest Output")

    st.text(result.stdout)

    if result.stderr:
        st.error(result.stderr)

    st.success("Backtest completed. Charts saved in outputs folder.")