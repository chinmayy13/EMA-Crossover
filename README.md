# EMA Crossover Backtest — Quantitative Research Pipeline


A small research project exploring a classic technical trading strategy: the EMA crossover.

The project includes a simple backtesting engine, performance analytics, and an interactive dashboard to experiment with parameters and visualize results.

Live demo:  
https://ema-crossover-2.onrender.com

---

## Overview

This project implements a basic systematic trading workflow:

1. Download historical market data  
2. Compute technical indicators (EMAs)  
3. Generate trading signals  
4. Run a vectorized backtest  
5. Evaluate strategy performance  
6. Visualize results through a dashboard  

The goal was to build something closer to a lightweight **quant research tool** rather than just a notebook experiment.

---

## Strategy

The strategy is based on a common moving average crossover rule.

- When the **short EMA** crosses above the **long EMA** → buy signal  
- When the **short EMA** crosses below the **long EMA** → sell signal  

Positions are evaluated over historical daily price data and compared against a buy-and-hold benchmark.

---

## Features

- EMA indicator calculation  
- Trading signal generation  
- Vectorized backtesting  
- Transaction cost modelling  

Performance metrics:

- Sharpe ratio  
- Sortino ratio  
- CAGR  
- Maximum drawdown  
- Volatility  

Other components:

- Interactive dashboard built with Dash + Plotly  
- Cached historical data for reliable deployment  

---

## Project Structure

EMA_CROSSOVER_DASHBOARD

src  
&nbsp;&nbsp;&nbsp;&nbsp;data_loader.py  
&nbsp;&nbsp;&nbsp;&nbsp;indicators.py  
&nbsp;&nbsp;&nbsp;&nbsp;signals.py  
&nbsp;&nbsp;&nbsp;&nbsp;backtester.py  
&nbsp;&nbsp;&nbsp;&nbsp;metrics.py  
&nbsp;&nbsp;&nbsp;&nbsp;visualization.py  
&nbsp;&nbsp;&nbsp;&nbsp;dashboard.py  

data  
&nbsp;&nbsp;&nbsp;&nbsp;SPY.csv  

outputs  
&nbsp;&nbsp;&nbsp;&nbsp;charts  

notebooks  

tests  

main.py  
app.py  
requirements.txt  
README.md  

---

## Dashboard

The dashboard allows you to:

- select a ticker  
- adjust EMA parameters  
- run the backtest  
- view price signals and equity curve  
- inspect performance metrics  

Charts are generated using **Plotly** and the interface is built with **Dash + Bootstrap components**.

---

## Running Locally

Clone the repository

git clone https://github.com/chinmayy13/EMA-Crossover.git  
cd EMA-Crossover  

Install dependencies

pip install -r requirements.txt  

Run the dashboard

python -m src.dashboard  

Open in browser

http://127.0.0.1:8050

---

## Notes

- Historical data is cached in the `data/` folder to avoid repeated API calls.  
- The project is intended for educational and research purposes.  
- It is **not financial advice**.

---

## Possible Improvements

Some extensions that could make the system more realistic:

- walk-forward optimization  
- parameter search heatmaps  
- portfolio level backtesting  
- multiple strategy comparison  
- trade statistics (win rate, profit factor)

---

## Author

Chinmay Kumar  
Final-year Chemical Engineering undergraduate, IIT Madras
