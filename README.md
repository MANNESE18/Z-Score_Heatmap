# Z-Score Mean Reversion Trading Strategy


This project is a quantitative backtesting tool designed to evaluate a **Z-Score Mean Reversion** strategy between a specific equity and a benchmark index. By calculating the relative spread and volatility-adjusted weights, the script performs a grid search across multiple lookback periods (spans) and Z-score thresholds. The final output is an interactive dashboard featuring three heatmaps that visualize the strategy's **Sharpe Ratio**, **Total Return**, and **Max Drawdown**, allowing for rapid identification of optimal parameters.

## Features

* **Dynamic Data Retrieval**: Integrated with the `yfinance` API to fetch real-time and historical adjusted closing prices for any user-defined ticker and index.

* **Parametric Grid Search**: Automates the testing of 165+ different parameter combinations (spanning 10 to 150 days and Z-scores from 0.5 to 3.0) using `numpy.vectorize` for performance.

* **Volatility-Adjusted Weighting**: Implements an inverse-volatility weighting scheme for long/short positions to normalize risk exposure between the stock and the index.

* **Interactive Visualization**: Generates a professional-grade HTML dashboard using `Plotly` subplots, featuring color-coded heatmaps for risk and return metrics.

## Built With

* **Python**: The core programming language used for logic and execution.

* **Pandas & NumPy**: Utilized for high-performance data manipulation, Exponential Weighted Moving (EWM) statistics, and vectorized array operations.

* **Plotly**: Employed to create the interactive multi-column heatmap visualizations.

* **yfinance**: Used as the primary data source for historical market price action.

## Key Achievements in Code

**1. Advanced Spread Logic**

Unlike standard pair-trading models, this code calculates a bidirectional relative spread by taking the `np.maximum` of both the Stock/Index and Index/Stock ratios. This ensures the Z-score captures significant divergence regardless of which asset is outperforming, creating a more robust signal for mean reversion.

**2. Risk-Parity Weighting**

The strategy doesn't just signal a trade; it calculates the optimal weight for the investment using the inverse of the assets' historical volatility (1/σ). This ensures that higher-volatility regimes receive lower capital allocation, effectively "levelling" the risk across the study's timeframe.

**3. Vectorized Performance**

By leveraging `np.vectorize` to map complex strategy functions across a multi-dimensional grid of lookback spans and Z-score thresholds, the script avoids inefficient Python loops. This allows for the simultaneous calculation of hundreds of backtests in seconds.
