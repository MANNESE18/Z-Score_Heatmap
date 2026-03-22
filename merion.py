import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf



stock_ticker = input("Enter Stock Ticker or Type 'quit' to exit:").upper()
index_ticker = input("Enter Index Ticker or Type 'quit' to exit:").upper()
start_year = int(input("Enter start year for study:"))
end_year = int(input("Enter end year for study:"))

data_start = start_year-1

# downloads info from yfinance API based on inputs from user
data = yf.download([stock_ticker, index_ticker], start=f"{data_start}-01-01", auto_adjust=True)

# sets up data frames
stock_index = data['Close'].copy()

stock_index = stock_index.rename(columns={
    stock_ticker: 'ac_stock', 
    index_ticker: 'ac_index'
})


# table1: sharpe ratio calculations

def strategy_sharpe(span , z_score):

    # calculations for initial variables needed
    stock_index['SI_Comp'] = stock_index['ac_stock']/stock_index['ac_index']
    stock_index['IS_Comp'] = stock_index['ac_index']/stock_index['ac_stock']
    stock_index['spread'] = np.maximum(stock_index['SI_Comp'], stock_index['IS_Comp'])
    stock_index['spread_mean'] = stock_index['spread'].ewm(span=span).mean()
    stock_index['spread_std'] = stock_index['spread'].ewm(span=span).std()
    stock_index['z_score'] = (stock_index['spread'] - stock_index['spread_mean'])/(stock_index['spread_std'])

    # helps to define further conditions in code and how the "trades" will be signaled
    stock_index['higher'] = np.maximum(stock_index['ac_stock'], stock_index['ac_index'])
    stock_index['lower'] = np.minimum(stock_index['ac_stock'], stock_index['ac_index'])
    stock_index['signal'] = np.where((stock_index['z_score'] >= z_score) | (stock_index['z_score'] <= -z_score), 1, 0)

    # calculations for the weights of stock investment
    stock_index['daily_return_for_higher'] = stock_index['higher'].pct_change()
    stock_index['daily_return_for_lower'] = stock_index['lower'].pct_change()
    stock_index['std_for_higher'] = stock_index['daily_return_for_higher'].ewm(span=span).std()
    stock_index['std_for_lower'] = stock_index['daily_return_for_lower'].ewm(span=span).std()
    stock_index['inv_std_sum'] = (1/stock_index['std_for_higher']) + (1/stock_index['std_for_lower'])
    stock_index['weight_for_higher'] = (1/stock_index['std_for_higher']) / (stock_index['inv_std_sum'])
    stock_index['weight_for_lower'] = (1/stock_index['std_for_lower']) / (stock_index['inv_std_sum'])

    # if stock > index, then the action would always be short the stock
    # if stock < index, then the action would always be long the stock
    # only investment will be w/ the stock (instructions define whether to short or long the stock)
    strategy_higher = stock_index['higher'].pct_change()*stock_index['signal'].shift(1)*stock_index['weight_for_higher'].shift(1)*(-1)
    strategy_lower = stock_index['lower'].pct_change()*(stock_index['signal'].shift(1))*(stock_index['weight_for_lower'].shift(1))
    stock_index['strategy'] = np.where(stock_index['higher'] == stock_index['ac_stock'], strategy_higher, strategy_lower)

    # calculations for Sharpe ratio
    strategy_daily = stock_index['strategy'].fillna(0)
    strategy_daily_mean = strategy_daily.mean()
    strategy_daily_std = strategy_daily.std()
    if strategy_daily_std != 0:
        sharpe = (strategy_daily_mean/strategy_daily_std)*np.sqrt(252)
    else:
        sharpe = 0

    return sharpe





# table2: total return calculations

def strategy_return(span,z_score):

    # calculations for initial variables needed
    stock_index['SI_Comp'] = stock_index['ac_stock']/stock_index['ac_index']
    stock_index['IS_Comp'] = stock_index['ac_index']/stock_index['ac_stock']
    stock_index['spread'] = np.maximum(stock_index['SI_Comp'], stock_index['IS_Comp'])
    stock_index['spread_mean'] = stock_index['spread'].ewm(span=span).mean()
    stock_index['spread_std'] = stock_index['spread'].ewm(span=span).std()
    stock_index['z_score'] = (stock_index['spread'] - stock_index['spread_mean'])/(stock_index['spread_std'])

    # helps to define further conditions in code and how the "trades" will be signaled
    stock_index['higher'] = np.maximum(stock_index['ac_stock'], stock_index['ac_index'])
    stock_index['lower'] = np.minimum(stock_index['ac_stock'], stock_index['ac_index'])
    stock_index['signal'] = np.where((stock_index['z_score'] >= z_score) | (stock_index['z_score'] <= -z_score), 1, 0)

    # calculations for the weights of stock investment
    stock_index['daily_return_for_higher'] = stock_index['higher'].pct_change()
    stock_index['daily_return_for_lower'] = stock_index['lower'].pct_change()
    stock_index['std_for_higher'] = stock_index['daily_return_for_higher'].ewm(span=span).std()
    stock_index['std_for_lower'] = stock_index['daily_return_for_lower'].ewm(span=span).std()
    stock_index['inv_std_sum'] = (1/stock_index['std_for_higher']) + (1/stock_index['std_for_lower'])
    stock_index['weight_for_higher'] = (1/stock_index['std_for_higher']) / (stock_index['inv_std_sum'])
    stock_index['weight_for_lower'] = (1/stock_index['std_for_lower']) / (stock_index['inv_std_sum'])

    # if stock > index, then the action would always be short the stock
    # if stock < index, then the action would always be long the stock
    # only investment will be w/ the stock (instructions define whether to short or long the stock)
    strategy_higher = stock_index['higher'].pct_change()*stock_index['signal'].shift(1)*stock_index['weight_for_higher'].shift(1)*(-1)
    strategy_lower = stock_index['lower'].pct_change()*(stock_index['signal'].shift(1))*(stock_index['weight_for_lower'].shift(1))
    stock_index['strategy'] = np.where(stock_index['higher'] == stock_index['ac_stock'], strategy_higher, strategy_lower)

    # calculates total return of investment to display
    stock_index['return'] = ((1 + stock_index['strategy'].fillna(0)).cumprod()-1)
    strategy_return = stock_index['return'].iloc[-1]

    return strategy_return


# table3: max drawdown calculations

def strategy_max_drawdown(span,z_score):

    stock_index['SI_Comp'] = stock_index['ac_stock']/stock_index['ac_index']
    stock_index['IS_Comp'] = stock_index['ac_index']/stock_index['ac_stock']
    stock_index['spread'] = np.maximum(stock_index['SI_Comp'], stock_index['IS_Comp'])
    stock_index['spread_mean'] = stock_index['spread'].ewm(span=span).mean()
    stock_index['spread_std'] = stock_index['spread'].ewm(span=span).std()
    stock_index['z_score'] = (stock_index['spread'] - stock_index['spread_mean'])/(stock_index['spread_std'])

    # helps to define further conditions in code and how the "trades" will be signaled
    stock_index['higher'] = np.maximum(stock_index['ac_stock'], stock_index['ac_index'])
    stock_index['lower'] = np.minimum(stock_index['ac_stock'], stock_index['ac_index'])
    stock_index['signal'] = np.where((stock_index['z_score'] >= z_score) | (stock_index['z_score'] <= -z_score), 1, 0)

    # calculations for the weights of stock investment
    stock_index['daily_return_for_higher'] = stock_index['higher'].pct_change()
    stock_index['daily_return_for_lower'] = stock_index['lower'].pct_change()
    stock_index['std_for_higher'] = stock_index['daily_return_for_higher'].ewm(span=span).std()
    stock_index['std_for_lower'] = stock_index['daily_return_for_lower'].ewm(span=span).std()
    stock_index['inv_std_sum'] = (1/stock_index['std_for_higher']) + (1/stock_index['std_for_lower'])
    stock_index['weight_for_higher'] = (1/stock_index['std_for_higher']) / (stock_index['inv_std_sum'])
    stock_index['weight_for_lower'] = (1/stock_index['std_for_lower']) / (stock_index['inv_std_sum'])

    # if stock > index, then the action would always be short the stock
    # if stock < index, then the action would always be long the stock
    # only investment will be w/ the stock (instructions define whether to short or long the stock)
    strategy_higher = stock_index['higher'].pct_change()*stock_index['signal'].shift(1)*stock_index['weight_for_higher'].shift(1)*(-1)
    strategy_lower = stock_index['lower'].pct_change()*(stock_index['signal'].shift(1))*(stock_index['weight_for_lower'].shift(1))
    stock_index['strategy'] = np.where(stock_index['higher'] == stock_index['ac_stock'], strategy_higher, strategy_lower)

    # find max drawdown
    equity_curve = (1 + stock_index['strategy']).cumprod()
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()

    return max_drawdown


# define variable arrays used by all fucntions
short_end = np.arange(10,55,5)
long_end = np.arange(70,151,20)
span = np.concatenate([short_end,long_end])
z_score = np.arange(.5, 3.25, .25)

# set up how functions should try different array combos
strategy_func = np.vectorize(strategy_sharpe)
return_func = np.vectorize(strategy_return)
drawdown_func = np.vectorize(strategy_max_drawdown)

# sets up a grid for each function
grid_sharpe = strategy_func(span[:,None], z_score[None, :])
grid_return =  return_func(span[:,None], z_score[None, :])
grid_max_drawdown =  drawdown_func(span[:,None], z_score[None, :])

fig = make_subplots(rows=1, cols=3, subplot_titles=("Sharpe Ratio", "Total Return", "Max Drawdown"))

fig.update_layout(
    title = f"<b>Z-Score Trading Strategy: {stock_ticker} vs. {index_ticker} | {start_year}-{end_year}<b>",
    title_x=.5,
    title_y=.95,
    font = dict(size=16),
    margin=dict(t=130)
)

# creates actual heatmap tables
fig.add_trace(go.Heatmap(z = grid_sharpe, y=span, x = z_score, colorscale = 'rDYlGn', texttemplate='%{z:.2f}', showscale=False), row=1, col=1)
fig.add_trace(go.Heatmap(z = grid_return, y=span, x = z_score, colorscale = 'rDYlGn', texttemplate='%{z:.2%}', showscale=False), row=1, col=2)
fig.add_trace(go.Heatmap(z = grid_max_drawdown, y=span, x = z_score, colorscale = 'Reds_r', texttemplate='%{z:.2%}', showscale=False), row=1, col=3)

fig.write_html("Z-Score_Trading_Strategy.html")
fig.show()
    





    
