from functools import reduce
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
from coinbase.wallet.client import Client
import yfinance as yf
import talib
import csv
import requests
import pandas as pd
import ta
from collections import deque
from smartmoneyconcepts import smc
import sys
from scipy.signal import argrelextrema


pd.set_option('display.max_rows', None)

API_KEY = None
API_SECRET = None 


exchange = ccxt.coinbasepro({
    'apikey': API_KEY, 
    'apisecret': API_SECRET,
})

# Fetch daily BTC/USDT data for the past 200 days
# Load historical BTC OHLC data from CSV file
df = pd.read_csv('/Users/theryan/Desktop/Trader/BTCUSDFULLRUN.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

def fetch_daily_btc_data(df, start_index, limit):
    daily_data = df.copy()
    daily_data['timestamp'] = pd.to_datetime(daily_data['timestamp'])
    daily_data.set_index('timestamp', inplace=True)
    daily_data = daily_data.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    daily_data.reset_index(inplace=True)
    daily_data = daily_data.iloc[start_index:start_index+limit]
    return daily_data

def fetch_120min_btc_data(df, current_time, limit=6):
    data_120min = df.copy()
    data_120min['timestamp'] = pd.to_datetime(data_120min['timestamp'])
    data_120min.set_index('timestamp', inplace=True)
    data_120min = data_120min.resample('120min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    data_120min.reset_index(inplace=True)
    
    end_time = current_time
    start_time = end_time - timedelta(hours=12)
    
    mask = (data_120min['timestamp'] > start_time) & (data_120min['timestamp'] <= end_time)
    short_table = data_120min.loc[mask]
    
    return short_table

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def process(): 
    balance = 5000
    btc_holdings = 0
    buy_info = []
    trades = []
    portfolio_values = []
    buy_price = []
    trade_active = False
    sells = 0
    buys = 0
    stop_loss = False
    end_date = pd.to_datetime('2023-12-31').date()
    spread = 0.0045  # 0.45% price spread is 0.0045
    take_profit_pct = 0.003  # 0.3% take profit percentage is 0.003
    buy_timestamp = None

    daily = fetch_daily_btc_data(df, 0, len(df)) 
    daily["SHORT_MA"] = daily['close'].rolling (window=1).mean()
    daily["LONG_MA"] = daily['close'].rolling (window=3).mean()
    #BUY WHEN CURRENT SHORT_MA IS GREATER THAN CURRENT LONG_MA, AND PREV SHORT_MA IS LESS THAN PREV LONG_MA
    #SELL WHEN CURRENT SHORT_MA IS LESS THAN CURRENT LONG_MA AND PREV SHORT_MA IS GREATER THAN PREV LONG_MA
    Buy = []
    Sell =[]
    m_buy = [] 
    m_sell = []
    buy_prices = [] 
    sell_prices = []
    portfolio_values = []
    btc_prices = []
    count = 0
    for i in range(len(daily)):
        btc_value = btc_holdings * daily['close'].iloc[i]
        portfolio_value = balance + btc_value
        portfolio_values.append(portfolio_value)
        btc_prices.append(daily['close'].iloc[i])
        if daily.SHORT_MA.iloc[i] > daily.LONG_MA.iloc[i] and daily.SHORT_MA.iloc[i-1] < daily.LONG_MA.iloc[i-1] and trade_active == False:
            for j in range(6): 
                current_time = daily['timestamp'].iloc[i]
                print(current_time)
                short_table = fetch_120min_btc_data(df, current_time)
                short_table["SHORT_MA"] = short_table['close'].rolling (window=1).mean()
                short_table["LONG_MA"] = short_table['close'].rolling (window=3).mean()
                print(short_table)
                if short_table['SHORT_MA'].iloc[-1] > short_table.LONG_MA.iloc[-1] and short_table.SHORT_MA.iloc[-2] < short_table.LONG_MA.iloc[-2] and trade_active == False:
                    m_buy.append(i) 
                    print("BOUGHT HERE")
                    buy_price = short_table['close'].iloc[-1] * (1 + spread)
                    buy_prices.append(buy_price)
                    btc_holdings = balance / buy_price
                    balance = 0
                    trade_active = True
                    buy_timestamp = current_time
                    break 

        elif daily.SHORT_MA.iloc[i] < daily.LONG_MA.iloc[i]  and daily.SHORT_MA.iloc[i-1] > daily.LONG_MA.iloc[i-1] and trade_active == True:
            for k in range(6): 
                current_time = daily['timestamp'].iloc[i]
                print(current_time)
                short_table = fetch_120min_btc_data(df, current_time)
                short_table["SHORT_MA"] = short_table['close'].rolling (window=1).mean()
                short_table["LONG_MA"] = short_table['close'].rolling (window=4).mean()
                print(short_table)
                # Check for take profit condition
                if trade_active and buy_timestamp is not None and (current_time - buy_timestamp).total_seconds() >= (86400*2):  # 86400 seconds = 1 day
                    current_profit = (short_table['close'].iloc[-1] * (1 - spread) - buy_prices[-1]) * btc_holdings
                    if current_profit >= balance * take_profit_pct:
                        sell_price = short_table['close'].iloc[-1] * (1 - spread)
                        sell_prices.append(sell_price)
                        balance = btc_holdings * sell_price
                        btc_holdings = 0
                        trade_active = False
                        buy_timestamp = None
                        m_sell.append(i)
                        break
                if short_table['SHORT_MA'].iloc[-1] < short_table.LONG_MA.iloc[-1] and short_table.SHORT_MA.iloc[-2] > short_table.LONG_MA.iloc[-2] and trade_active == True:
                    trade_active = False
                    m_sell.append(i)
                    sell_price = short_table['close'].iloc[-1] * (1 - spread)
                    sell_prices.append(sell_price)
                    balance = btc_holdings * sell_price
                    btc_holdings = 0
                    break

    win = 0
    losses = 0
    total_gain = 0
    total_loss = 0
    print(len(sell_prices), len(buy_prices))
    for k in range(0,len(buy_prices)): 
        if (buy_prices[k] *1.0045) < (sell_prices[k] * 0.9955): 
            win += 1 
            total_gain += (sell_prices[k] * 0.9955) - (buy_prices[k] *1.0045)
        else: 
            losses += 1  
            total_loss += (sell_prices[k] * 0.9955) - (buy_prices[k] *1.0045)
    
    print(win, losses, total_gain, total_loss, "TOTAL GAIN/LOSS: ", total_gain + total_loss)
    print("Final Balance:", balance)


    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot the portfolio value
    ax1.plot(daily['timestamp'], portfolio_values, color='blue', label='Portfolio Value')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value', color='blue')
    ax1.tick_params('y', colors='blue')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    # Create a second y-axis for BTC price
    ax2 = ax1.twinx()
    ax2.plot(daily['timestamp'], btc_prices, color='orange', label='BTC Price')
    ax2.set_ylabel('BTC Price', color='orange')
    ax2.tick_params('y', colors='orange')

    # Plot red triangles for sells
    sell_dates = [daily['timestamp'].iloc[i] for i in m_sell]
    sell_prices = [btc_prices[i] for i in m_sell]
    ax2.scatter(sell_dates, sell_prices, color='red', marker='^', s=100, label='Sell')

    # Plot green triangles for buys
    buy_dates = [daily['timestamp'].iloc[i] for i in m_buy]
    buy_prices = [btc_prices[i] for i in m_buy]
    ax2.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy')

    # Add legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Set the title
    plt.title('Portfolio Value and BTC Price over Time')

    # Display the plot
    plt.tight_layout()
    plt.show()

process()
