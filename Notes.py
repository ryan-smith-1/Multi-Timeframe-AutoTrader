import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
from coinbase.wallet.client import Client

pd.set_option('display.max_rows', None)

API_KEY = None
API_SECRET = None 

exchange = ccxt.coinbasepro({
    'apikey': API_KEY, 
    'apisecret': API_SECRET,
})

# Fetch daily BTC/USDT data for the past 200 days
end_date = datetime.now()
start_date = end_date - timedelta(days=200)
bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='1d', since=int(start_date.timestamp()) * 1000)
df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

def tr(data):
    data['previous_close'] = data['close'].shift(1)
    data['high-low'] = data['high'] - data['low']
    data['high-pc'] = abs(data['high'] - data['previous_close'])
    data['low-pc'] = abs(data['low'] - data['previous_close'])
    tr = data[['high-low', 'high-pc', 'low-pc']].max(axis=1)
    return tr

def atr(data, period):
    data['tr'] = tr(data)
    atr = data['tr'].rolling(period).mean()
    data['atr'] = atr
    return atr

######SUPERTREND_VARIABLES######
period = 12
multiplier = 3
######SUPERTREND_VARIABLES######

def supertrend(data, period, multiplier):
    bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='1m', limit = 15)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data = df.copy()
    data['atr'] = atr(data, period)  # Calculate ATR using the live table data
    data['upper_band'] = ((data['high'] + data['low'])/2) + (multiplier * data['atr'].iloc[-1])
    data['lower_band'] = ((data['high'] + data['low'])/2) - (multiplier * data['atr'].iloc[-1])
    data['in_uptrend'] = True

    for current in range(1, len(data)):
        prev = current - 1
        if data['close'][current] > data['upper_band'][prev]:
            data.loc[current, 'in_uptrend'] = True
        elif data['close'][current] < data['lower_band'][prev]:
            data.loc[current, 'in_uptrend'] = False
        else:
            data.loc[current, 'in_uptrend'] = data['in_uptrend'][prev]

            if data['in_uptrend'][current] and data['lower_band'][current] < data['lower_band'][prev]:
                data.loc[current, 'lower_band'] = data['lower_band'][prev]

            if not data['in_uptrend'][current] and data['upper_band'][current] > data['upper_band'][prev]:
                data.loc[current, 'upper_band'] = data['upper_band'][prev]

    return data

def dema(data, period, column='close'):
    ema = data[column].ewm(span=period, adjust=False).mean()
    dema = 2 * ema - ema.ewm(span=period, adjust=False).mean()
    return dema

def backtest(df, initial_balance):
    wait_time = 3
    balance = initial_balance
    btc_holdings = 0
    buy_info = []
    trades = []
    portfolio_values = []
    buy_price = []
    trade_active = False  # Track if a trade is active
    coinbase_API_key = "8sjh4qcRNVFKGEhX"
    coinbase_API_secret = "iVacysje8THlgeYMkGyr8LJ8zG7jWEOz"
    client = Client(coinbase_API_key, coinbase_API_secret)

    currency_code = "BTC-USD"
    live_table = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close'])

    df['DEMA'] = dema(df, period=200)
    fetch_count = 0
    dema_fetch_count = 0
    while True:
        # Fetch new BTC price data every second
        current_price = client.get_spot_price(currency=currency_code)
        new_price = float(current_price.amount)
        current_timestamp = pd.Timestamp.now()
        # Create a temporary dataframe with the current price data
        new_data = pd.DataFrame({'timestamp': [current_timestamp], 'open': [new_price], 'high': [new_price], 'low': [new_price], 'close': [new_price]})

        # Concatenate the new data with the live table
        if (fetch_count * wait_time) >= 60:
            live_table = pd.concat([live_table, new_data], ignore_index=True)
            fetch_count = 0

        # Update the DEMA every 24 hours
        if (dema_fetch_count * wait_time) >= 86400:  # 24 hours in seconds
            df = pd.concat([df, live_table], ignore_index=True)
            df['DEMA'] = dema(df, period=200)
            live_table = live_table.iloc[-15:]  # Keep only the last 15 minutes of data
            dema_fetch_count = 0

        # Keep only the last 15 minutes of data in the live table
        live_table = live_table[live_table['timestamp'] >= current_timestamp - pd.Timedelta(minutes=15)]

        # Calculate supertrend indicator for the current moment
        current_supertrend = supertrend(live_table, period, multiplier)
        current_in_uptrend = current_supertrend['in_uptrend'].iloc[-1]

        if current_in_uptrend and new_price > df['DEMA'].iloc[-1] and balance > 0:
            # Buy entire usd balance worth of btc at market
            # Set stop loss condition for current price
            buy_amount = balance
            btc_bought = buy_amount / new_price
            btc_holdings += btc_bought
            balance -= buy_amount
            buy_info.append(btc_bought)
            buy_price.append(new_price)
            trade_active = True
            print("BOUGHT")
        elif not current_in_uptrend and trade_active:
            # Sell entire btc balance at market
            last_buy = buy_info.pop()
            balance += last_buy * new_price
            btc_holdings -= last_buy
            buy_p = buy_price.pop()
            print("Profit/Loss from selling", last_buy, "BTC: ", (last_buy * new_price) - (last_buy * buy_p))
            trade_active = False

        # Calculate the current portfolio value
        portfolio_value = balance + (btc_holdings * new_price)
        portfolio_values.append(portfolio_value)
        print(current_in_uptrend, "New Price: ", new_price, "DEMA: ", df['DEMA'].iloc[-1], "USD BALANCE: ", balance, "BTC BALANCE: ", btc_holdings * new_price)

        # Wait for the next second
        time.sleep(wait_time)
        fetch_count += 1
        dema_fetch_count += 1

    return trades, portfolio_values

# Perform backtesting
initial_balance = 5000  # Starting balance in USD
trades, portfolio_values = backtest(df, initial_balance)

# Print the trades
print("Trades:")
for trade in trades:
    print(trade)

# Print the final portfolio value
final_portfolio_value = portfolio_values[-1]
print(f"\nFinal Portfolio Value: ${final_portfolio_value:.2f}")

# Calculate the total profit/loss
total_profit_loss = final_portfolio_value - initial_balance
print(f"Total Profit/Loss: ${total_profit_loss:.2f}")

# Calculate the percentage return
percentage_return = (total_profit_loss / initial_balance) * 100
print(f"Percentage Return: {percentage_return:.2f}%")

plt.figure(figsize=(12, 6))
plt.plot(df['close'], label='BTC Price')
plt.plot(portfolio_values, label='Portfolio Value (USD)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('BTC Price, Portfolio Value, and DEMA Over Time')
plt.legend()
plt.grid(True)
plt.show()