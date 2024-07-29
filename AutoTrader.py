import pandas as pd
#from datetime import datetime, timedelta
import time
import talib
import requests
import sys
import base64
import json
from typing import Any, Dict, Optional
import uuid
from cryptography.hazmat.primitives.asymmetric import ed25519
import math
import datetime

#THE MODEL MUST BE RUN ABOUT 15-30 mins to 1-2 hours AFTER MIDNIGHT GMT (8:00PM EST) ~ => 8:15PM-10:00PM EST



API_KEY = None
BASE64_PRIVATE_KEY = None


class CryptoAPITrading:
    def __init__(self):
        self.api_key = API_KEY
        private_bytes = base64.b64decode(BASE64_PRIVATE_KEY)
        # Note that the cryptography library used here only accepts a 32 byte ed25519 private key
        self.private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_bytes[:32])
        self.base_url = "https://trading.robinhood.com"

    @staticmethod
    def _get_current_timestamp() -> int:
        return int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp())

    @staticmethod
    def get_query_params(key: str, *args: Optional[str]) -> str:
        if not args:
            return ""

        params = []
        for arg in args:
            params.append(f"{key}={arg}")

        return "?" + "&".join(params)

    def make_api_request(self, method: str, path: str, body: str = "") -> Any:
        timestamp = self._get_current_timestamp()
        headers = self.get_authorization_header(method, path, body, timestamp)
        url = self.base_url + path

        try:
            response = {}
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=10)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=json.loads(body), timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.json()
        except requests.RequestException as e:
            print(f"Error making API request: {e}")
            return None

    def get_authorization_header(
            self, method: str, path: str, body: str, timestamp: int
    ) -> Dict[str, str]:
        message_to_sign = f"{self.api_key}{timestamp}{path}{method}{body}"
        signature = self.private_key.sign(message_to_sign.encode("utf-8"))

        return {
            "x-api-key": self.api_key,
            "x-signature": base64.b64encode(signature).decode("utf-8"),
            "x-timestamp": str(timestamp),
        }

    def get_account(self) -> Any:
        path = "/api/v1/crypto/trading/accounts/"
        return self.make_api_request("GET", path)

    # The symbols argument must be formatted in trading pairs, e.g "BTC-USD", "ETH-USD". If no symbols are provided,
    # all supported symbols will be returned
    def get_trading_pairs(self, *symbols: Optional[str]) -> Any:
        query_params = self.get_query_params("symbol", *symbols)
        path = f"/api/v1/crypto/trading/trading_pairs/{query_params}"
        return self.make_api_request("GET", path)

    # The asset_codes argument must be formatted as the short form name for a crypto, e.g "BTC", "ETH". If no asset
    # codes are provided, all crypto holdings will be returned
    def get_holdings(self, *asset_codes: Optional[str]) -> Any:
        query_params = self.get_query_params("asset_code", *asset_codes)
        path = f"/api/v1/crypto/trading/holdings/{query_params}"
        return self.make_api_request("GET", path)

    # The symbols argument must be formatted in trading pairs, e.g "BTC-USD", "ETH-USD". If no symbols are provided,
    # the best bid and ask for all supported symbols will be returned
    def get_best_bid_ask(self, *symbols: Optional[str]) -> Any:
        query_params = self.get_query_params("symbol", *symbols)
        path = f"/api/v1/crypto/marketdata/best_bid_ask/{query_params}"
        return self.make_api_request("GET", path)

    # The symbol argument must be formatted in a trading pair, e.g "BTC-USD", "ETH-USD"
    # The side argument must be "bid", "ask", or "both".
    # Multiple quantities can be specified in the quantity argument, e.g. "0.1,1,1.999".
    def get_estimated_price(self, symbol: str, side: str, quantity: str) -> Any:
        path = f"/api/v1/crypto/marketdata/estimated_price/?symbol={symbol}&side={side}&quantity={quantity}"
        return self.make_api_request("GET", path)

    def place_order(
            self,
            client_order_id: str,
            side: str,
            order_type: str,
            symbol: str,
            order_config: Dict[str, str],
    ) -> Any:
        body = {
            "client_order_id": client_order_id,
            "side": side,
            "type": order_type,
            "symbol": symbol,
            f"{order_type}_order_config": order_config,
        }
        path = "/api/v1/crypto/trading/orders/"
        return self.make_api_request("POST", path, json.dumps(body))

    def cancel_order(self, order_id: str) -> Any:
        path = f"/api/v1/crypto/trading/orders/{order_id}/cancel/"
        return self.make_api_request("POST", path)

    def get_order(self, order_id: str) -> Any:
        path = f"/api/v1/crypto/trading/orders/{order_id}/"
        return self.make_api_request("GET", path)

    def get_orders(self) -> Any:
        path = "/api/v1/crypto/trading/orders/"
        return self.make_api_request("GET", path)
api_trading_client = CryptoAPITrading()

def get_daily_data():
    # Initialize variables
    base_url = "https://min-api.cryptocompare.com/data/v2/histoday"
    fsym = "BTC"
    tsym = "USD"
    limit = 12
    closing_prices = []
    # Make the API request
    params = {
        'fsym': fsym,
        'tsym': tsym,
        'limit': limit
    }
    response = requests.get(base_url, params=params)
    data = response.json()['Data']['Data']

    # Extract closing prices from the response
    closing_prices += [entry['close'] for entry in data]
    dema_data = closing_prices
    #HIGHEST INDEXES ARE MOST RECENT DATA.   
    if len(dema_data)>12: 
        dema_data.pop() #remove the incomplete day

    take_profit_data = dema_data[-2:] 
    short_ma_data = dema_data[-1]  # This is equivalent to window=1
    long_ma_data = sum(dema_data[-3:]) / 3  # This is equivalent to window=3
    return dema_data, take_profit_data, short_ma_data, long_ma_data

def get_120min_data():
    # Initialize variables
    base_url = "https://min-api.cryptocompare.com/data/v2/histohour"
    fsym = "BTC"
    tsym = "USD"
    limit = 12  # Fetch 24 hours of data to create 12 120-minute candles
    hourly_data = []

    # Make the API request
    params = {
        'fsym': fsym,
        'tsym': tsym,
        'limit': limit
    }
    response = requests.get(base_url, params=params)
    data = response.json()['Data']['Data']

    # Extract hourly data from the response
    hourly_data = [{'open': entry['open'], 'high': entry['high'], 
                    'low': entry['low'], 'close': entry['close'], 
                    'volume': entry['volumefrom']} for entry in data]

    # Aggregate hourly data into 120-minute data
    data_120min = []
    for i in range(0, len(hourly_data), 2):
        if i+1 < len(hourly_data):
            candle_120min = {
                'open': hourly_data[i]['open'],
                'high': max(hourly_data[i]['high'], hourly_data[i+1]['high']),
                'low': min(hourly_data[i]['low'], hourly_data[i+1]['low']),
                'close': hourly_data[i+1]['close'],
                'volume': hourly_data[i]['volume'] + hourly_data[i+1]['volume']
            }
            data_120min.append(candle_120min)

    # Extract required data
    closing_prices = [candle['close'] for candle in data_120min]
    dema_data = closing_prices
    
    # HIGHEST INDEXES ARE MOST RECENT DATA
    if len(dema_data) > 6:
        dema_data.pop()  # remove the incomplete period if any

    take_profit_data = dema_data[-2:]
    short_ma_data = dema_data[-1]  # This is equivalent to window=1
    long_ma_data = sum(dema_data[-2:]) / 2  # This is equivalent to window=2

    return dema_data, take_profit_data, short_ma_data, long_ma_data


def sell_all_positions(): 
    sys.exit("Program has been exited but a trade is still active... It must be canceled in Robinhood.")

def calculate_dema(closing_prices, spread, trade_active):
    if trade_active:
        closing_prices = [price * (1 - spread) for price in closing_prices]
    else:
        closing_prices = [price * (1 + spread) for price in closing_prices]

    new_df = pd.DataFrame(closing_prices, columns=['close'])
    dema = talib.DEMA(new_df['close'], timeperiod=6)
    return dema.iloc[-1]

def wait_for_order_execution(api_client, order_id, max_wait_time=120):
    start_time = time.time()
    while time.time() - start_time < float(120):
        order_status = api_client.get_order(order_id)
        if order_status and 'state' in order_status:
            if order_status['state'] == 'filled':
                return True
            elif order_status['state'] in ['canceled', 'failed']:
                print(f"Order {order_id} {order_status['state']}")
                return False
        else:
            print(f"Failed to get order status for {order_id}")
        time.sleep(5)  # Wait for 5 seconds before checking again
    print(f"Order {order_id} not filled within {max_wait_time} seconds")
    return False

def make_money():
    days_running = 0
    take_profit_pct = 0.003  # 0.3% take profit percentage
    trade_active = False
    buy_timestamp = None
    buy_price = None

    while True:  # Continuous loop for live trading
        days_running += 1 
        try:
            daily_data, _, short_ma_daily, long_ma_daily = get_daily_data()
            
            # Get current price and spread information
            price_data = api_trading_client.get_best_bid_ask('BTC-USD')
            if not price_data or 'results' not in price_data or not price_data['results']:
                raise Exception("Failed to get current price and spread information")

            current_price_info = price_data['results'][0]
            print(current_price_info)
            spread = float(current_price_info['buy_spread'])  # Using buy_spread, as it's the same as sell_spread
            
            if short_ma_daily > long_ma_daily and not trade_active:
                # Potential buy signal on daily timeframe
                _120min_data, _, short_ma_120min, long_ma_120min = get_120min_data()
                
                # Calculate DEMA using daily data
                current_dema = calculate_dema(daily_data, spread, trade_active)
                
                if (short_ma_120min > long_ma_120min or _120min_data[-1] > current_dema) and not trade_active:
                    # Buy signal confirmed
                    account = api_trading_client.get_account()
                    if not account or 'buying_power' not in account:
                        raise Exception("Failed to get account information")

                    usd_buying_power = float(account['buying_power'])
                    
                    # Calculate 1% of the portfolio value
                    usd_to_subtract = usd_buying_power * 0.01
                    
                    # Subtract 1% from the buying power
                    usd_buying_power -= usd_to_subtract
                    
                    buy_price = float(current_price_info['ask_inclusive_of_buy_spread'])
                    
                    roundto = 5
                    shifted = (usd_buying_power/buy_price) * (10 ** roundto) 
                    amount_of_btc2 = math.floor(shifted)
                    amount_of_btc2 = amount_of_btc2 / (10 ** roundto)

                    
                    # Place market buy order
                    market_order_config= {
                        "asset_quantity": amount_of_btc2
                    }
                    order_id = str(uuid.uuid4())

                    order_response = api_trading_client.place_order(order_id, "buy", "market", 'BTC-USD', market_order_config)
                    

                    if order_response and 'id' in order_response:
                        if wait_for_order_execution(api_trading_client, order_response['id']):
                            trade_active = True
                            buy_timestamp = datetime.datetime.now()
                            print(f"\n--- BUY ORDER EXECUTED ---")
                            print(f"Time: {buy_timestamp}")
                            print(f"Amount: {amount_of_btc2} BTC")
                            print(f"Price: ${buy_price}")
                            print(f"Total Cost: ${amount_of_btc2 * buy_price}")
                            print(f"Spread: {spread}")
                            print(f"Order ID: {order_response['id']}")
                            print("----------------------------\n")
                    else:
                        raise Exception("Buy order failed")
                else:
                    print("\n--- BUY SIGNAL NOT CONFIRMED ---")
                    print(f"Short MA (120min): {short_ma_120min}")
                    print(f"Long MA (120min): {long_ma_120min}")
                    print(f"Current price: {_120min_data[-1]}")
                    print(f"Current DEMA: {current_dema}")
                    print("Reason: 120-minute indicators do not confirm buy signal")
                    print("------------------------------------\n")
            else:
                if trade_active:
                    print("\n--- NO SELL SIGNAL ---")
                    print(f"Short MA (daily): {short_ma_daily}")
                    print(f"Long MA (daily): {long_ma_daily}")
                    print("Reason: Daily indicators do not suggest selling")
                    print("------------------------\n")
                else:
                    print("\n--- NO BUY SIGNAL ---")
                    print(f"Short MA (daily): {short_ma_daily}")
                    print(f"Long MA (daily): {long_ma_daily}")
                    print("Reason: Daily indicators do not suggest buying")
                    print("------------------------\n")
            
            if trade_active:
                if short_ma_daily < long_ma_daily:
                    # Potential sell signal on daily timeframe
                    _120min_data, _, short_ma_120min, long_ma_120min = get_120min_data()

                    # Update the condition to match ms.py (using window=1 and window=4)
                    long_ma_120min_4 = sum(_120min_data[-4:]) / 4
                    if short_ma_120min < long_ma_120min_4:
                        # Sell signal confirmed
                        holdings = api_trading_client.get_holdings('BTC')
                        if holdings and 'results' in holdings and holdings['results']:
                            btc_to_sell = float(holdings['results'][0]['quantity_available_for_trading'])
                            
                            # Calculate the USD value to subtract (1 USD)
                            usd_to_subtract = 1
                            btc_price = float(current_price_info['bid_inclusive_of_sell_spread'])
                            btc_to_subtract = usd_to_subtract / btc_price
                            
                            # Subtract 1 USD worth of BTC
                            btc_to_sell -= btc_to_subtract
                            
                            btc_to_sell = math.floor(btc_to_sell * 1e5) / 1e5  # Round down to 5 decimal places
                            
                            # Place market sell order with updated order_config
                            market_order_config = {
                                "asset_quantity": btc_to_sell
                            }
                            order_id = str(uuid.uuid4())
                            order_response = api_trading_client.place_order(order_id, "sell", "market", 'BTC-USD', market_order_config)
                            
                            if order_response and 'id' in order_response:
                                if wait_for_order_execution(order_response['id']):
                                    sell_price = float(current_price_info['bid_inclusive_of_sell_spread'])
                                    sell_timestamp = datetime.datetime.now()
                                    profit = (sell_price - buy_price) * btc_to_sell
                                    profit_percentage = (sell_price - buy_price) / buy_price * 100
                                    print(f"\n--- SELL ORDER EXECUTED ---")
                                    print(f"Time: {sell_timestamp}")
                                    print(f"Amount: {btc_to_sell} BTC")
                                    print(f"Price: ${sell_price}")
                                    print(f"Total Revenue: ${btc_to_sell * sell_price}")
                                    print(f"Profit: ${profit}")
                                    print(f"Profit Percentage: {profit_percentage:.2f}%")
                                    print(f"Spread: {spread}")
                                    print(f"Order ID: {order_response['id']}")
                                    print(f"Time held: {sell_timestamp - buy_timestamp}")
                                    print("-----------------------------\n")
                                    trade_active = False
                                    buy_timestamp = None
                                    buy_price = None
                            else:
                                raise Exception("Sell order failed")
                    else:
                        print("\n--- SELL SIGNAL NOT CONFIRMED ---")
                        print(f"Short MA (120min): {short_ma_120min}")
                        print(f"Long MA (120min, window=4): {long_ma_120min_4}")
                        print("Reason: 120-minute indicators do not confirm sell signal")
                        print("-------------------------------------\n")
                
                # Check for take profit condition
                current_time = datetime.datetime.now()
                if (current_time - buy_timestamp).total_seconds() >= (86400*2):
                    current_price = float(current_price_info['bid_inclusive_of_sell_spread'])
                    if (current_price - buy_price) / buy_price >= take_profit_pct:
                        holdings = api_trading_client.get_holdings('BTC')
                        if holdings and 'results' in holdings and holdings['results']:
                            btc_to_sell = float(holdings['results'][0]['quantity_available_for_trading'])
                            btc_to_sell = math.floor(btc_to_sell * 1e5) / 1e5  # Round down to 5 decimal places
                            
                            # Place market sell order for take profit
                            order_config = {"asset_quantity": str(btc_to_sell)}
                            order_id = str(uuid.uuid4())
                            order_response = api_trading_client.place_order(order_id, "sell", "market", 'BTC-USD', order_config)
                            
                            if order_response and 'id' in order_response:
                                if wait_for_order_execution(api_trading_client, order_response['id']):
                                    sell_timestamp = datetime.datetime.now()
                                    profit = (current_price - buy_price) * btc_to_sell
                                    profit_percentage = (current_price - buy_price) / buy_price * 100
                                    print(f"\n--- TAKE PROFIT SELL EXECUTED ---")
                                    print(f"Time: {sell_timestamp}")
                                    print(f"Amount: {btc_to_sell} BTC")
                                    print(f"Price: ${current_price}")
                                    print(f"Total Revenue: ${btc_to_sell * current_price}")
                                    print(f"Profit: ${profit}")
                                    print(f"Profit Percentage: {profit_percentage:.2f}%")
                                    print(f"Spread: {spread}")
                                    print(f"Order ID: {order_response['id']}")
                                    print(f"Time held: {sell_timestamp - buy_timestamp}")
                                    print("-----------------------------------\n")
                                    trade_active = False
                                    buy_timestamp = None
                                    buy_price = None
                            else:
                                raise Exception("Take profit sell order failed")
                    else:
                        print("\n--- TAKE PROFIT CONDITION NOT MET ---")
                        print(f"Current price: ${current_price}")
                        print(f"Buy price: ${buy_price}")
                        print(f"Current profit: {((current_price - buy_price) / buy_price) * 100:.2f}%")
                        print(f"Required profit: {take_profit_pct * 100:.2f}%")
                        print("---------------------------------------\n")
                else:
                    time_held = current_time - buy_timestamp
                    print("\n--- HOLDING POSITION ---")
                    print(f"Time held: {time_held}")
                    print(f"Minimum hold time not yet reached (48 hours)")
                    print("-------------------------\n")
            
            # Add a sleep here to avoid excessive API calls
            print("Program has started day number", days_running)
            time.sleep(86400)  # Sleep for 1440 minutes before next check (1 day)

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            if trade_active:
                sell_all_positions()
            else:
                print("No open positions. Exiting the program.")
                sys.exit()


def start_trading():
    try:
        make_money()
    except KeyboardInterrupt:
        print("Program interrupted by user. Exiting...")
        if 'trade_active' in locals():
            sell_all_positions()
            sys.exit("Exited the program, must still sell your btc.")
        sys.exit()

if __name__ == "__main__":
    start_trading()





