DEMA + SUPERTREND Strategy BTC bot. Built for coinbase but could work with more. 
Current supertrend settings are period of 12 and multiplier of 3

Working Version: working-version.py is the most up to date version of the bot, but could include logic bugs in trading strategy/calculations


Before Running/Installation - 
Set Coinbase REST API keys (The non pro/advanced one), in all three spots. 
pip install all required libraries. 



IMPROVEMENTS TO BE MADE - 
  1. Create a way to properly backtest the exact strategy being used. Need historical data by the hour for the DEMA, and need historical data by the minute for supertrend. Stoploss ordrs in backtesting?  
 
  3. Adding some sort of RSI index to the buy and sell conditions. 
