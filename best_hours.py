from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tpqoa
plt.style.use('seaborn')

# PARAMETERS
symbol = 'USD_JPY'
start = '2021-01-01'
yesterday = datetime.now() - timedelta(days=1)
yesterday = datetime.strftime(yesterday, '%Y-%m-%d')
end = yesterday
timeframe = 'M30'

api = tpqoa.tpqoa('C:/Users/Nick/Documents/GitHub/AlgoTrader/oanda.cfg')
mid = api.get_history(instrument=symbol, start=start, end=end, granularity=timeframe,
                      price='M', localize=False)
bid = api.get_history(instrument=symbol, start=start, end=end, granularity=timeframe,
                      price='B')
ask = api.get_history(instrument=symbol, start=start, end=end, granularity=timeframe,
                      price='A')
mid['bid'] = bid.c
mid['ask'] = ask.c
mid['spread'] = (bid.c - ask.c).to_frame()
mid.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': "close"}, inplace=True)
mid.drop(['complete'], axis=1, inplace=True)
data = mid.dropna()

data['NYTime'] = data.index.tz_convert('America/New_York')
data['hour'] = data.NYTime.dt.hour
data['price_change_abs'] = data.close.diff().abs()
data.dropna(inplace=True)

print(data.columns)
by_hour = data.groupby('hour')[['spread', 'price_change_abs']].mean()
by_hour.loc[0:24, 'spread'].plot(kind='bar', figsize=(12, 8), fontsize=13)
plt.show()

