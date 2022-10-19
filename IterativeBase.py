from os.path import exists

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tpqoa
from indicators import IndicatorCalculator
plt.style.use("seaborn")


class IterativeBase:
    ''' Base class for iterative (event-driven) backtesting of trading strategies.
    '''

    def __init__(self, symbol, start, end, timeframe, balance, use_spread=True):
        '''
        Parameters
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        start: str
            start date for data import
        end: str
            end date for data import
        balance: float
            initial amount to be invested per trade
        use_spread: boolean (default = True) 
            whether trading costs (bid-ask spread) are included
        '''
        self.symbol = symbol
        self.start = start
        self.end = end
        self.timeframe = timeframe
        self.initial_balance = balance
        self.current_balance = balance
        self.units = 0
        self.trades = 0
        self.position = 0
        self.use_spread = use_spread
        self.data_path = f"forex_data/{self.symbol}_{self.start}_{self.end}_{self.timeframe}_features.csv"
        self.data = None
        self.model = joblib.load('model.joblib')
        self.get_data()

    def get_data(self):
        ''' Imports the data from five_minute_pairs.csv (source can be changed).'''
        print(f'Getting data: instrument={self.symbol}, start={self.start}, end={self.end}, timeframe={self.timeframe}')
        if exists(self.data_path):
            self.data = pd.read_csv(self.data_path, parse_dates=['time'], index_col='time').dropna()
        else:
            api = tpqoa.tpqoa('C:/Users/Nick/Documents/GitHub/AlgoTrader/oanda.cfg')
            mid = api.get_history(instrument=self.symbol, start=self.start, end=self.end, granularity=self.timeframe,
                                  price='M')
            bid = api.get_history(instrument=self.symbol, start=self.start, end=self.end, granularity=self.timeframe,
                                  price='B')
            ask = api.get_history(instrument=self.symbol, start=self.start, end=self.end, granularity=self.timeframe,
                                  price='A')
            mid['bid'] = bid.c
            mid['ask'] = ask.c
            mid['spread'] = (bid.c - ask.c).to_frame()
            mid.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': "close"}, inplace=True)
            mid["returns"] = np.log(mid.close / mid.close.shift(1))
            self.data = mid.dropna()
            self.get_features_data()
        print('Data retrieved.')

    def get_features_data(self):
        print('Getting features...')
        feat_collector = IndicatorCalculator(self.data)
        feat_collector.calculate_features()
        feat_collector.to_file(self.data_path)
        self.data = pd.read_csv(self.data_path, parse_dates=['time'], index_col='time').dropna()
        print('Features acquired.')

    def plot_data(self, cols=None):
        ''' Plots the closing price for the symbol.
        '''
        if cols is None:
            cols = "close"
        self.data[cols].plot(figsize=(12, 8), title=self.symbol)

    def get_values(self, bar):
        ''' Returns the date, the price and the spread for the given bar.
        '''
        date = str(self.data.index[bar].date())
        close = round(self.data.close.iloc[bar], 5)
        spread = round(self.data.spread.iloc[bar], 5)
        return date, close, spread

    def print_current_balance(self, bar):
        ''' Prints out the current (cash) balance.
        '''
        date, close, spread = self.get_values(bar)
        # print("{} | Current Balance: {}".format(date, round(self.current_balance, 2)))

    def buy_instrument(self, bar, units=None, amount=None):
        ''' Places and executes a buy order (market order).
        '''
        date, close, spread = self.get_values(bar)
        if self.use_spread:
            close += spread / 2  # ask price
        if amount is not None:  # use units if units are passed, otherwise calculate units
            units = int(amount / close)
        self.current_balance -= units * close  # reduce cash balance by "purchase price"
        self.units += units
        self.trades += 1
        # print("{} |  Buying {} for {}".format(date, units, round(close, 5)))

    def sell_instrument(self, bar, units=None, amount=None):
        ''' Places and executes a sell order (market order).
        '''
        date, close, spread = self.get_values(bar)
        if self.use_spread:
            close -= spread / 2  # bid price
        if amount is not None:  # use units if units are passed, otherwise calculate units
            units = int(amount / close)
        self.current_balance += units * close  # increases cash balance by "purchase price"
        self.units -= units
        self.trades += 1
        # print("{} |  Selling {} for {}".format(date, units, round(close, 5)))

    def print_current_position_value(self, bar):
        ''' Prints out the current position value.
        '''
        date, close, spread = self.get_values(bar)
        cpv = self.units * close
        # print("{} |  Current Position Value = {}".format(date, round(cpv, 2)))

    def print_current_nav(self, bar):
        ''' Prints out the current net asset value (nav).
        '''
        date, close, spread = self.get_values(bar)
        nav = self.current_balance + self.units * close
        # print("{} |  Net Asset Value = {}".format(date, round(nav, 2)))

    def close_pos(self, bar):
        ''' Closes out a long or short position (go neutral).
        '''
        date, close, spread = self.get_values(bar)
        # print(75 * "-")
        # print("{} | +++ CLOSING FINAL POSITION +++".format(date))
        self.current_balance += self.units * close  # closing final position (works with short and long!)
        self.current_balance -= (abs(self.units) * spread / 2 * self.use_spread)  # substract half-spread costs
        # print("{} | closing position of {} for {}".format(date, self.units, close))
        self.units = 0  # setting position to neutral
        self.trades += 1
        perf = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        self.print_current_balance(bar)
        # print("{} | net performance (%) = {}".format(date, round(perf, 2)))
        # print("{} | number of trades executed = {}".format(date, self.trades))
        # print(75 * "-")
        metrics = {'Net Performance': round(perf, 5)}
        return metrics
