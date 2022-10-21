import os
from os.path import exists

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tpqoa

from model_classes.MLModel import MLModel
from indicator_getters.indicators import IndicatorCalculator
plt.style.use("seaborn")


class IterativeBase:
    ''' Base class for iterative (event-driven) backtesting of trading strategies.
    '''

    def __init__(self, symbol, start, end, timeframe, balance, strategy, use_spread=True):
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
        self.strategy = strategy
        self.initial_balance = balance
        self.current_balance = balance
        self.units = round(self.current_balance * 10)
        self.trades = 0
        self.position = 0
        self.position_opened_bar = None
        self.take_profit = None
        self.stop_loss = None
        self.use_spread = use_spread
        self.data_path = f"forex_data/{self.symbol}_{self.start}_{self.end}_{self.timeframe}_features.csv"
        self.data = None

        self.dirname = os.path.dirname(__file__)

        self.model_features = ['paverage2close', 'proc15close']
        self.features_to_get = ['paverage2close', 'proc15close', 'atr']
        self.model_timeframe = timeframe
        self.model = self.get_model()
        self.get_data()

    def get_model(self):
        model_path = os.path.join(self.dirname, f"models/{self.symbol}-{self.start}-{self.end}-{self.timeframe}.joblib")
        if exists(model_path):
            print("Getting pre-existing model.")
            return joblib.load(model_path)
        else:
            print('Getting model...')
            ml_model = MLModel(self.symbol, self.start, self.end, self.model_timeframe, 0.00007,
                               features=self.model_features)
            ml_model.fit_model()
            print('Model acquired.')
            joblib.dump(ml_model.model, model_path)
            return ml_model.model

    def get_data(self):
        ''' Imports the data from five_minute_pairs.csv (source can be changed).'''
        print(f'Getting data: instrument={self.symbol}, start={self.start}, end={self.end}, timeframe={self.timeframe}')
        if exists(self.data_path):
            self.data = pd.read_csv(self.data_path, parse_dates=['time'], index_col='time').dropna()
        else:
            api = tpqoa.tpqoa(os.path.join(self.dirname, '../oanda.cfg'))
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
        feat_collector.calculate_features(self.features_to_get)
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
        datetime = str(self.data.index[bar])
        close = round(self.data.close.iloc[bar], 5)
        spread = round(self.data.spread.iloc[bar], 5)
        return datetime, close, spread

    def print_current_balance(self, bar):
        ''' Prints out the current (cash) balance.
        '''
        date, close, spread = self.get_values(bar)
        # print("{} | Current Balance: {}".format(date, round(self.current_balance, 2)))

    def print_current_position_value(self, bar):
        ''' Prints out the current position value.
        '''
        date, close, spread = self.get_values(bar)
        cpv = self.units * close
        print("{} |  Current Position Value = {}".format(date, round(cpv, 2)))

    def print_current_nav(self, bar):
        ''' Prints out the current net asset value (nav).
        '''
        date, close, spread = self.get_values(bar)
        nav = self.current_balance + self.units * close
        print("{} |  Net Asset Value = {}".format(date, round(nav, 2)))

    # def close_pos(self, bar):
    #     ''' Closes out a long or short position (go neutral).
    #     '''
    #     date, close, spread = self.get_values(bar)
    #     # print(75 * "-")
    #     # print("{} | +++ CLOSING FINAL POSITION +++".format(date))
    #     self.current_balance += self.units * close  # closing final position (works with short and long!)
    #     self.current_balance -= (abs(self.units) * spread / 2 * self.use_spread)  # substract half-spread costs
    #     # print("{} | closing position of {} for {}".format(date, self.units, close))
    #     self.trades += 1
    #     perf = (self.current_balance - self.initial_balance) / self.initial_balance * 100
    #     self.print_current_balance(bar)
    #     # print("{} | net performance (%) = {}".format(date, round(perf, 2)))
    #     # print("{} | number of trades executed = {}".format(date, self.trades))
    #     # print(75 * "-")
    #     metrics = {'Net Performance': round(perf, 5)}
    #     return metrics