import numpy as np
import tpqoa
from sklearn.linear_model import LogisticRegression
import pandas as pd
from os.path import exists
from indicators import IndicatorCalculator


class MLModel:
    ''' Class for the vectorized backtesting of Machine Learning-based trading strategies (Classification).
    '''

    def __init__(self, symbol, start, end, timeframe, tc, features=None):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.timeframe = timeframe
        self.tc = tc
        self.features = features

        self.model = LogisticRegression(C=1e6, max_iter=100000, multi_class="ovr")
        self.data = None
        self.training_data, self.testing_data = None, None
        self.get_data()

    def get_data(self):
        ''' Imports the data from five_minute_pairs.csv (source can be changed).'''
        print(f'Getting data: instrument={self.symbol}, start={self.start}, end={self.end}, timeframe={self.timeframe}')
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
        mid.drop(['complete'], axis=1, inplace=True)
        self.data = mid.dropna()
        self.get_features_data()
        print('Data retrieved, preparing training/testing data...')
        self.get_training_and_testing_data(training_ratio=0.7)

    def get_features_data(self):
        print('Getting features...')
        feat_collector = IndicatorCalculator(self.data)
        feat_collector.calculate_features(self.features)
        self.data = feat_collector.get_data()
        print('Features acquired.')

    def get_training_and_testing_data(self, training_ratio):
        # determining datetime for start, end and split (for training an testing period)
        split_index = int(len(self.data) * training_ratio)
        train_end = self.data.index[(split_index - 1)]
        train_start = self.data.index[0]
        test_end = self.data.index[-1]
        self.training_data = self.data.loc[train_start:train_end].copy()
        self.testing_data = self.data.loc[train_end:test_end].copy()
        self.training_data.dropna()
        self.testing_data.dropna()
        print('Training/testing data prepared.')

    def fit_model(self):
        ''' Fitting the ML Model.'''
        self.model.fit(self.testing_data[self.features], np.sign(self.testing_data["returns"]))
