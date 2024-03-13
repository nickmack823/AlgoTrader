import itertools
import random
import numpy as np
import tpqoa
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
from indicator_getters.indicators import IndicatorCalculator
from model_classes import DNNModel

plt.style.use("seaborn")


class MLBacktester:
    ''' Class for the vectorized backtesting of Machine Learning-based trading strategies (Classification).
    '''

    def __init__(self, symbol, start, end, timeframe, tc):
        '''
        Parameters
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        start: str
            start date for data import
        end: str
            end date for data import
        tc: float
            proportional transaction/trading costs per trade
        '''
        self.symbol = symbol
        self.start = start
        self.end = end
        self.timeframe = timeframe
        self.tc = tc
        self.model = LogisticRegression(C=1e6, max_iter=100000, multi_class="ovr")
        self.results = None
        self.data = None
        self.training_data, self.testing_data = None, None
        self.feature_columns = None
        self.data_path = f'forex_data/{self.symbol}_{self.start}_{self.end}_{self.timeframe}_features.csv'
        # self.data_path = r"C:\Users\Administrator\Desktop\AlgoTrader\forex_data\{}_{}_{}_{}_features.csv"\
        #     .format(self.symbol, self.start, self.end, self.timeframe)
        self.get_data()

        # Metrics
        self.trades = 0
        self.units = 0
        self.trades = 0
        self.position = 0

    def get_data(self):
        print(f'Getting data: instrument={self.symbol}, start={self.start}, end={self.end}, timeframe={self.timeframe}')
        if exists(self.data_path):
            self.data = pd.read_csv(self.data_path, parse_dates=['time'], index_col='time').dropna()
        else:
            api = tpqoa.tpqoa('/oanda.cfg')
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
        print('Data retrieved.')
        self.get_training_and_testing_data(training_ratio=0.7)

    def get_features_data(self):
        print('Getting features...')
        feat_collector = IndicatorCalculator(self.data)
        feat_collector.calculate_features()
        feat_collector.to_file(self.data_path)
        self.data = pd.read_csv(self.data_path, parse_dates=['time'], index_col='time').dropna()
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

    def dnn(self):
        print('Deep Neural Network...')
        df = self.data.copy()
        df['direction'] = np.where(df['returns'] > 0, 1, 0)  # 0 or 1 for DNN model
        df.dropna(inplace=True)

        # excluded_features = ['returns', 'bid', 'ask', 'spread', 'complete']
        # features = [c for c in df.columns if c not in excluded_features]
        features = ['paverage2close']
        cols = []
        lags = 5
        # Adding Feature Lags
        for f in features:
            for lag in range(1, lags + 1):
                col = f'{f}_lag_{lag}'
                df[col] = df[f].shift(lag)
                cols.append(col)
            df.dropna(inplace=True)

        # Split into Train/Test
        split = int(len(df) * 0.66)
        train = df.iloc[:split].copy()
        test = df.iloc[split:].copy()

        # Feature Scaling (Standardization/Data Normalization)
        mean, std = train.mean(), train.std()
        train_standardized = (train - mean) / std

        # Creating and Fitting the Model
        DNNModel.set_seeds(100)
        dnn_model = DNNModel.create_model(hl=3, hu=50, dropout=True, input_dim=len(cols))
        dnn_model.fit(x=train_standardized[cols], y=train['direction'], epochs=50, verbose=False,
                      validation_split=0.2, shuffle=False, class_weight=DNNModel.cw(train))

        print(dnn_model.evaluate(train_standardized[cols], train['direction']))  # Evaluate fit on training set

        # Predicting on Test Set
        test_standardized = (test - mean) / std  # standardize WITH TRAIN SET PARAMS

        pred = dnn_model.predict(test_standardized[cols])
        plt.hist(pred, bins=50)
        plt.show()
        test['probability'] = dnn_model.predict(test_standardized[cols])

        #
        test['position'] = np.where(test.probability < 0.50, -1, np.nan)  # short where prob < 0.47
        test['position'] = np.where(test.probability > 0.50, 1, test.position)  # long where prob < 0.53

        print(test.probability)

        test.index = test.index.tz_localize('UTC')
        test['NYTime'] = test.index.tz_convert('America/New_York')
        test['hour'] = test.NYTime.dt.hour
        test['position'] = np.where(~test.hour.between(2, 12), 0, test.position)  # neutral in non-busy hours
        test['position'] - test.position.ffill()  # in all other cases: hold position
        test.position.value_counts(dropna=False)

        test['strategy'] = test['position'] * test['returns']
        test["creturns"] = test["returns"].cumsum().apply(np.exp)
        test["cstrategy"] = test['strategy'].cumsum().apply(np.exp)
        test[['creturns', 'cstrategy']].plot(figsize=(12, 8))

    def fit_model(self, features):
        ''' Fitting the ML Model.'''
        self.feature_columns = features
        self.model.fit(self.testing_data[features], np.sign(self.testing_data["returns"]))

    def test_strategy(self):
        '''Backtests the ML-based strategy.'''
        # make predictions on the test set
        predict = self.model.predict(self.testing_data[self.feature_columns])
        self.testing_data["pred"] = predict  # prediction is position taken

        # calculate Strategy Returns
        self.testing_data["strategy"] = self.testing_data["pred"] * self.testing_data["returns"]

        # determine the number of trades in each bar
        self.testing_data["trades"] = self.testing_data["pred"].diff().fillna(0).abs()

        # subtract transaction/trading costs from pre-cost return
        # self.testing_data.strategy = self.testing_data.strategy - self.testing_data.trades * self.tc

        # calculate cumulative returns for strategy & buy and hold
        self.testing_data["creturns"] = self.testing_data["returns"].cumsum().apply(np.exp)
        self.testing_data["cstrategy"] = self.testing_data['strategy'].cumsum().apply(np.exp)
        self.testing_data['strategy_net'] = self.testing_data.strategy - (self.testing_data.trades * self.tc)
        self.testing_data['cstrategy_net'] = self.testing_data.strategy_net.cumsum().apply(np.exp)
        self.results = self.testing_data

        hits = np.sign(self.results.returns * self.results.pred).value_counts()
        correct_ratio = hits[1.0] / sum(hits)

        perf = self.results["cstrategy"].iloc[-1]  # absolute performance of the strategy
        outperf = perf - self.results["creturns"].iloc[-1]  # out-/underperformance of strategy
        net = self.results['cstrategy_net'].iloc[-1]
        metrics = {'Absolute Performance': round(perf, 4),
                   'Outperformance': round(outperf, 4),
                   'Net Gain': round(net, 4),
                   'Correct Pred Ratio': round(correct_ratio, 4)}

        return metrics

    def plot_results(self):
        ''' Plots the performance of the trading strategy and compares to "buy and hold".'''
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = "Logistic Regression: {} | TC = {}".format(self.symbol, self.tc)
            self.results[["creturns", "cstrategy", 'cstrategy_net']].plot(title=title, figsize=(12, 8))
            plt.show()
            print(self.results)

    def hours_granularity(self, timeframe=None):
        df = pd.read_csv(self.data_path, parse_dates=['time'], index_col='time', usecols=['time', 'spread', 'close'])
        if timeframe is not None:
            df = df.resample(timeframe).last().dropna()
        df['NYTime'] = df.index.tz_convert('America/New_York')
        df['hour'] = df.NYTime.dt.hour
        df['price_change_abs'] = df.close.diff().abs()
        df['cover_cost'] = df.price_change_abs > df.spread

        df.dropna().groupby('hour').cover_cost.mean().plot(kind='bar', figsize=(12, 8), fontsize=13)
        plt.xlabel('NY Time', fontsize=15)
        plt.ylabel('Cover Costs', fontsize=15)
        plt.title(f'Granularity: {timeframe}', fontsize=18)
        plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        plt.show()


def test_features(model, features, to_file=True):
    test_file = f'feature_tests/feature_tests_{model.symbol}_{model.start}_{model.end}_{model.timeframe}.txt'

    if not exists(test_file) and to_file:
        with open(test_file, 'w') as f:
            pass

    for n in range(0, len(features)):
        top_10 = test_feature_combinations(model, features, n + 1)
        print(f'Top 10 {n + 1}-Feature Combinations: {top_10}')

        if to_file:
            f = open(test_file, 'r')
            lines = f.readlines()
            f.close()
            with open(test_file, 'w') as f:
                f.writelines(lines)
                f.write(f'Top 10 {n + 1}-Feature Combinations: {top_10}\n')


def test_feature_combinations(model, features, comb_number):
    # combs = list(itertools.combinations(features, comb_number))
    print(f'Testing {comb_number}-Feature Combinations')
    # ib = IterativeBacktest('EUR_USD', '2022-01-01', '2022-06-28', 'M5', 1000, use_spread=True)
    if comb_number > 3:
        combs = []
        while len(combs) < 10000:
            def get_rand_comb():
                random_comb = [feat for feat in random.sample(features, comb_number)]
                if random_comb in combs:
                    get_rand_comb()
                else:
                    return random_comb

            combs.append(get_rand_comb())
    else:
        combs = list(itertools.combinations(features, comb_number))

    result_combs, results = [], []
    next_print = 0.1
    for comb in combs:
        if comb is None:
            continue
        comb_list = [f for f in comb]
        model.fit_model(comb_list)
        result_combs.append(comb_list)
        # ib.model = model.model
        # results.append(ib.test_strategy(comb_list))
        results.append(model.test_strategy())
        if (combs.index(comb) + 1) / len(combs) > next_print:
            print(f'{combs.index(comb) + 1}/{len(combs)} feature combinations tested.')
            next_print += 0.1

    for comb in result_combs:
        results[result_combs.index(comb)]['Features'] = comb
    r = results.copy()
    # r.sort(key=lambda result: result['Net Gain'])
    r.sort(key=lambda result: result['Correct Pred Ratio'])
    top_10 = r[-10:]
    top_10.reverse()
    return top_10


if __name__ == "__main__":
    # TODO: Find best timeframe for optimizing trading costs
    ptc = 0.00007
    model = MLBacktester('EUR_USD', '2022-01-01', '2022-06-28', 'M30', ptc)
    model.dnn()
    excluded_features = ['returns', 'bid', 'ask', 'spread', 'complete']
    features = [c for c in model.data.columns if c not in excluded_features]
    # # features = ['paverage2close', 'proc15close']
    # test_features(model, features, to_file=False)
    # model.hours_granularity(timeframe='30min')
