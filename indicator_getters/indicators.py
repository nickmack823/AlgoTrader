import tpqoa
import pandas as pd
import pandas_ta as ta


class IndicatorCalculator:

    def __init__(self, data):
        self.data = data
        self.features_dict = {
            'paverage': self.paverage,
            'stochastic': self.stochastic,
            'proc': self.proc,
            'macd': self.macd,
            'momentum': self.momentum,
            'bollinger': self.bollinger,
            'williams': self.williams,
            'sinewave': self.sinewave,
            'cci': self.cci,
            'slope': self.slope,
            'ema': self.ema,
            'sma': self.sma,
            'vwap': self.vwap,
            'stdev': self.stdev,
            'atr': self.atr,
            'rsi': self.rsi,
            'adosc': self.adosc}

    def display_data(self):
        print(self.data.columns)
        print(self.data.head())
        print(self.data.tail())

    def to_file(self, file_path):
        self.data.to_csv(file_path)

    def returns(self):
        self.data.ta.log_return(cumulative=False, append=True)
        self.data.rename(columns={'LOGRET_1': 'returns'}, inplace=True)

    def paverage(self, period=2):
        averages = pd.DataFrame(self.data[['open', 'high', 'low', 'close']].rolling(period).mean())
        for col in averages.columns:
            self.data[f'paverage{period}{col}'] = averages[col]

    def stochastic(self, periods=(3, 4, 5, 8, 9, 10)):
        # for period in periods:
        #     self.data.ta.stoch(k=period, append=True)
        self.data.ta.stoch(k=5, d=3, append=True)

    # PROC (Price Rate of Change)
    def proc(self, periods=(12, 13, 14, 15)):
        for period in periods:
            close_today = self.data.close.iloc[period:]
            close_yesterday = self.data.close.iloc[:-period]

            proc = pd.DataFrame(close_today - close_yesterday.values / close_yesterday.values)
            proc.columns = ['close']
            for col in proc.columns:
                self.data[f'proc{period}{col}'] = proc[col]

    def macd(self, fast=15, slow=30):
        self.data.ta.macd(fast=fast, slow=slow, append=True)
        # self.data.rename(columns={f'MACDh_{fast}_{slow}_9': f'MACD_hist_{fast}_{slow}_9',
        #                           f'MACDs_{fast}_{slow}_9': f'MACD_signal_{fast}_{slow}_9'}, inplace=True)

    def momentum(self, periods=(3, 4, 5, 8, 9, 10)):
        for period in periods:
            self.data.ta.mom(length=period, append=True)

    def bollinger(self, period=20):
        self.data.ta.bbands(length=period, std=2, append=True)

    def williams(self, periods=(6, 7, 8, 9, 10)):
        for period in periods:
            self.data.ta.willr(length=period, append=True)

    # Binary Values
    def candlestick_patterns(self):
        self.data.ta.cdl_pattern(name='doji', append=True)

    def psar(self):
        self.data.ta.psar(append=True)

    def sinewave(self):
        self.data.ta.ebsw(append=True)

    def cci(self):
        self.data.ta.cci(append=True)

    def slope(self, periods=(3, 4, 5, 10, 20, 30)):
        for period in periods:
            self.data.ta.slope(length=period, append=True)

    def ema(self, periods=(10, 50, 100)):
        for period in periods:
            self.data.ta.ema(length=period, append=True)

    def sma(self):
        self.data.ta.sma(append=True)

    def vwap(self):
        self.data.ta.vwap(append=True)

    def stdev(self,):
        self.data.ta.tos_stdevall(append=True)

    def atr(self):
        self.data.ta.atr(append=True)

    def rsi(self):
        self.data.ta.rsi(append=True)

    def adosc(self, periods=(2, 3, 4, 5)):
        for period in periods:
            self.data.ta.adosc(length=period, append=True)

    def vol_profile(self):
        help(ta.vp)
        self.data.ta.vp(append=True)

    def calculate_features(self, features=None):
        if features is None:
            print('Calculating all features...')
            functions = [self.paverage, self.stochastic, self.proc, self.macd, self.momentum, self.bollinger,
                         self.williams, self.sinewave, self.cci, self.slope, self.ema, self.sma, self.vwap,
                         self.stdev, self.atr, self.rsi, self.adosc]
        else:
            print(f'Calculating features: {features}')
            functions = []
            for key in self.features_dict.keys():
                for feat in features:
                    if key in feat:
                        functions.append(self.features_dict[key])

        for func in functions:
            func()
        print('Features calculated.')

    def get_data(self):
        return self.data

    def set_data(self, new_data):
        self.data = new_data


if __name__ == "__main__":
    symbol, start, end, timeframe = 'EUR_USD', '2022-07-20', '2022-07-21', 'M5'
    api = tpqoa.tpqoa('/oanda.cfg')
    mid = api.get_history(instrument=symbol, start=start, end=end, granularity=timeframe,
                          price='M')
    bid = api.get_history(instrument=symbol, start=start, end=end, granularity=timeframe,
                          price='B')
    ask = api.get_history(instrument=symbol, start=start, end=end, granularity=timeframe,
                          price='A')
    mid['bid'] = bid.c
    mid['ask'] = ask.c
    mid['spread'] = (bid.c - ask.c).to_frame()
    mid.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': "close"}, inplace=True)
    data = mid.dropna()
    data.drop(labels='complete', axis=1, inplace=True)

    ic = IndicatorCalculator(data)
    ic.calculate_features(['macd', 'ema'])
    ic.display_data()
    ic.to_file('bingus.csv')



