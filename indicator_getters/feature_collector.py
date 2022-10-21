from indicator_feature_functions import *


class FeatureCollector:

    def __init__(self, prices):
        self.prices = prices
        self.master = None

    def get_features_df(self, excluded=None):
        """
        :param prices: a cleaned DataFrame of price values
        :param file_path: path to save features CSV to
        """
        # Lists for each period set for indicators
        keys = {'momentum': ([3, 4, 5, 8, 9, 10], momentum),
                'stochastic': ([3, 4, 5, 8, 9, 10], stochastic),
                'williams': ([6, 7, 8, 9, 10], williams),
                'proc': ([12, 13, 14, 15], proc),
                'wadl': ([15], wadl),
                'adosc': ([2, 3, 4, 5], adosc),
                'macd': ([15, 30], macd),
                'cci': ([15], cci),
                'bollinger': ([15], bollinger),
                'heikenashi': ([15], heiken_ashi),
                'paverage': ([2], paverage),
                'slope': ([3, 4, 5, 10, 20, 30], slope),
                'fourier': ([10, 20, 30], fourier_fitting),
                'sine': ([5, 6], sine_fitting)
                }
        # List of columns names
        features = keys.keys()
        # for feature in features:
        #     print(f'Calculating {feature}...')
        #     periods, func = keys[feature]
        #     if feature == 'bollinger':
        #         result = bollinger(self.prices, periods, deviations=2)
        #     elif feature == 'heikenashi':
        #         heikenashi_prices = self.prices.copy()
        #         heikenashi_prices['Symbol'] = 'SYMB'
        #         HKA = OHLCresample(heikenashi_prices, '15H')
        #         result = heiken_ashi(HKA, periods)
        #     else:
        #         result = func(self.prices, periods)

        # Calculate all features for indicators
        print('Calculating Momentum...')
        momentum_result = momentum(self.prices, keys['momentum'][0])
        print('Calculating Stochastic...')
        stochastic_result = stochastic(self.prices, keys['stochastic'][0])
        print('Calculating Williams %R...')
        williams_result = williams(self.prices, keys['williams'][0])
        print('Calculating Price Rate of Change...')
        proc_result = proc(self.prices, keys['proc'][0])
        print('Calculating WADL...')
        wadl_result = wadl(self.prices, keys['wadl'][0])
        print('Calculating ADOSC...')
        adosc_result = adosc(self.prices, keys['adosc'][0])
        print('Calculating MACD...')
        macd_result = macd(self.prices, keys['macd'][0])
        print('Calculating Momentum...')
        cci_result = cci(self.prices, keys['cci'][0])
        print('Calculating Bollinger Bands...')
        bollinger_result = bollinger(self.prices, keys['bollinger'][0], deviations=2)

        print('Calculating Heiken Ashi...')
        heikenashi_prices = self.prices.copy()
        heikenashi_prices['Symbol'] = 'SYMB'
        HKA = OHLCresample(heikenashi_prices, '15H')
        heiken_ashi_result = heiken_ashi(HKA, keys['heikenashi'][0])

        print('Calculating Price Average...')
        paverage_result = paverage(self.prices, keys['paverage'][0])
        print('Calculating Slope...')
        slope_result = slope(self.prices, keys['slope'][0])
        print('Calculating Fourier...')
        fourier_result = fourier_fitting(self.prices, keys['fourier'][0])
        print('Calculating Sine...')
        sine_result = sine_fitting(self.prices, keys['sine'][0])

        # Create list of resulting dictionaries
        results = [momentum_result.close, stochastic_result.close, williams_result.close, proc_result.proc,
                   wadl_result.wadl, adosc_result.AD, macd_result.line, cci_result.cci,
                   bollinger_result.bands, heiken_ashi_result.candles, paverage_result.price_averages,
                   slope_result.slope, fourier_result.coefficients, sine_result.coefficients]


        # Populate the master DataFrame
        master = self.prices.copy()

        for result, feature in zip(results, features):
            if feature == 'macd':
                settings = keys['macd'][0]
                col_id = f'{feature}{settings[0]}{settings[1]})'
                master[col_id] = result
            else:
                for param in keys[feature][0]:
                    for param_val in list(result[param]):
                        col_id = f'{feature}{param}{param_val}'
                        vals = result[param][param_val]
                        master[col_id] = vals

        threshold = round(0.7 * len(master))  # For removing columns that don't have (threshold) clean data

        master[['open', 'high', 'low', 'close']] = self.prices[['open', 'high', 'low', 'close']]

        # Heiken Ashi is resampled, meaning there will be some empty data
        master.heikenashi15open = master.heikenashi15open.fillna(method='bfill')
        master.heikenashi15high = master.heikenashi15high.fillna(method='bfill')
        master.heikenashi15low = master.heikenashi15low.fillna(method='bfill')
        master.heikenashi15close = master.heikenashi15close.fillna(method='bfill')

        # Drop columns w/ >= 30% NaN data
        master_cleaned = master.copy()
        master_cleaned = master_cleaned.dropna(axis=1, thresh=threshold)

        self.master = master_cleaned
        return self.master

    def to_file(self, file_path):
        self.master.to_csv(file_path)


