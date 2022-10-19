import pandas as pd
import numpy as np
from scipy import stats
import scipy.optimize
from scipy.optimize import OptimizeWarning
import warnings
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# from matplotlib.finance import _candlestick
from matplotlib.dates import date2num
from datetime import datetime


class Holder:
    pass


# Heiken Ashi
def heiken_ashi(prices: pd.DataFrame, periods: list):
    """
    :param prices: dataframe of OHLC and volume data
    :param periods: periods for which to create the candles
    :return: heiken ashi OHLC candles
    """
    results = Holder()
    heiken_dict = {}

    # HA formula calculations
    HA_close = prices[['open', 'high', 'low', 'close']].sum(axis=1) / 4
    HA_open, HA_high, HA_low = HA_close.copy(), HA_close.copy(), HA_close.copy()
    HA_open.iloc[0] = HA_close.iloc[0]

    for i in range(1, len(prices)):
        HA_open.iloc[i] = (HA_open.iloc[i-1] + HA_close.iloc[i-1]) / 2
        HA_high.iloc[i] = max([prices.high.iloc[i], HA_open.iloc[i], HA_close.iloc[i]])
        HA_low.iloc[i] = min([prices.high.iloc[i], HA_open.iloc[i], HA_close.iloc[i]])

    df = pd.concat((HA_open, HA_high, HA_low, HA_close), axis=1)
    df.columns = ['open', 'high', 'low', 'close']

    #df.index = df.index.droplevel(0)

    heiken_dict[periods[0]] = df

    results.candles = heiken_dict

    return results

# def fourier_series(self, x, a0, a1, b1, w):
#     """
#     Fourier Series expansion fitting function
#     :param x: the hours (independent variable)
#     :param a0: first fourier series coefficient
#     :param a1: second fourier series coefficient
#     :param b1: third fourier series coefficient
#     :param w: fourier series frequency
#     :return: the value of the fourier function
#     """
#     # F = a0 + a1*cos(wx) + b1*sin(wx)
#     f = a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x)
#     return f
#
# def sine_series(self, x, a0, b1, w):
# """
# Sine Series expansion fitting function
# :param x: the hours (independent variable)
# :param a0: first sine series coefficient
# :param b1: second sine series coefficient
# :param w: sine series frequency
# :return: the value of the sine function
# """
# s = a0 + b1 * np.sin(w * x)
# return s
#
# def detrend(self, prices, method='difference'):
# """
# Detrends data to allow for Fourier/Sine fitting
# :param prices: datafrake of OHLC currency data
# :param method: method of detrending - 'linear' or 'difference'
# :return: detrended price series
# """
# detrended = None
# if method == 'difference':
#     # '1' is now, '0' is yesterday; subtracts yesterday's value from each day's value
#     detrended = prices.close[1:] - prices.close[:-1].values
# elif method == 'linear':
#     x = np.arange(0, len(prices))  # range of numbers from 0 to len(prices)
#     y = prices.close.values
#     model = LinearRegression()
#     model.fit(x.reshape(-1, 1), y.reshape(-1, 1))  # Fits lin. regression line to values
#
#     trend = model.predict(x.reshape(-1, 1))
#     trend = trend.reshape((len(prices),))
#
#     detrended = prices.close - trend
#
# return detrended
#
# def fourier_fitting(self, periods=(10, 20, 30), method='difference'):
# # Compute the coefficients of the series
# detrended = self.detrend(self.data, method=method)
#
# for period in periods:
#     coefficients = []
#     for i in range(period, len(self.data) - period):
#         x = np.arange(0, period)
#         y = detrended.iloc[i - period:i]  # shifter
#
#         # Ignores errors from optimizer when it can't fit to curve
#         with warnings.catch_warnings():
#             warnings.simplefilter('error', OptimizeWarning)
#
#             try:
#                 result = scipy.optimize.curve_fit(self.fourier_series, x, y)
#             except (RuntimeError, OptimizeWarning) as e:
#                 result = np.empty((1, 4))  # set each of 4 params to NaN
#                 result[0, :] = np.NAN
#
#         coefficients = np.append(coefficients, result[0], axis=0)
#
#     warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
#
#     coefficients = np.array(coefficients).reshape((len(coefficients) // 4, 4))
#
#     df = pd.DataFrame(coefficients, index=self.data.iloc[period:-period].index)
#     df.columns = ['a0', 'a1', 'b1', 'w']
#     df.fillna(method='bfill')  # sets each NaN value to closest real number value behind it
#
#     for col in df.columns:
#         self.data[f'fourier{period}{col}'] = df[col]
#
# def sine_fitting(self, periods=(5, 6), method='difference'):
# # Compute the coefficients of the series
# detrended = self.detrend(self.data, method=method)
#
# for period in periods:
#     coefficients = []
#     for i in range(period, len(self.data) - period):
#         x = np.arange(0, period)
#         y = detrended.iloc[i - period:i]  # shifter
#
#         # Ignores errors from optimizer when it can't fit to curve
#         with warnings.catch_warnings():
#             warnings.simplefilter('error', OptimizeWarning)
#
#             try:
#                 result = scipy.optimize.curve_fit(self.sine_series, x, y)
#             except (RuntimeError, OptimizeWarning) as e:
#                 result = np.empty((1, 3))
#                 result[0, :] = np.NAN
#
#         coefficients = np.append(coefficients, result[0], axis=0)
#
#     warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
#
#     coefficients = np.array(coefficients).reshape((len(coefficients) // 3, 3))
#
#     df = pd.DataFrame(coefficients, index=self.data.iloc[period:-period].index)
#     df.columns = ['a0', 'b1', 'w']
#     df.fillna(method='bfill')  # sets each NaN value to closest real number value behind it
#
#     for col in df.columns:
#         self.data[f'sine{period}{col}'] = df[col]


def detrend(prices, method='difference'):
    """
    Detrends data to allow for Fourier/Sine fitting
    :param prices: datafrake of OHLC currency data
    :param method: method of detrending - 'linear' or 'difference'
    :return: detrended price series
    """
    detrended = None
    if method == 'difference':
        # '1' is now, '0' is yesterday; subtracts yesterday's value from each day's value
        detrended = prices.close[1:] - prices.close[:-1].values
    elif method == 'linear':
        x = np.arange(0, len(prices))  # range of numbers from 0 to len(prices)
        y = prices.close.values
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y.reshape(-1, 1))  # Fits lin. regression line to values

        trend = model.predict(x.reshape(-1, 1))
        trend = trend.reshape((len(prices),))

        detrended = prices.close - trend

    return detrended


def fourier_series(x, a0, a1, b1, w):
    """
    Fourier Series expansion fitting function
    :param x: the hours (independent variable)
    :param a0: first fourier series coefficient
    :param a1: second fourier series coefficient
    :param b1: third fourier series coefficient
    :param w: fourier series frequency
    :return: the value of the fourier function
    """
    # F = a0 + a1*cos(wx) + b1*sin(wx)
    f = a0 + a1*np.cos(w*x) + b1*np.sin(w*x)
    return f


def sine_series(x, a0, b1, w):
    """
    Sine Series expansion fitting function
    :param x: the hours (independent variable)
    :param a0: first sine series coefficient
    :param b1: second sine series coefficient
    :param w: sine series frequency
    :return: the value of the sine function
    """
    s = a0 + b1*np.sin(w*x)
    return s


def fourier_fitting(prices, periods, method='difference'):
    """

    :param prices: OHLC dataframe
    :param periods: list of periods for which to compute coefficients - [3, 5, 10...]
    :param method: method by which to detrend price data
    :return: dict of dataframes containing coefficients for input periods
    """
    results = Holder()
    coefficient_dataframes = {}

    # Option to plot and demonstrate expansion fit for each iteration
    plot = False

    # Compute the coefficients of the series
    detrended = detrend(prices, method=method)

    for period in periods:
        coefficients = []
        for i in range(period, len(prices)-period):
            x = np.arange(0, period)
            y = detrended.iloc[i-period:i]  # shifter

            # Ignores errors from optimizer when it can't fit to curve
            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)

                try:
                    result = scipy.optimize.curve_fit(fourier_series, x, y)
                except (RuntimeError, OptimizeWarning) as e:
                    result = np.empty((1,4))  # set each of 4 params to NaN
                    result[0,:] = np.NAN

            if plot:
                xt = np.linspace(0, period, 100)  # 100 data points b/w 0 and period
                yt = fourier_series(xt, result[0][0], result[0][1], result[0][2], result[0][3])

                plt.plot(x, y)
                plt.plot(xt, yt, 'r')

                plt.show()

            coefficients = np.append(coefficients, result[0], axis=0)

        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        coefficients = np.array(coefficients).reshape((len(coefficients) // 4, 4))

        df = pd.DataFrame(coefficients, index=prices.iloc[period:-period].index)
        df.columns = ['a0', 'a1', 'b1', 'w']
        df.fillna(method='bfill')  # sets each NaN value to closest real number value behind it

        coefficient_dataframes[period] = df

    results.coefficients = coefficient_dataframes

    return results


def sine_fitting(prices, periods, method='difference'):
    """

    :param prices: OHLC dataframe
    :param periods: list of periods for which to compute coefficients - [3, 5, 10...]
    :param method: method by which to detrend price data
    :return: dict of dataframes containing coefficients for input periods
    """
    results = Holder()
    coefficient_dataframes = {}

    # Option to plot and demonstrate expansion fit for each iteration
    plot = False

    # Compute the coefficients of the series
    detrended = detrend(prices, method=method)

    for period in periods:
        coefficients = []
        for i in range(period, len(prices) - period):
            x = np.arange(0, period)
            y = detrended.iloc[i - period:i]  # shifter

            # Ignores errors from optimizer when it can't fit to curve
            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)

                try:
                    result = scipy.optimize.curve_fit(sine_series, x, y)
                except (RuntimeError, OptimizeWarning) as e:
                    result = np.empty((1, 3))
                    result[0, :] = np.NAN

            if plot:
                xt = np.linspace(0, period, 100)  # 100 data points b/w 0 and period
                yt = sine_series(xt, result[0][0], result[0][1], result[0][2])

                plt.plot(x, y)
                plt.plot(xt, yt, 'r')

                plt.show()

            coefficients = np.append(coefficients, result[0], axis=0)

        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        coefficients = np.array(coefficients).reshape((len(coefficients) // 3, 3))

        df = pd.DataFrame(coefficients, index=prices.iloc[period:-period].index)
        df.columns = ['a0', 'b1', 'w']
        df.fillna(method='bfill')  # sets each NaN value to closest real number value behind it

        coefficient_dataframes[period] = df

    results.coefficients = coefficient_dataframes

    return results


# Williams Accumulation Distribution Function
def wadl(prices, periods):
    """

    :param prices: dataframe of OHLC prices
    :param periods: list of periods for which to calculate the function
    :return: WAD indicator lines for each period
    """
    results = Holder()
    wadl_dict = {}

    for period in periods:
        WAD = []

        for i in range(period, len(prices)-period):
            curr_close = prices.close.iloc[i]
            prev_close = prices.close.iloc[i - 1]

            # Calculate True Range High/Low
            TRH = max([prices.high.iloc[i], prev_close]) # True Range High
            TRL = min([prices.low.iloc[i], prev_close])  # True Range Low

            # Calculate Price Move
            PM = None
            if curr_close > prev_close:
                PM = curr_close - TRL
            elif curr_close < prev_close:
                PM = curr_close - TRH
            elif curr_close == prev_close:
                PM = 0

            # Calculate Accumulation Distribution
            AD = PM * prices.volume.iloc[i]

            WAD = np.append(WAD, AD)

        WAD = WAD.cumsum()  # Adds previous value to current value
        WAD = pd.DataFrame(WAD, index=prices.iloc[period:-period].index)
        WAD.columns = ['close']

        wadl_dict[period] = WAD

    results.wadl = wadl_dict

    return results


# Data Resampling
def OHLCresample(df, timeframe):
    """

    :param df: DataFrame with data to resample
    :param timeframe: timeframe for resampling
    :return: resampled OHLC data for given timeframe
    """
    columns = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    resampled = df.resample(timeframe).agg(columns)
    resampled = resampled.dropna()

    return resampled


def momentum(prices, periods):
    """
    :param prices:
    :param periods:
    :return: momentum indicator values
    """
    results = Holder()
    open = {}
    close = {}

    for period in periods:
        open[period] = pd.DataFrame(prices.open.iloc[period:]-prices.open.iloc[:-period].values,
                                    index=prices.iloc[period:].index)
        close[period] = pd.DataFrame(prices.close.iloc[period:] - prices.close.iloc[:-period].values,
                                    index=prices.iloc[period:].index)
        open[period].columns = ['open']
        close[period].columns = ['close']

    results.open = open
    results.close = close

    return results

def stochastic(prices, periods):
    """

    :param prices:
    :param periods:
    :return:
    """
    results = Holder()
    close_dict = {}

    for period in periods:
        k_values = []
        for i in range(period, len(prices)-period):
            close = prices.close.iloc[i+1]
            high = max(prices.high.iloc[i-period:i])
            low = min(prices.low.iloc[i-period:i])

            if high == low:
                k = 0
            else:
                k = 100 * (close - low)/(high - low)
            k_values = np.append(k_values, k)

        df = pd.DataFrame(k_values, index=prices.iloc[period+1:-period+1].index)
        df.columns = ['K']
        df['D'] = df.K.rolling(3).mean()
        df = df.dropna()

        close_dict[period] = df

    results.close = close_dict

    return results


# Williams %R
def williams(prices, periods):
    results = Holder()
    close_dict = {}

    for period in periods:
        r_values = []
        for i in range(period, len(prices) - period):
            close = prices.close.iloc[i + 1]
            high = max(prices.high.iloc[i - period:i])
            low = min(prices.low.iloc[i - period:i])

            if high == low:
                r = 0
            else:
                r = -100 * (high - close) / (high - low)
            r_values = np.append(r_values, r)

        df = pd.DataFrame(r_values, index=prices.iloc[period + 1:-period + 1].index)
        df.columns = ['R']
        df = df.dropna()

        close_dict[period] = df

    results.close = close_dict

    return results


# PROC (Price Rate of Change)
def proc(prices, periods):
    """

    :param prices:
    :param periods:
    :return:
    """
    results = Holder()
    proc = {}

    for period in periods:
        close_today = prices.close.iloc[period:]
        close_yesterday = prices.close.iloc[:-period]

        proc[period] = pd.DataFrame(close_today-close_yesterday.values / close_yesterday.values)
        proc[period].columns = ['close']

    results.proc = proc

    return results


# Accumulation Distribution Oscillator
def adosc(prices, periods):
    """

    :param prices:
    :param periods:
    :return:
    """
    results = Holder()
    accum_dists = {}
    for period in periods:
        AD = []
        for i in range(period, len(prices)-period):
            close = prices.close.iloc[i + 1]
            high = max(prices.high.iloc[i - period:i])
            low = min(prices.low.iloc[i - period:i])
            volume = prices.volume.iloc[i + 1]

            if high == low:
                CLV = 0
            else:
                CLV = ((close - low) - (high - close)) / (high - low)

            AD = np.append(AD, CLV*volume)

        AD = AD.cumsum()
        AD = pd.DataFrame(AD, index=prices.iloc[period+1:-period+1].index)
        AD.columns = ['AD']

        accum_dists[period] = AD

    results.AD = accum_dists

    return results

# MACD
def macd(prices, periods):
    """

    :param prices:
    :param periods:
    :return:
    """
    results = Holder()

    ema_1 = prices.close.ewm(span=periods[0]).mean()
    ema_2 = prices.close.ewm(span=periods[1]).mean()

    MACD = pd.DataFrame(ema_1 - ema_2)
    MACD.columns = ['L']

    MACD_signal = MACD.rolling(3).mean()
    MACD_signal.columns = ['SL']

    results.line = MACD
    results.signal = MACD_signal

    return results

# CCI (Commodity Channel Index)
def cci(prices, periods):
    """

    :param prices:
    :param periods:
    :return:
    """
    results = Holder()
    CCI = {}

    for period in periods:
        moving_avg = prices.close.rolling(period).mean()
        standard_deviation = prices.close.rolling(period).std()

        D = (prices.close - moving_avg) / standard_deviation

        CCI[period] = pd.DataFrame((prices.close - moving_avg) / (0.015 * D))
        CCI[period].columns = ['close']

    results.cci = CCI

    return results


# Bollinger Bands
def bollinger(prices, periods, deviations):
    """

    :param prices:
    :param periods:
    :param deviations:
    :return:
    """
    results = Holder()
    bollinger_bands = {}
    for period in periods:
        middle_moving_avg = prices.close.rolling(period).mean()
        standard_deviation = prices.close.rolling(period).std()
        upper = middle_moving_avg + deviations * standard_deviation
        lower = middle_moving_avg - deviations * standard_deviation

        df = pd.concat((upper, middle_moving_avg, lower), axis=1)
        df.columns = ['upper', 'mid', 'lower']
        bollinger_bands[period] = df

    results.bands = bollinger_bands

    return results


# Price Averages
def paverage(prices, periods):
    """
    :param prices:
    :param periods:
    :return:
    """
    results = Holder()
    averages = {}
    for period in periods:
        averages[period] = pd.DataFrame(prices[['open', 'high', 'low', 'close']].rolling(period).mean())

    results.price_averages = averages

    return results


# Slope (for fitting lin. reg. line to data and get its slope)
def slope(prices, periods):
    """
    :param prices:
    :param periods:
    :return:
    """
    results = Holder()
    slope_dict = {}

    for period in periods:

        slopes = []

        for i in range(period, len(prices) - period):
            y = prices.high.iloc[i - period:i].values
            x = np.arange(0, len(y))

            lin_regression = stats.linregress(x, y=y)
            m = lin_regression.slope

            slopes = np.append(slopes, m)

        slopes = pd.DataFrame(slopes, index=prices.iloc[period:-period].index)
        if len(slopes.columns) == 1:
            slopes.columns = ['high']
        slope_dict[period] = slopes

    results.slope = slope_dict

    return results
