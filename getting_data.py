import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import exists

plt.style.use('seaborn')


def inspect(dataframe):
    print(dataframe.head())
    print(dataframe.tail())
    print(dataframe.describe())


def visualize(dataframe):
    dataframe.plot(figsize=(15, 8), fontsize=13)
    plt.legend(fontsize=13)
    plt.show()


if not exists('stocks.csv'):
    tickers = ['AAPL', 'BA', 'KO', 'IBM', 'DIS', 'MSFT']
    data = yf.download(tickers, start='2010-01-01', end='2022-06-17')
    data.to_csv("stocks.csv")

stocks = pd.read_csv('stocks.csv', header=[0, 1], index_col=[0], parse_dates=[0])
# stocks.columns = stocks.columns.to_flat_index()  # Sends from multi index to single index
# stocks.columns = pd.MultiIndex.from_tuples(stocks.columns)  # Back to multi-index
# stocks = stocks.swaplevel(axis=1).sort_index(axis=1)

inspect(stocks)

# Daily closing prices
close = stocks.loc[:, 'Close'].copy()
# visualize(close)

# Normalizing Time Series to a Base Value (100)
print(close.head())
aapl_norm = close.AAPL.div(close.iloc[0, 0]).mul(100)  # Divides each AAPL price by AAPL's starting price, then mult by 100
print(aapl_norm.head())

close_norm = close.div(close.iloc[0]).mul(100)  # Does same for each element in first row and their columns
print(close_norm.head())
# visualize(close_norm)

# shift() method
aapl = close.AAPL.copy().to_frame()
aapl['lag1'] = aapl.shift(periods=1)  # Shifts index by one period
aapl['Diff_shift()'] = aapl.AAPL.sub(aapl.lag1)  # Difference b/w each day and the next
aapl['%_change'] = aapl.AAPL.div(aapl.lag1).sub(1).mul(100)  # % change difference

# diff() and pct_change()
aapl['Diff_diff()'] = aapl.AAPL.diff(periods=1)  # Gets diff from 1 day (period) to another day
aapl['%_change_pct_change()'] = aapl.AAPL.pct_change(periods=1).mul(100)
print(aapl.head())
# Resamples data to get price on last business day of month, then finds pct_change from each month to next
month_resample = aapl.AAPL.resample('BM').last().pct_change(periods=1).mul(100)
print(month_resample.head())

# Measuring performance with MEAN Return and STD of Returns
aapl = close.AAPL.copy().to_frame()
returns = aapl.pct_change().dropna()
print(returns.head())
# returns.plot(kind='hist', figsize=(12, 8), bins=100)  # Plots histogram of % daily returns
# plt.show()

daily_mean_return = returns.mean()
variance_daily_returns = returns.var()
# std_daily_returns = np.sqrt(variance_daily_returns)
std_daily_returns = returns.std()
print(f'Daily Mean: {daily_mean_return}\nDaily Variance: {variance_daily_returns}\nDaily STD: {std_daily_returns}')
annual_mean_return = daily_mean_return * 252  # 252 trading days in year
annual_variance_return = variance_daily_returns * 252
annual_std_return = np.sqrt(annual_variance_return)
print(f'Annual Mean: {daily_mean_return}\nAnnual Variance: {variance_daily_returns}\nAnnual STD: {std_daily_returns}')

# Return and Risk
returns = close.pct_change().dropna()
inspect(returns)
summary = returns.describe().T.loc[:, ['mean', 'std']] # Transpose row w/ col, then get only mean and std
print(summary.head())
summary['mean'] = summary['mean'] * 252  # Annualize
summary['std'] = summary['std'] * 252
print(summary.head())

# summary.plot.scatter(x='std', y='mean', figsize=(12, 8), s=50, fontsize=15)
# for i in summary.index:  # Annotate dots
#     plt.annotate(i, xy=(summary.loc[i, 'std'] + 0.002, summary.loc[i, 'mean'] + 0.002), size=15)
# plt.xlabel('Annual Risk (STD)', fontsize=15)
# plt.ylabel('Annual Return', fontsize=15)
# plt.title('Risk/Return', fontsize=20)
# plt.show()

# Covariance and Correlation
covariance = returns.cov()
correlation = returns.corr()
print(covariance.head())
print(correlation.head())
# plt.figure(figsize=(12, 8))
# sns.set(font_scale=1.4)
# sns.heatmap(correlation, cmap='Reds', annot=True, annot_kws={"size": 15}, vmax=0.6)
# plt.show()

# Simple Returns vs. Logarithmic Returns
example = pd.DataFrame(index=[2016, 2017, 2018], data=[100, 50, 95], columns=['Price'])
print(example.head())
simple_returns = example.pct_change().dropna()
print(simple_returns.head())
# Simple returns expects 100 * s_r * s_r price value, but this is way off
log_returns = np.log(example / example.shift(1)).dropna()
print(log_returns.head())
expected_log_return = 100 * np.exp(2 * log_returns.mean())  # 2 = num of periods
print(expected_log_return)
