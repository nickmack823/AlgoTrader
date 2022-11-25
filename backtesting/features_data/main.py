from os.path import exists
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from indicator_getters.feature_collector import create_features_file
from indicator_getters.indicator_feature_functions import *
import plotly as py
from plotly import subplots
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression

# (1) Load data
features_path = f'EURUSD_1hr_2022_features.csv'
prices = pd.read_csv("../backtesting/forex_data/EURUSD_1hr_2022.csv")
# Rename gmt time to date
prices.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
# Default DT is M-D-Y, Dukoscopy is D-M-Y
prices.date = pd.to_datetime(prices.date, format='%d.%m.%Y %H:%M:%S.%f')
prices = prices.set_index(prices.date)
prices = prices[['open', 'high', 'low', 'close', 'volume']]
# Drop downtime data
prices = prices.drop_duplicates(keep=False)
# prices = raw_data.iloc[:200]


def plot():
    moving_avg = prices.close.rolling(center=False, window=30).mean()  # 30: 30hr avg

    # (2) Get indicator values from selected function(s)
    # HA_results = heiken_ashi(df, [1])
    # HA = HA_results.candles[1]  # df of candles
    # detrended = detrend(df, method='difference')
    # fourier = fourier_parameter_fitting(df, [10, 15], method='difference')
    # WADL = wadl(df, [15])
    # wadl_line = WADL.wadl[15]
    # resampled = OHLCresample(df, '15H')
    # resampled.index = resampled.index.droplevel(0)

    # (3) Plot
    trace0 = go.Ohlc(x=prices.index, open=prices.open, high=prices.high,
                     low=prices.low, close=prices.close, name='Price')  # Price trace
    trace1 = go.Scatter(x=prices.index, y=moving_avg, name="MA 30hr")  # MA trace

    # trace2 = go.Bar(x=df.index, y=df.volume, name='Volume') # Volume trace
    # trace2 = go.Ohlc(x=HA.index, open=HA.open, high=HA.high, low=HA.low, close=HA.close, name='Heiken Ashi')
    # trace2 = go.Scatter(x=df.index, y=detrended, name="Fourier)
    # trace2 = go.Scatter(x=wadl_line.index, y=wadl_line.close, name="WAD")
    # trace2 = go.Ohlc(x=resampled.index.to_pydatetime(), open=resampled.open, high=resampled.high, low=resampled.low,
    #                  close = resampled.close, name='Resampled')

    data = [trace0, trace1]

    fig = subplots.make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(trace0, 1, 1)
    fig.add_trace(trace1, 1, 1)
    # fig.add_trace(trace2, 2, 1)

    py.offline.plot(fig, filename=csv_name)


if not exists(features_path):
    create_features_file(prices, features_path)
features = pd.read_csv(features_path)
print(features.head())

X_train, X_test, y_train, y_test = train_test_split(features[['open']],
                                                    features[['close']], test_size=0.2, shuffle=False)
# print(X_test.describe())
#
# print(X_train.describe())
# print(X_train.head())
print(X_test.columns)

model = LinearRegression()
# # Uses H, L, O to train with in order to predict close
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f'Model Coefficients: {model.coef_}')
print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}')
print(f'Coefficient of Determiniation: {r2_score(y_test, y_pred)}')

trades = []
for i in range(0, len(features[['open']].iloc[1:10])):
    open_price = features[['open']].iloc[i][0]
    prediction = model.predict([[open_price]])[0][0]
    actual_close = features[['close']].iloc[i][0]

    lot_size = 0.1
    pips = (actual_close - open_price) * 10000
    print(prediction)
    print(open_price)
    if prediction > open_price:
        profit = (100000 * lot_size) * pips
        trade = {'buy': open_price,
                 'tp': prediction,
                 'stop': prediction - open_price,
                 'p/l': profit}
        trades.append(trade)
    elif prediction < open_price:
        profit = (100000 * lot_size) * (pips * -1)
        trade = {'buy': open_price,
                 'tp': prediction,
                 'stop': prediction - open_price,
                 'p/l': profit}
        trades.append(trade)

for trade in trades:
    print(trade)