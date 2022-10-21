from datetime import datetime, timedelta

from IterativeBase import *


class IterativeBacktest(IterativeBase):
    ''' Class for iterative (bar-by-bar) backtesting of trading strategies.'''

    def close_position(self, close):

        price_traded_at = round(self.data.close.iloc[self.position_opened_bar], 5)
        lot_size = (self.units / 100000)

        if self.position == 1:
            diff = (close - price_traded_at)
            profit = diff * lot_size * 100000
        else:
            diff = (price_traded_at - close)
            profit = diff * lot_size * 100000

        self.current_balance += profit
        self.update_units()
        self.position = 0
        self.trades += 1

    def buy(self, bar):
        date, close, spread = self.get_values(bar)
        if self.use_spread:
            close += spread / 2  # ask price
        if self.position == -1:
            self.close_position(close)  # close current short before opening a buy

        self.position_opened_bar = bar
        self.position = 1
        self.trades += 1
        # print("{} |  Buying {} for {}".format(date, self.units, round(close, 5)))

    def sell(self, bar):
        date, close, spread = self.get_values(bar)
        if self.use_spread:
            close -= spread / 2  # bid price
        if self.position == 1:
            self.close_position(close)

        self.position_opened_bar = bar
        self.position = -1
        self.trades += 1
        # print("{} |  Selling {} for {}".format(date, self.units, round(close, 5)))

    def update_units(self):
        self.units = round(self.current_balance * 10)

    def set_stop_loss(self, bar, trailing=False):
        atr = self.data['ATRr_14'].iloc[bar]
        price = self.data['close'].iloc[bar]
        # Initial stop loss
        if not trailing:
            if self.position == 1:
                self.stop_loss = price - (atr * 2)
            elif self.position == -1:
                self.stop_loss = price + (atr * 2)
        # Trailing stop loss updates if price moves favorably
        else:
            price_back_two_bars = self.data['close'].iloc[-2]
            if self.position == 1 and price > price_back_two_bars:
                self.stop_loss = price - (atr * 2)
                print(f'Moving stop loss to {self.stop_loss} | Position == {self.position}')
            elif self.position == -1 and price < price_back_two_bars:
                self.stop_loss = price + (atr * 2)
                print(f'Moving stop loss to {self.stop_loss} | Position == {self.position}')

    def check_price_triggers(self, bar):
        date, close, spread = self.get_values(bar)
        # Check for stop loss trigger
        if self.stop_loss is not None:
            if self.position == 1 and close <= self.stop_loss:
                self.close_position(close)
            elif self.position == -1 and close >= self.stop_loss:
                self.close_position(close)
        # Check for take profit trigger
        if self.take_profit is not None:
            if self.position == 1 and close >= self.take_profit:
                self.close_position(close)
            elif self.position == -1 and close <= self.take_profit:
                self.close_position(close)

    def check_strategy(self, bar):
        if self.strategy == 'model':
            self.prediction = self.model.predict(self.data[self.model_features].iloc[bar].to_frame().T)[0]
        elif self.strategy == 'macd_scalp':
            print(self.data.index)
            price = self.data['close'].iloc[-1]
            ema = self.data['EMA_100'].iloc[-1]
            if price > ema:
                trend = 'up'
            else:
                trend = 'down'

            macd_back_one = self.data['MACD_15_30_9'].iloc[-2]
            signal_back_one = self.data['MACDs_15_30_9'].iloc[-2]
            macd = self.data['MACD_15_30_9'].iloc[-1]
            signal = self.data['MACDs_15_30_9'].iloc[-1]

            # Upcross, downcross
            if trend == 'up' and macd_back_one < signal_back_one and macd > signal:
                self.prediction = 1
            elif trend == 'down' and macd_back_one > signal_back_one and macd < signal:
                self.prediction = -1

    def test_strategy(self, features):
        # reset
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.current_balance = self.initial_balance  # reset initial capital
        self.get_data()  # reset dataset
        lines = []

        # sma crossover strategy
        for bar in range(len(self.data) - 1):  # all bars (except the last bar)
            date, close, spread = self.get_values(bar)

            # Stay within selected trading hours
            date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
            if date.hour < 3 or date.hour > 12:
                continue

            self.check_strategy(bar)  # Get prediction for current bar

            if self.prediction == 1:  # signal to go long
                if self.position in [0, -1]:
                    self.buy(bar)
                    atr = self.data['ATRr_14'].iloc[bar]
                    self.take_profit = close + (atr * 3)
                    self.set_stop_loss(bar, trailing=False)
            elif self.prediction == -1:  # signal to go short
                if self.position in [0, 1]:
                    self.sell(bar)
                    atr = self.data['ATRr_14'].iloc[bar]
                    self.take_profit = close + (atr * 3)
                    self.set_stop_loss(bar, trailing=False)
            elif self.prediction == 0:
                self.close_position(close)

            line = f"({date}) Balance: {self.current_balance} | Position: {self.position} " \
                   f"| Trades: {self.trades}\n"
            lines.append(line)

        self.close_position(self.get_values(bar+1)[1])

        line = f"FINAL BALANCE: {self.current_balance} | NET PROFIT: {self.current_balance - self.initial_balance} " \
               f"| TOTAL TRADES: {self.trades} {self.position}\n"
        lines.append(line)

        return lines


if __name__ == "__main__":
    yesterday = datetime.now() - timedelta(days=1)
    yesterday = datetime.strftime(yesterday, '%Y-%m-%d')
    symbol, start, end, timeframe, strategy = 'EUR_USD', '2021-10-01', '2022-10-01', 'M30', 'model'
    ib = IterativeBacktest(symbol, start, end, timeframe, 1000, strategy, use_spread=True)
    results = ib.test_strategy(ib.model_features)

    with open(os.path.join(ib.dirname, f"tests/{symbol}-{start} to {end}-{timeframe}-{strategy}.txt"), "w") as f:
        f.writelines(results)

    # TODO: Compare actual demo account performance to backtester results to ensure validity

