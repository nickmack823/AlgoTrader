import datetime
import sys

import backtrader
import backtrader as bt
import joblib
import pandas as pd
import ta


class Base(bt.Strategy):

    def __init__(self):
        # Base data columns
        self.data_close = self.datas[0].close
        self.data_open = self.datas[0].open
        self.data_high = self.datas[0].high
        self.data_low = self.datas[0].low
        self.data_datetime = self.datas[0].datetime

        # Use [0] (i.e. self.data_close[0]) to get CURRENT, [-1] to get 1 bar back ([-2], [-3], etc...)

        self.order = None
        self.bar_executed = None
        self.tp, self.sl = None, None

    def log(self, txt):
        if self.logging:
            dt = str(self.datas[0].datetime.date(0)) + ' ' + str(self.datas[0].datetime.time(0))
            print(f'{dt} {txt}')  # Print date and close

    # Trade logging, called when order is made
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # An active Buy/Sell order has been submitted/accepted - Nothing to do
            return

        # Check if an order has been completed (broker could reject order if not enough cash)
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, PRICE: {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, PRICE: {order.executed.price:.2f}')
            self.bar_executed = len(self)

        elif order.status in [order.Canceled]:
            self.log('Order Canceled')
        elif order.status in [order.Margin]:
            self.log('Order Margin')
        elif order.status in [order.Rejected]:
            self.log('Order Rejected')

        # Reset orders
        self.order = None

    def log_bar(self):
        range_total = 0
        for i in range(-13, 1):
            true_range = self.data_high[i] - self.data_low[i]
            range_total += true_range
        ATR = range_total / self.params.atr
        self.log(f'Open: {self.data_open[0]:.4f}, High: {self.data_high[0]:.4f}, '
                 f'Low: {self.data_low[0]:.4f}, Close: {self.data_close[0]:.4f}, ATR: {ATR:.4f}')


class Trend(Base):
    # DEFAULT STARTING VALS
    params = (('atr', 14), ('atr_sl', 1.25), ('atr_tp', 2),
              ('proc', 15), ('paverage', 2),
              ('win_mult', 1), ('lose_mult', 1), ('win_streak_limit', 1), ('lose_streak_limit', 1),
              ('macd_me1', 12), ('macd_me2', 26), ('macd_signal', 9),
              ('rsi', 14), ('adx_period', 14), ('adx_cutoff', 30))
    base_name = 'Trend'
    full_name = ''

    def __init__(self, params=None, logging=True):
        # Instantiate moving averages
        super().__init__()
        self.logging = logging
        self.full_name += self.base_name

        self.units = self.broker.get_cash()/5
        self.inital_units = self.broker.get_cash() / 5
        self.curr_streak = {'win': 0, 'loss': 0}

        # Set parameters and name
        if params is not None:
            for param, val in params.items():
                setattr(self.params, param, val)
                self.full_name += '__' + param + '-' + str(val)
        else:
            params = dir(self.params)
            for param in list(params):
                if param[0] != '_' and param not in ['isdefault', 'notdefault']:
                    val = getattr(self.params, param)
                    self.full_name += '_' + param + '-' + str(val)

        # Write full name to temp txt file for later retrieval for quantstats title
        with open('TEMP.txt', 'w') as f:
            f.write(self.full_name)

        self.macd = bt.indicators.MACD(self.datas[0], period_me1=self.params.macd_me1, period_me2=self.params.macd_me2,
                                       period_signal=self.params.macd_signal)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(self.datas[0], period=self.params.adx_period)
        # self.DIplus = bt.indicators.PlusDirectionalIndicator(self.datas[0], period=self.params.adx_period)
        self.DIminus = bt.indicators.MinusDirectionalIndicator(self.datas[0], period=self.params.adx_period)
        self.rsi = bt.indicators.RSI_EMA(self.datas[0], period=self.params.rsi)

    # def adx(self):

    def calculate_indicators(self):
        range_total = 0
        indicator_values = {}
        for i in range(-(self.params.atr - 1), 1):
            true_range = self.data_high[i] - self.data_low[i]
            range_total += true_range
        indicator_values['atr'] = range_total / self.params.atr

        close = self.data_close[-1]
        close_n = self.data_close[-(self.params.proc - 1)]
        indicator_values['proc'] = ((close - close_n) / (close_n)) * 100

        range_total = 0
        for j in range(-(self.params.paverage - 1), 1):
            close = self.data_close[j]
            range_total += close
        indicator_values['paverage'] = range_total / self.params.paverage

        return indicator_values

    def update_streak(self, streak):
        self.curr_streak[streak] += 1
        if streak == 'loss':
            self.curr_streak['win'] = 0
        elif streak == 'win':
            self.curr_streak['loss'] = 0

    def baseline(self):
        baseline_indicator_value = 0
        # TODO: If price closes above baseli

    def confirmation(self, indicators):
        bools = []
        if 'adx' in indicators:
            bools.append(self.adx[0] > self.params.adx_cutoff)

        return bools

    def first_friday(self, year, month):
        """Return datetime.date for monthly option expiration given year and
        month
        """
        # The 15th is the lowest third day in the month
        first = datetime.date(year, month, 1)
        # What day of the week is the 1st?
        w = first.weekday()
        # Friday is weekday 4
        if w != 4:
            # Replace just the day (of month)
            first = first.replace(day=(1 + (4 - w) % 7))
        return first

    # Iterates through bars
    def next(self):
        # Don't do anything if an order is already open
        if self.order:
            return
        date = self.data_datetime.date(0)

        first_friday = self.first_friday(date.year, date.month)

        # Don't trade on first firday of month (Non-Farm Payroll)
        if date.day == first_friday.day:
            return

        curr_hour = self.data_datetime.time(0).hour

        #Outside trading hours
        # if not (3 < curr_hour < 12):
        #     if self.position:
        #         self.log(f'CLOSING OUT DAY {self.data_close[0]:2f}')
        #         self.order = self.close()
        #     return

        can_trade = all(self.confirmation(['adx', 'rsi']))
        if not can_trade:
            return

        prediction = 0

        # Get indicators
        indicator_values = self.calculate_indicators()
        ATR = indicator_values['atr']

        close = self.data_close[0]


        # uptrend = self.DIplus[0] > self.DIminus[0]
        # downtrend = self.DIplus[0] < self.DIminus[0]

        # RSI
        upperband = self.rsi.params.upperband
        lowerband = self.rsi.params.lowerband
        rsi_buy_signal = self.rsi[-1] > upperband > self.rsi[0]  # Check if went below overbought
        rsi_sell_signal = self.rsi[-1] < lowerband < self.rsi[0]  # Check if went above oversold

        if rsi_buy_signal and uptrend:
            prediction = 1
        elif rsi_sell_signal and downtrend:
            prediction = -1

        # No open position, make an order
        if not self.position:
            if prediction == 1:
                self.log(f'BUY CREATE {self.data_close[0]:2f}')
                # Keep track of the created order to avoid a 2nd order
                # self.order = self.buy(size=self.units)
                self.sl = close - (ATR * self.params.atr_sl)
                self.tp = close + (ATR * self.params.atr_tp)
                self.order = self.buy(size=self.units, price=close)
            elif prediction == -1:
                self.log(f'SELL CREATE {self.data_close[0]:2f}')
                # Keep track of the created order to avoid a 2nd order
                self.sl = close + (ATR * self.params.atr_sl)
                self.tp = close - (ATR * self.params.atr_tp)
                self.order = self.sell(size=self.units, price=close)
        else:
            # We are already in the market, look for a signal to CLOSE trades
            stop, take = False, False
            if self.position.size > 0:
                if close >= self.tp:
                    take = True
                elif close <= self.sl:
                    stop = True
            elif self.position.size < 0:
                if close <= self.tp:
                    take = True
                elif close >= self.sl:
                    stop = True
            if take:
                self.log(f'TAKE PROFIT HIT {self.data_close[0]:2f}')
                self.order = self.close()
                self.update_streak('win')
            elif stop:
                self.log(f'STOP LOSS HIT {self.data_close[0]:2f}')
                self.order = self.close()
                self.update_streak('loss')

    # Calls after backtest completes
    def stop(self):
        pass