import datetime
import sys
import time

import backtrader
import backtrader as bt
import joblib
import pandas as pd
import ta


def first_friday(year, month):
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
        ATR = range_total / self.params.get_atr
        self.log(f'Open: {self.data_open[0]:.4f}, High: {self.data_high[0]:.4f}, '
                 f'Low: {self.data_low[0]:.4f}, Close: {self.data_close[0]:.4f}, ATR: {ATR:.4f}')


class Trend(Base):
    # DEFAULT STARTING VALS
    params = (('atr', 14), ('atr_sl', 1.25), ('atr_tp', 2),
              ('sma', 20), ('ema', 20), ('vidya', 20),  # Baseline?
              ('confirmation_1', 'didi'), ('confirmation_2', ''),
              ('didi_mid', 8), ('didi_long', 25),
              ('win_mult', 1), ('lose_mult', 1), ('win_streak_limit', 1), ('lose_streak_limit', 1),
              ('macd_me1', 12), ('macd_me2', 26), ('macd_signal', 9),
              ('rsi', 14), ('adx', 14), ('adx_cutoff', 30))
    base_name = 'Trend'
    full_name = ''

    def __init__(self, params=None, logging=True):
        # Instantiate moving averages
        super().__init__()
        self.logging = logging
        self.full_name += self.base_name

        self.units = self.broker.get_cash() / 5
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

        self.sma = bt.indicators.SMA(self.datas[0], period=self.params.sma)
        self.ema = bt.indicators.ExponentialMovingAverage(self.datas[0], period=self.params.ema)
        self.macd = bt.indicators.MACD(self.datas[0], period_me1=self.params.macd_me1, period_me2=self.params.macd_me2,
                                       period_signal=self.params.macd_signal)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(self.datas[0], period=self.params.adx)
        self.DIplus = bt.indicators.PlusDirectionalIndicator(self.datas[0], period=self.params.adx)
        self.DIminus = bt.indicators.MinusDirectionalIndicator(self.datas[0], period=self.params.adx)
        self.rsi = bt.indicators.RSI_EMA(self.datas[0], period=self.params.rsi)
        self.prev_didi = 0

    # def adx(self):
    def get_atr(self, period):
        period_total = 0
        for i in range(-(period - 1), 1):
            true_range = self.data_high[i] - self.data_low[i]
            period_total += true_range
        atr = period_total / period

        return atr

    def get_sma(self, period):
        period_total = 0
        for i in range(-(period - 1), 1):
            period_total += self.data_close[i]
        # print(self.data_close[0])

        sma = period_total / period
        return sma

    def get_ema(self, period):
        # EMA calculation has to start somewhere, so use SMA w/ EMA period as origin
        inital_sma = self.get_sma(period)
        previous_emas = [inital_sma]
        weight_multiplier = (2 / (period + 1))

        for i in range(-(period - 1), 1):
            prev = previous_emas[-1]
            ema = (self.data_close[0] - prev) * weight_multiplier + prev
            previous_emas.append(ema)

        current_ema = previous_emas[-1]
        return current_ema

    # Smoothed moving average
    def get_smma(self, period):
        inital_sma = self.get_sma(period)
        previous_smmas = [inital_sma]
        weight_multiplier = (1 / (period + 1))

        for i in range(-(period - 1), 1):
            prev = previous_smmas[-1]
            smma = (self.data_close[0] - prev) * weight_multiplier + prev
            previous_smmas.append(smma)

        current_smma = previous_smmas[-1]
        return current_smma

    # Chande Momentum Oscillator
    def get_cmo(self, period, bars_ago=0):
        sum_up, sum_down = 0, 0
        # For VIDYA, to get CMO at each specific bar
        start, end = -(period - 1) - bars_ago, 1 - bars_ago
        for i in range(start, end):
            prev_close = self.data_close[i - 1]
            close = self.data_close[i]
            if close > prev_close:
                sum_up += (close - prev_close)
            else:
                sum_down += abs(close - prev_close)

        cmo = 100 * ((sum_up - sum_down) / (sum_up + sum_down))
        return cmo

    # Pricei x F x ABS(CMOi) + VIDYAi-1 x (1 - F x ABS(CMOi))
    # Chande's Variable Index Dynamic Average
    def get_vidya(self, period):
        f = 2 / (self.get_ema(period) + 1)  # Smoothing factor
        prev_vidyas = [self.get_sma(period)]  # SMA or EMA?
        for i in range(-(period - 1), 1):
            prev = prev_vidyas[-1]
            close = self.data_close[i]
            cmo = self.get_cmo(period, bars_ago=i)  # Current CMO at this bar
            vidya = close * f * abs(cmo) + prev * (1 - f * abs(cmo))
            prev_vidyas.append(vidya)

        current_vidya = prev_vidyas[-1]
        return current_vidya

    def get_adx(self, period):
        DMpluses, DMminuses = [], []
        for i in range(-(period - 1), 1):
            upmove = self.data_high[0] - self.data_high[-1]
            downmove = self.data_low[-1] - self.data_low[0]
            DMplus = upmove if (upmove > downmove) and upmove > 0 else 0
            DMminus = downmove if (downmove > downmove) and downmove > 0 else 0

            DMpluses.append(DMplus)
            DMminuses.append(DMminus)
            DMplus_smma = self.get_smma_of_values(period, DMpluses)
            DMminus_smma = self.get_smma_of_values(period, DMminuses)

            atr = self.get_atr(period)
            # DIplus = 100 * DMplus_smma / atr
            # DIminus = 100 * DMminus_smma / atr

    def get_didi(self):
        # NOTE: Didi crossing BELOW ZERO is a LONG SIGNAL, and vice versa
        ma_mid = self.get_sma(self.params.didi_mid)
        ma_long = self.get_sma(self.params.didi_long)
        didi_long = (ma_long / ma_mid - 1) * 100

        return didi_long

    def update_streak(self, streak):
        self.curr_streak[streak] += 1
        if streak == 'loss':
            self.curr_streak['win'] = 0
        elif streak == 'win':
            self.curr_streak['loss'] = 0

    def baseline(self):
        # baseline = self.get_ema(self.params.ema)
        # baseline = self.get_sma(self.params.sma)
        baseline = self.get_vidya(self.params.vidya)
        close = self.data_close[0]

        # Check for signal in past 3 candles
        recent_signals = []
        for i in range(-2, 1):
            recent_candle = self.data_close[i]
            if recent_candle < baseline < close:  # Cross above
                recent_signals.append("LONG")
            elif recent_candle > baseline > close:  # Cross below
                recent_signals.append("SHORT")
            else:
                recent_signals.append("NONE")

        buy, sell = False, False
        if "LONG" in recent_signals and "SHORT" not in recent_signals:
            buy = True,
            sell = False
        elif "SHORT" in recent_signals and "LONG" not in recent_signals:
            sell = True
            buy = False

        if close > baseline:
            price_where = "PRICE_ABOVE"
        else:
            price_where = "PRICE_BELOW"

        return buy, sell, price_where

    def buy_sell_signal(self, indicator):
        if indicator == "rsi":
            # RSI
            upperband = self.rsi.params.upperband
            lowerband = self.rsi.params.lowerband
            rsi_buy_signal = self.rsi[-1] > upperband > self.rsi[0]  # Check if went below overbought
            rsi_sell_signal = self.rsi[-1] < lowerband < self.rsi[0]  # Check if went above oversold

            return rsi_buy_signal, rsi_sell_signal

    def confirmation_1(self, indicator):
        buy, sell, trend = False, False, "NONE"
        if indicator == "didi":
            didi = self.get_didi()
            if didi > 0:
                trend = "DOWNTREND"
            elif didi < 0:
                trend = "UPTREND"
            else:
                trend = "NONE"

            if didi > 0 and self.prev_didi < 0:
                sell = True
                buy = False
            elif didi < 0 and self.prev_didi > 0:
                buy = True
                sell = False

        return buy, sell, trend

    def can_trade(self):
        # Don't do anything if an order is already open
        if self.order:
            return False

        date = self.data_datetime.date(0)

        first_friday_of_month = first_friday(date.year, date.month)

        # Don't trade on first firday of month (Non-Farm Payroll)
        if date.day == first_friday_of_month.day:
            return False

        curr_hour = self.data_datetime.time(0).hour

        # Outside trading hours
        # if not (3 < curr_hour < 12):
        #     if self.position:
        #         self.log(f'CLOSING OUT DAY {self.data_close[0]:2f}')
        #         self.order = self.close()
        #     return

        # Trend not strong, don't enter
        if not self.adx[0] > self.params.adx_cutoff:
            return False

        return True

    def check_stop_take(self):
        close = self.data_close[0]
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

        if stop or take:
            return True

    # Iterates through bars
    def next(self):

        # See if trade conditions are all met
        if not self.can_trade():
            return

        # Variables
        prediction = 0
        uptrend, downtrend = False, False
        close = self.data_close[0]

        # Get indicators
        ATR = self.get_atr(self.params.atr)
        BASELINE_BUY, BASELINE_SELL, BASELINE_WHERE = self.baseline()
        C1_BUY, C1_SELL, C1_TREND = self.confirmation_1(self.params.confirmation_1)
        C2_BUY, C2_SELL, C2_TREND = C1_BUY, C1_SELL, C1_TREND  # TODO

        # CONFIRMATION INDICATORS (check if in up/down trend for long/short signals)
        if C1_TREND == "UPTREND" and C2_TREND == "UPTREND" and BASELINE_WHERE == "PRICE_ABOVE":
            uptrend = True
            downtrend = False
        elif C1_TREND == "DOWNTREND" and C2_TREND == "DOWNTREND" and BASELINE_WHERE == "PRICE_BELOW":
            downtrend = True
            uptrend = False

        # CHECK FOR SIGNALS
        BUY_SIGNAL, SELL_SIGNAL = self.buy_sell_signal('rsi')
        if (BUY_SIGNAL or BASELINE_BUY or C1_BUY) and uptrend:
            prediction = 1
        elif (SELL_SIGNAL or BASELINE_SELL or C1_SELL) and downtrend:
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
            # We're in the market, check for SL/TP hit and act accordingly
            exited = self.check_stop_take()
            if exited:
                return

            # Check if price crosses baseline unfavorably
            if self.position.size > 0 and BASELINE_WHERE == "PRICE_BELOW":
                self.order = self.close()
            elif self.position.size < 0 and BASELINE_WHERE == "PRICE_ABOVE":
                self.order = self.close()

    # Calls after backtest completes
    def stop(self):
        pass
