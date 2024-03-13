import datetime
import backtrader as bt

import IndicatorGetter


# def get_exchange_rate(currency):
#     url = f"https://api.exchangerate.host/convert?from={currency}&to=USD&amount=1"
#     response = requests.get(url)
#     data = response.json()
#     rate = data["result"]
#     return rate

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


class Indicators:

    def __init__(self, data):
        self.data_close = data.close
        self.data_open = data.open
        self.data_high = data.high
        self.data_low = data.low

    def get_atr(self, period, bars_ago=0):
        period_total = 0
        start, end = -period - bars_ago - 1, 1 - bars_ago
        for i in range(start, end):
            true_range = self.data_high[i] - self.data_low[i]
            period_total += true_range
        atr = period_total / period

        return atr

    def get_sma(self, period, bars_ago=0):
        period_total = 0
        start, end = -period - bars_ago - 1, 1 - bars_ago
        for i in range(start, end):
            period_total += self.data_close[i]
        sma = period_total / period

        return sma

    def get_ema(self, period, bars_ago=0):
        # EMA calculation has to start somewhere, so use SMA w/ EMA period as origin
        inital_sma = self.get_sma(period)
        previous_emas = [inital_sma]
        weight_multiplier = (2 / (period + 1))

        start, end = -period - bars_ago - 1, 1 - bars_ago
        for i in range(start, end):
            prev = previous_emas[-1]
            ema = (self.data_close[0] - prev) * weight_multiplier + prev
            previous_emas.append(ema)

        current_ema = previous_emas[-1]
        return current_ema


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


# CURRENTLY LOOKING FOR: Baseline
class NNFX(Base):
    params = (('baseline', 'sma'),
              ('atr', 14), ('atr_sl', 1.25), ('atr_tp', 2),
              ('vidya', 20), ('accelerator_lwma', 20), ('ash', 20))

    def __init__(self, logging=False):
        super().__init__()
        self.logging = logging

        self.units = self.broker.get_cash() / 5
        self.inital_units = self.broker.get_cash() / 5
        self.curr_streak = {'win': 0, 'loss': 0}

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

        return True

    def check_baseline(self):
        baseline = IndicatorGetter.vidya()
        close = self.data_close[0]

        # Check for signal in past 3 candles
        recent_signals = []
        for i in range(-3, 0):
            recent_candle = self.data_close[i]
            one_after = self.data_close[i + 1]

            baseline = self.baseline[i + 1]
            bars_ago = abs(i + 1)

            if recent_candle < baseline < one_after:  # Cross above
                recent_signals.append("LONG")
            elif recent_candle > baseline > one_after:  # Cross below
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

    def make_prediction(self):
        # Variables
        prediction = 0

        # Get indicators
        BASELINE_BUY, BASELINE_SELL, BASELINE_WHERE = self.check_baseline()
        if BASELINE_BUY:
            prediction = 1
        elif BASELINE_SELL:
            prediction = -1

        return prediction

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

    def update_streak(self, streak):
        self.curr_streak[streak] += 1
        if streak == 'loss':
            self.curr_streak['win'] = 0
        elif streak == 'win':
            self.curr_streak['loss'] = 0

    def set_sl_tp(self, prediction):
        close = self.data_close[0]
        ATR = self.Indicators.get_atr(self.params.atr)
        if prediction == 1:
            self.sl = close - (ATR * self.params.atr_sl)
            self.tp = close + (ATR * self.params.atr_tp)
        elif prediction == -1:
            self.sl = close + (ATR * self.params.atr_sl)
            self.tp = close - (ATR * self.params.atr_tp)

    def next(self):

        # See if trade conditions are all met
        if not self.can_trade():
            return

        # No open position, make an order
        if not self.position:

            prediction = self.make_prediction()

            self.set_sl_tp(prediction)

            close = self.data_close[0]

            if prediction == 1:
                self.log(f'BUY CREATE {self.data_close[0]:2f}')
                self.order = self.buy(size=self.units, price=close)
            elif prediction == -1:
                self.log(f'SELL CREATE {self.data_close[0]:2f}')
                self.order = self.sell(size=self.units, price=close)
        else:
            # We're in the market, check for SL/TP hit and act accordingly
            exited = self.check_stop_take()
            if exited:
                return

            BASELINE_BUY, BASELINE_SELL, BASELINE_WHERE = self.check_baseline()
            # Check if price crosses baseline unfavorably
            if self.position.size > 0 and BASELINE_WHERE == "PRICE_BELOW":
                self.order = self.close()
            elif self.position.size < 0 and BASELINE_WHERE == "PRICE_ABOVE":
                self.order = self.close()

    # Calls after backtest completes
    def stop(self):
        pass


class Trend(Base):
    # DEFAULT STARTING VALS
    params = (('atr', 14), ('atr_sl', 1.25), ('atr_tp', 2),
              ('sma', 20), ('ema', 20),
              ('didi_mid', 8), ('didi_long', 25),
              ('win_mult', 1), ('lose_mult', 1), ('win_streak_limit', 1), ('lose_streak_limit', 1),
              ('macd_fast', 12), ('macd_slow', 26), ('macd_signal', 9),
              ('rsi', 14), ('adx', 14), ('adx_cutoff', 30))
    base_name = 'Trend'
    full_name = ''

    def __init__(self, params=None, logging=True):
        super().__init__()
        self.logging = logging
        self.full_name += self.base_name

        self.units = self.broker.get_cash() / 5
        self.inital_units = self.broker.get_cash() / 5
        self.curr_streak = {'win': 0, 'loss': 0}

        self.Indicators = Indicators(self.datas[0])

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

        self.macd = bt.indicators.MACD(self.datas[0], period_me1=self.params.macd_fast,
                                       period_me2=self.params.macd_slow,
                                       period_signal=self.params.macd_signal)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(self.datas[0], period=self.params.adx)
        self.DIplus = bt.indicators.PlusDirectionalIndicator(self.datas[0], period=self.params.adx)
        self.DIminus = bt.indicators.MinusDirectionalIndicator(self.datas[0], period=self.params.adx)
        self.rsi = bt.indicators.RSI_EMA(self.datas[0], period=self.params.rsi)
        self.prev_didi = 0

    def update_position_size(self):

        # Risk management TODO: Figure out how to have consistent risk b/w pairs
        two_percent_of_acc = round(self.broker.get_cash() * 0.02)
        atr = self.Indicators.get_atr(self.params.atr)

        # TODO: Get current quote currency (ex. USD in EUR/USD, CAD in NZD/CAD)
        with open("CURRENT_QUOTE_CURRENCY.txt", "r") as f:
            quote_currency = f.readline()

        if quote_currency == "JPY":
            tick_value = 0.01
            atr = atr * 100
        else:
            tick_value = 0.0001
            atr = atr * 10000

        pip_value = two_percent_of_acc / (1.5 * atr)

        # TODO: Get value of base currency of symbol (AUD in AUD/NZD, EUR in EUR/USD) in USD (my account currency)
        base_value = self.data_close[0]

        # Units for trade
        position_size = round(pip_value / (tick_value * base_value))
        self.units = position_size

    def update_streak(self, streak):
        self.curr_streak[streak] += 1
        if streak == 'loss':
            self.curr_streak['win'] = 0
        elif streak == 'win':
            self.curr_streak['loss'] = 0

    def get_trend(self):
        # Get the MACD line and the signal line
        macd_line = self.macd.macd[-1]
        signal_line = self.macd.signal[-1]

        # Calculate the difference between the MACD line and the signal line
        diff = macd_line - signal_line

        # Check the direction of the trend
        if diff > 0:
            # The MACD line is above the signal line, indicating an uptrend
            return "UPTREND"
        else:
            # The MACD line is below the signal line, indicating a downtrend
            return "DOWNTREND"

    def buy_sell_signal(self, indicator):
        if indicator == "rsi":
            # RSI
            upperband = self.rsi.params.upperband
            lowerband = self.rsi.params.lowerband
            rsi_buy_signal = self.rsi[-1] > upperband > self.rsi[0]  # Check if went below overbought
            rsi_sell_signal = self.rsi[-1] < lowerband < self.rsi[0]  # Check if went above oversold

            return rsi_buy_signal, rsi_sell_signal

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

    def make_prediction(self):
        # Variables
        prediction = 0
        # trend = self.get_trend()
        # uptrend, downtrend = trend == "UPDTREND", trend == "DOWNTREND"

        # CHECK FOR SIGNALS
        BUY_SIGNAL, SELL_SIGNAL = self.buy_sell_signal('rsi')

        if BUY_SIGNAL:
            prediction = 1
        elif SELL_SIGNAL:
            prediction = -1

        return prediction

    def set_sl_tp(self, prediction):
        close = self.data_close[0]
        ATR = self.Indicators.get_atr(self.params.atr)
        if prediction == 1:
            self.sl = close - (ATR * self.params.atr_sl)
            self.tp = close + (ATR * self.params.atr_tp)
        elif prediction == -1:
            self.sl = close + (ATR * self.params.atr_sl)
            self.tp = close - (ATR * self.params.atr_tp)

    # Iterates through bars
    def next(self):

        # See if trade conditions are all met
        if not self.can_trade():
            return

        # No open position, make an order
        if not self.position:

            self.update_position_size()

            prediction = self.make_prediction()

            # Set SL and TP
            self.set_sl_tp(prediction)

            close = self.data_close[0]

            if prediction == 1:
                self.log(f'BUY CREATE {self.data_close[0]:2f}')
                self.order = self.buy(size=self.units, price=close)
            elif prediction == -1:
                self.log(f'SELL CREATE {self.data_close[0]:2f}')
                self.order = self.sell(size=self.units, price=close)
        else:
            # We're in the market, check for SL/TP hit and act accordingly
            exited = self.check_stop_take()
            if exited:
                return

    # Calls after backtest completes
    def stop(self):
        pass


class MACD(Base):
    # DEFAULT STARTING VALS
    params = (('atr', 14), ('atr_sl', 1.25), ('atr_tp', 2), ('ema', 100),
              ('macd_fast', 12), ('macd_slow', 26), ('macd_signal', 9),
              ('rsi', 14), ('adx', 14), ('adx_cutoff', 30))
    base_name = 'MACD'
    full_name = ''

    def __init__(self, params=None, logging=True):
        super().__init__()
        self.logging = logging
        self.full_name += self.base_name

        self.units = self.broker.get_cash() / 5

        self.Indicators = Indicators(self.datas[0])

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

        self.macd = bt.indicators.MACD(self.datas[0], period_me1=self.params.macd_fast,
                                       period_me2=self.params.macd_slow,
                                       period_signal=self.params.macd_signal)
        self.ema = bt.indicators.ExponentialMovingAverage(self.datas[0], period=self.params.ema)

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
        # if not self.adx[0] > self.params.adx_cutoff:
        #     return False

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
            # self.update_streak('win')
        elif stop:
            self.log(f'STOP LOSS HIT {self.data_close[0]:2f}')
            self.order = self.close()
            # self.update_streak('loss')

        if stop or take:
            return True

    def update_position_size(self):

        # Risk management TODO: Figure out how to have consistent risk b/w pairs
        two_percent_of_acc = round(self.broker.get_cash() * 0.02)
        atr = self.Indicators.get_atr(self.params.atr)

        # TODO: Get current quote currency (ex. USD in EUR/USD, CAD in NZD/CAD)
        with open("CURRENT_QUOTE_CURRENCY.txt", "r") as f:
            quote_currency = f.readline()

        if quote_currency == "JPY":
            tick_value = 0.01
            atr = atr * 100
        else:
            tick_value = 0.0001
            atr = atr * 10000

        pip_value = two_percent_of_acc / (1.5 * atr)

        # TODO: Get value of base currency of symbol (AUD in AUD/NZD, EUR in EUR/USD) in USD (my account currency)
        base_value = self.data_close[0]

        # Units for trade
        position_size = round(pip_value / (tick_value * base_value))
        self.units = position_size

    def make_prediction(self):
        # Variables
        prediction = 0

        close = self.data_close[0]
        ema = self.ema[0]

        # MACD
        prev_macd, macd = self.macd.macd[-1], self.macd.macd[0]
        prev_signal, signal = self.macd.signal[-1], self.macd.signal[0]

        if prev_macd < prev_signal and macd > signal and close > ema:
            prediction = -1
        elif prev_macd > prev_signal and macd < signal and close < ema:
            prediction = 1

        return prediction

    def set_sl_tp(self, prediction):
        close = self.data_close[0]
        ATR = self.Indicators.get_atr(self.params.atr)
        if prediction == 1:
            self.sl = close - (ATR * self.params.atr_sl)
            self.tp = close + (ATR * self.params.atr_tp)
        elif prediction == -1:
            self.sl = close + (ATR * self.params.atr_sl)
            self.tp = close - (ATR * self.params.atr_tp)

    # Iterates through bars
    def next(self):
        # See if trade conditions are all met
        if not self.can_trade():
            return

        # No open position, make an order
        if not self.position:

            self.update_position_size()

            prediction = self.make_prediction()

            # Set SL and TP
            self.set_sl_tp(prediction)

            close = self.data_close[0]

            if prediction == 1:
                self.log(f'BUY CREATE {self.data_close[0]:2f}')
                self.order = self.buy(size=self.units, price=close)
            elif prediction == -1:
                self.log(f'SELL CREATE {self.data_close[0]:2f}')
                self.order = self.sell(size=self.units, price=close)
        else:
            # We're in the market, check for SL/TP hit and act accordingly
            exited = self.check_stop_take()
            if exited:
                return

    # Calls after backtest completes
    def stop(self):
        pass
