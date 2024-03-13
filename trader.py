import json
import logging
import multiprocessing
import os
import random
import time
import requests
from datetime import datetime, timedelta, date
from os.path import exists
import coloredlogs
import tpqoa
import pandas as pd
import pandas_ta as ta

today = datetime.today()
year, month, day = today.strftime('%Y'), today.strftime('%B'), today.strftime('%d')


def first_friday(y, m):
    """Return datetime.date for monthly option expiration given year and
    month
    """
    # The 15th is the lowest third day in the month
    first = date(y, m, 1)
    # What day of the week is the 1st?
    w = first.weekday()
    # Friday is weekday 4
    if w != 4:
        # Replace just the day (of month)
        first = first.replace(day=(1 + (4 - w) % 7))
    return first


def get_exchange_rate(currency):
    if currency == "USD":
        return 1

    attempts = 0
    while attempts < 50:
        try:
            url = f"https://api.exchangerate.host/convert?from={currency}&to=USD&amount=1"
            response = requests.get(url)
            data = response.json()
            rate = data["result"]
            return rate
        except:
            attempts += 1

    return 1


def record_daily_trades(tpqoa_object, strategy):
    print("RECORDING TRADES")
    now = datetime.now()
    last_transaction_id = int(tpqoa_object.get_account_summary()['lastTransactionID'])
    transactions = tpqoa_object.get_transactions(tid=last_transaction_id - 200)
    transactions_today = []
    for transaction in transactions:
        date_string = transaction['time']
        curr_date = datetime.strptime(date_string[0:10], '%Y-%m-%d')
        curr_date = curr_date.replace(hour=int(date_string[11:13])) \
            .replace(minute=int(date_string[14:16])) \
            .replace(second=int(date_string[17:19]))
        # if curr_date.date() == now.date():
        transactions_today.append(transaction)

    to_record = []
    for transaction in transactions_today:
        if transaction['type'] == 'ORDER_FILL':
            to_record.append(transaction)

    keys_to_keep = ['id', 'time', 'type', 'orderID', 'instrument', 'units', 'price', 'reason', 'pl', 'financing',
                    'commission', 'accountBalance', 'tradesClosed', 'tradeOpened', 'halfSpreadCost']
    for transaction in to_record:

        # Remove unwanted key-value pairs
        to_remove = []
        for key in transaction.keys():
            if key not in keys_to_keep:
                to_remove.append(key)
        for key in to_remove:
            transaction.pop(key)

        # Extract id from tradeOpened y tradesClosed dictionaries
        transaction['tradeOpened'] = transaction['tradeOpened'][
            'tradeID'] if 'tradeOpened' in transaction.keys() else 'N/A'
        transaction['tradesClosed'] = transaction['tradesClosed'][0][
            'tradeID'] if 'tradesClosed' in transaction.keys() else 'N/A'

        # Clean up the date/time
        date_string = transaction['time']
        curr_date = datetime.strptime(date_string[0:10], '%Y-%m-%d')
        curr_date = curr_date.replace(hour=int(date_string[11:13])) \
            .replace(minute=int(date_string[14:16])) \
            .replace(second=int(date_string[17:19]))
        transaction['date'] = curr_date.date()
        transaction['time'] = (curr_date - timedelta(hours=4)).time()

        # Prettify
        ordered_transaction = {
            'id': transaction['id'],
            'date': transaction['date'],
            'time': transaction['time'],
            'instrument': transaction['instrument'],
            # 'timeframe': self.timeframe,
            'units': transaction['units'],
            'price': transaction['price'],
            'half_spread_cost': round(float(transaction['halfSpreadCost']), 2),
            'profit/loss': round(float(transaction['pl']), 2),
            'order_reason': transaction['reason'],
            'financing': round(float(transaction['financing']), 2),
            'commission': round(float(transaction['commission']), 2),
            'account_balance': transaction['accountBalance'],
            'trades_closed': transaction['tradesClosed'],
            'trade_opened': transaction['tradeOpened']}

        # Write to file
        fp = f'{strategy}_demo_trades.csv'
        if exists(fp):
            df_existing = pd.read_csv(fp)
            i = len(df_existing)
            df = pd.DataFrame(ordered_transaction, index=[i])
            df = df.drop_duplicates(keep='first')
            df.to_csv(fp, mode='a', index=True, header=False)
        else:
            df = pd.DataFrame(ordered_transaction, index=[0])
            df.index.name = 'transaction_number'
            df.to_csv(fp)


class Trader(tpqoa.tpqoa):

    def __init__(self, config_file, instrument, timeframe, params, strategy):
        super().__init__(config_file)
        self.instrument = instrument
        self.timeframe = pd.to_timedelta(timeframe)  # Convert timeframe to workable object
        self.params = params  # Dict of indicators and their params
        self.strategy = strategy

        self.LOGGER = self.get_logger()

        self.LOGGER.info(
            f"({self.instrument}) | Timeframe: {timeframe} |"
            f" Strategy: {self.strategy} | Parameters: {self.params}")

        self.tick_data = pd.DataFrame()
        self.last_bar = None
        self.data = None
        # self.feat_collector = None
        self.curr_balance = float(self.get_account_summary()['balance'])
        self.starting_balance = self.curr_balance
        self.units = None

        self.tp_mult = self.params['atr_tp']
        self.sl_mult = self.params['atr_sl']

        positions = self.get_positions()
        if len(positions) > 0:
            long_units = float(positions[0]['long']['units'])
            short_units = float(positions[0]['short']['units'])
            if long_units > 0:
                self.position = 1
            elif short_units < 0:
                self.position = -1
        else:
            self.position = 0

        self.stop_loss, self.take_profit = None, None

        self.prediction = None
        self.profits = []

        self.indicators_to_get = ['atr', 'adx', 'rsi']

        self.trading = True
        # self.terminate_session("TEST")

    def get_logger(self):
        log_paths = ["trading_logs", f"trading_logs/{year}", f"trading_logs/{year}/{month}",
                     f"trading_logs/{year}/{month}/{day}"]
        log_folder = f"trading_logs/{year}/{month}/{day}"

        # Make log directories
        for lp in log_paths:
            try:
                os.mkdir(lp)
            except FileExistsError:
                pass

        log_path_full = f'{log_folder}/{self.instrument}_TRADING.log'

        # Logging
        log = logging.getLogger()
        log.setLevel('INFO')

        file_log_handler = logging.FileHandler(log_path_full)
        log.addHandler(file_log_handler)

        stderr_log_handler = logging.StreamHandler()
        log.addHandler(stderr_log_handler)

        # nice output format
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_log_handler.setFormatter(formatter)
        stderr_log_handler.setFormatter(formatter)

        # Coloring
        coloredlogs.install(fmt="%(asctime)s %(message)s", level='INFO', logger=log)

        return log

    def get_indicators(self):
        self.data.ta.atr(append=True, length=self.params['atr'])
        self.data.ta.rsi(append=True, length=self.params['rsi'])
        self.data.ta.adx(append=True, length=self.params['adx'])

    def get_atr(self):
        return self.data["ATRr_" + str(self.params['atr'])].iloc[-1]

    def get_rsi(self, bars_back=1):
        return self.data["RSI_" + str(self.params['rsi'])].iloc[-bars_back]

    def get_adx(self):
        return self.data["ADX_" + str(self.params["adx"])].iloc[-1]

    def update_position_size(self):
        risk = self.curr_balance * 0.01  # percent of acc to risk, currently 1% (to acc. for margin requirements)
        atr = self.get_atr()

        currencies = self.instrument.split("_")
        base_currency, quote_currency = currencies[0], currencies[1]

        if quote_currency == "JPY":
            tick_value = 0.01
            atr = atr * 100
        else:
            tick_value = 0.0001
            atr = atr * 10000

        tick_div_price = tick_value / self.data['close'].iloc[-1]

        pip_value = risk / (1.5 * atr)

        exchange_rate = get_exchange_rate(base_currency)

        # Units for trade
        position_size = round(pip_value / (tick_div_price * exchange_rate))
        self.units = position_size

    def max_profit_loss_reached(self):
        total_profits = sum(self.profits)
        percent_change = total_profits / self.starting_balance
        if percent_change >= 0.02:
            return 'Max profit reached.'
        elif percent_change <= -0.02:
            return 'Max loss reached.'
        else:
            return False

    def get_recent_bar(self):
        self.LOGGER.info(f'({self.instrument}) Getting recently closed bar...')
        while True:  # Repeat until bar retrieved
            time.sleep(1)
            now = datetime.utcnow()
            now = now - timedelta(microseconds=now.microsecond)
            start = now - timedelta(days=1)
            df = self.get_history(instrument=self.instrument, start=start, end=now, granularity='S5', price='M',
                                  localize=False).dropna()
            df.rename(columns={'c': 'close', 'o': 'open', 'l': 'low', 'h': 'high'}, inplace=True)
            df.drop(['complete'], axis=1, inplace=True)
            df = df.resample(self.timeframe, label='right').last().dropna().iloc[:-1]
            final_row = {}
            for c in df.columns:
                final_row[c] = df[c].iloc[-1]
            self.data.loc[df.index[-1]] = final_row
            self.last_bar = self.data.index[-1]  # first defined here
            # Accept historical data if less than (timeframe) has elapsed since last full historical bar and now
            if pd.to_datetime(datetime.utcnow()).tz_localize('UTC') - self.last_bar < self.timeframe:
                break

        # Get indicator values of recent bar
        # self.feat_collector.set_data(self.data)
        # self.feat_collector.calculate_features(self.indicators_to_get)
        # self.data = self.feat_collector.get_data()
        self.get_indicators()
        self.last_bar = self.data.index[-1]
        # self.LOGGER.info('Recently closed bar retrieved, features calculated.')

    def get_historical_data(self, days=1):
        self.LOGGER.info(f'({self.instrument}) Getting historical data from {days} days ago to now...')
        # while True:  # Repeat until historical bars retrieved
        now = datetime.utcnow()
        now = now - timedelta(microseconds=now.microsecond)
        start = now - timedelta(days=days)
        df = self.get_history(instrument=self.instrument, start=start, end=now, granularity='S5', price='M',
                              localize=False).dropna()
        df.rename(columns={'c': 'close', 'o': 'open', 'l': 'low', 'h': 'high'}, inplace=True)
        df.drop(['complete'], axis=1, inplace=True)
        df = df.resample(self.timeframe, label='right').last().dropna().iloc[:-1]
        self.data = df.copy()  # first defined here
        self.last_bar = self.data.index[-1]  # first defined here
        now = pd.to_datetime(datetime.utcnow()).tz_localize('UTC')
            # # Accept historical data if less than (timeframe) has elapsed since last full historical bar and now
            # if now - self.last_bar < self.timeframe:
            #     break

        # Get indicator values of historical data
        # self.feat_collector = IndicatorCalculator(self.data)
        # self.feat_collector.calculate_features(self.indicators_to_get)
        # self.data = self.feat_collector.get_data()
        self.get_indicators()
        self.LOGGER.info(f'({self.instrument}) Historical data retrieved and indicators calculated.')
        self.LOGGER.info(f'\n({self.instrument}) Waiting for current candle to close...')

    def on_success(self, tick_time, bid, ask):
        """After successful streaming of tick data, stores data into class DataFrame"""
        success_data = {'bid': bid, 'ask': ask, 'mid': (ask + bid) / 2}
        df = pd.DataFrame(success_data, index=[pd.to_datetime(tick_time)])
        self.tick_data = self.tick_data.append(df)

        # If a time longer than timeframe has elapsed between last full bar and most recent tick, resample to get
        # most recently completed bar data
        recent_tick = pd.to_datetime(tick_time)

        self.check_price_triggers()  # Check for stop loss/take profit triggers

        # # NOTE: OANDA time is 4 hours ahead (GMT+0)
        # if datetime.now().hour >= 12:  # Stop trading at 12pm EST
        #     self.terminate_session('Reached end of trading hours.')

        # On candle close
        if recent_tick - self.last_bar > self.timeframe and self.can_trade():
            self.LOGGER.info(
                f'({self.instrument}) Candle closed, doing stuff... | Current Time: {datetime.now().time()}')
            self.get_recent_bar()  # Get recently closed candle data
            self.check_strategy()  # Check for model/strategy trigger
            self.execute_trades()  # Apply model/strategy decision
            self.LOGGER.info(f'({self.instrument}) Price: {self.data["close"].iloc[-1]} |'
                             f' Current Position: {self.position} - {self.units} Units | '
                             f'SL: {self.stop_loss} | TP: {self.take_profit} | Daily P/L: {sum(self.profits)}')

    def in_trade(self):
        if self.position != 0:
            return True
        else:
            return False

    def can_trade(self):
        # Don't do anything if an order is already open
        if self.in_trade():
            return False

        now = datetime.now()

        first_friday_of_month = first_friday(now.year, now.month)

        # Don't trade on first friday of month (Non-Farm Payroll)
        if now.day == first_friday_of_month.day:
            self.terminate_session("FIRST FRIDAY")
            return False

        curr_hour = int(now.strftime("%H"))

        # Outside trading hours
        if curr_hour >= 17:
            self.LOGGER.info(f'CLOSING OUT DAY')
            self.terminate_session("END OF HOURS")
            return False

        return True

    def check_strategy(self):
        curr_rsi = self.get_rsi()
        prev_rsi = self.get_rsi(bars_back=2)
        rsi_buy_signal = prev_rsi > 70 > curr_rsi  # Check if went below overbought
        rsi_sell_signal = prev_rsi < 30 < curr_rsi  # Check if went above oversold

        if rsi_buy_signal:
            self.prediction = 1
            self.LOGGER.info(f"PREDICTION=1 (PREV_RSI={prev_rsi}, CURR_RSI={curr_rsi}")
        elif rsi_sell_signal:
            self.prediction = -1
            self.LOGGER.info(f"PREDICTION=-1 (PREV_RSI={prev_rsi}, CURR_RSI={curr_rsi}")
        else:
            self.prediction = 0

        curr_adx = self.get_adx()
        if curr_adx < 30:
            if self.prediction != 0:
                self.LOGGER.info(f"ADX={curr_adx}, CAN'T TRADE")
            self.prediction = 0

    def check_price_triggers(self):
        # Check for stop loss trigger
        curr_price = self.tick_data['mid'].iloc[-1]
        if self.stop_loss is not None:
            if self.position == 1 and curr_price <= self.stop_loss:
                self.close_current_position(reason='Stop loss triggered (Long Position)')
            elif self.position == -1 and curr_price >= self.stop_loss:
                self.close_current_position(reason='Stop loss triggered (Short Position)')
        # Check for take profit trigger
        if self.take_profit is not None:
            if self.position == 1 and curr_price >= self.take_profit:
                self.close_current_position(reason='Take profit triggered (Long Position)')
            elif self.position == -1 and curr_price <= self.take_profit:
                self.close_current_position(reason='Take profit triggered (Short Position)')

    def stream(self, stop=None):
        self.stream_data(self.instrument, stop=stop)

    def set_stop_loss(self, trailing=False):
        atr = self.get_atr()
        price = self.data['close'].iloc[-1]

        # Initial stop loss
        if not trailing:
            if self.position == 1:
                self.stop_loss = price - (atr * self.sl_mult)
            elif self.position == -1:
                self.stop_loss = price + (atr * self.sl_mult)
            self.LOGGER.info(f'({self.instrument}) Setting stop loss to {self.stop_loss} | Position == {self.position}')
        # Trailing stop loss updates if price moves favorably
        else:
            price_back_two_bars = self.data['close'].iloc[-2]
            if self.position == 1 and price > price_back_two_bars:
                self.stop_loss = price - (atr * self.sl_mult)
                self.LOGGER.info(
                    f'({self.instrument}) Moving stop loss to {self.stop_loss} | Position == {self.position}')
            elif self.position == -1 and price < price_back_two_bars:
                self.stop_loss = price + (atr * self.sl_mult)
                self.LOGGER.info(
                    f'({self.instrument}) Moving stop loss to {self.stop_loss} | Position == {self.position}')

    def execute_trades(self):
        atr = self.get_atr()
        price = self.data['close'].iloc[-1]

        if self.prediction == 1:  # signal to go long
            if self.position in [0, -1]:
                self.close_current_position(reason="BUYING")
                self.update_position_size()  # Update number of units used

                self.buy()

                self.take_profit = price + (atr * self.tp_mult)
                self.LOGGER.info(
                    f'({self.instrument}) Setting take profit to {self.take_profit} | Position == {self.position}')

                self.set_stop_loss(trailing=False)
        elif self.prediction == -1:  # signal to go short
            if self.position in [0, 1]:
                self.close_current_position(reason="SELLING")
                self.update_position_size()  # Update number of units used

                self.sell()

                self.take_profit = price - (atr * self.tp_mult)
                self.LOGGER.info(
                    f'({self.instrument}) Setting take profit to {self.take_profit} | Position == {self.position}')

                self.set_stop_loss(trailing=False)

        elif self.prediction == 0:
            pass

    def buy(self):
        order = self.create_order(self.instrument, self.units, suppress=True, ret=True)
        self.report_trade(order, "GOING LONG")
        self.position = 1

    def sell(self):
        order = self.create_order(self.instrument, -self.units, suppress=True, ret=True)
        self.report_trade(order, "GOING SHORT")
        self.position = -1

    def report_trade(self, order, going):
        trade_time = order["time"]
        units = order["units"]
        price = order["price"]
        pl = float(order["pl"])
        self.profits.append(pl)
        cumpl = sum(self.profits)
        self.LOGGER.info(f"({self.instrument}) {trade_time} | {going}")
        self.LOGGER.info(
            f"({self.instrument}) {trade_time} | units = {units} | price = {price} | P&L = ${pl} | "
            f"Cum P&L = ${cumpl} | TP = {self.take_profit} | SL = {self.stop_loss}")

    def start_trading(self, days, max_attempts=2000, wait=1):  # Error Handling
        attempts = 0
        success = False
        got_historical_data = False
        while True:
            try:
                if not got_historical_data:
                    self.get_historical_data(days)
                    got_historical_data = True
                self.stream_data(self.instrument)
            except Exception as e:
                self.LOGGER.info(f'({self.instrument}) {e}')
            else:
                success = True
                break
            finally:
                attempts += 1
                self.LOGGER.info(f"({self.instrument}) Attempt: {attempts}\n")
                if not success:
                    if attempts >= max_attempts:
                        self.LOGGER.info(f"({self.instrument}) max_attempts {max_attempts} reached!")
                        try:  # try to terminate session
                            time.sleep(wait)
                            self.terminate_session(cause="Unexpected Session Stop (too many errors).")
                            self.start_trading(days=1)  # Restart
                        except Exception as e:
                            self.LOGGER.info(f'({self.instrument}) + {e}')
                            self.LOGGER.info(f"({self.instrument}) Could not terminate session properly!")
                        finally:
                            break
                    else:  # try again
                        time.sleep(wait)
                        self.tick_data = pd.DataFrame()

    def close_current_position(self, reason=None):
        if self.position != 0:
            self.LOGGER.info(f'({self.instrument}) Closing out current position | Reason: {reason}')
            positions = self.get_positions()
            long_units = round(float(positions[0]['long']['units']))
            short_units = round(float(positions[0]['short']['units']))
            if long_units > 0:
                curr_units = long_units
            else:
                curr_units = short_units
            close_order = self.create_order(self.instrument, units=curr_units * -1, suppress=True, ret=True)
            self.report_trade(close_order, "GOING NEUTRAL")
            self.position = 0
            self.stop_loss = None  # reset stop loss
            self.take_profit = None  # reset take profit

    def terminate_session(self, cause):
        self.stop_stream = True
        if self.position != 0:
            self.LOGGER.info(f'({self.instrument}) Closing out order to end trading session...')
            self.close_current_position(reason='Reached end of trading hours.')
        self.LOGGER.info(f"({self.instrument}) Cause of termination: {cause}")
        self.trading = False


def trade(instrument, timeframe, params, out):
    td = Trader(config_file='C:\\Users\\sm598\\OneDrive\\Documents\\GitHub\\AlgoTrader\\oanda.cfg',
                instrument=instrument, timeframe=timeframe, params=params, strategy="RSI_ADX")
    td.start_trading(days=5)

    out.put(True)


if __name__ == "__main__":
    best_pair_settings = "BEST_PAIR_SETTINGS.csv"

    best = pd.read_csv(best_pair_settings)

    traders, traders_stopped = [], []
    out_q = multiprocessing.Queue()
    for index, row in best.iterrows():
        symbol = row['symbol']
        t = row['timeframe'][1:] + "m"
        symbol_params = json.loads(row['params'].replace("'", '"'))

        trader = multiprocessing.Process(target=trade, args=(symbol, t, symbol_params, out_q))
        traders.append(trader)

    for trader in traders:
        trader.start()

    while len(traders_stopped) < len(traders):
        time.sleep(15)
        try:
            traders_stopped.append(out_q.get())
        except:
            pass

    for trader in traders:
        trader.kill()
        trader.join()

    recorder = tpqoa.tpqoa('C:\\Users\\sm598\\OneDrive\\Documents\\GitHub\\AlgoTrader\\oanda.cfg')
    record_daily_trades(recorder, "RSI_ADX")
