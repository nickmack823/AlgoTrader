import sys
import time
from datetime import datetime, timedelta
from os.path import exists
import joblib
from model_classes.MLModel import MLModel
import tpqoa
import pandas as pd
from indicator_getters.indicators import IndicatorCalculator


class Trader(tpqoa.tpqoa):

    def __init__(self, config_file, instrument, timeframe, strategy='model'):
        super().__init__(config_file)
        self.instrument = instrument
        self.timeframe = pd.to_timedelta(timeframe)  # Convert timeframe to workable object
        self.model_timeframe = timeframe[-1].upper() + timeframe[0:-1]
        self.tick_data = pd.DataFrame()
        self.last_bar = None
        self.data = None
        self.feat_collector = None
        self.curr_balance = float(self.get_account_summary()['balance'])
        self.starting_balance = self.curr_balance
        self.units = round(self.curr_balance * 10)

        self.tp_mult = 3.25
        self.sl_mult = 2.0

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

        self.strategy = strategy
        self.indicators_to_get = []
        self.model_features = ['paverage2close', 'proc15close']
        self.indicators_to_get = ['paverage', 'proc', 'atr', 'adx']
        if self.strategy == 'model':
            self.model = joblib.load("EUR_USD-2000-01-01-2022-10-22-M5.joblib")
            # self.model = None
            # self.get_model()
        elif self.strategy == 'macd_scalp':
            self.indicators_to_get = ['macd', 'ema']
        elif self.strategy == 'model_adx':
            self.model = joblib.load("EUR_USD-2000-01-01-2022-10-22-M5.joblib")

    def update_units(self):
        self.curr_balance = float(self.get_account_summary()['balance'])
        self.units = round(self.curr_balance * 10)

    def max_profit_loss_reached(self):
        total_profits = sum(self.profits)
        percent_change = total_profits/self.starting_balance
        if percent_change >= 0.02:
            return 'Max profit reached.'
        elif percent_change <= -0.02:
            return 'Max loss reached.'
        else:
            return False

    def get_recent_bar(self):
        print(f'Getting recently closed bar...')
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

        # Get features of historical data
        self.feat_collector.set_data(self.data)
        self.feat_collector.calculate_features(self.indicators_to_get)
        self.data = self.feat_collector.get_data()
        self.last_bar = self.data.index[-1]
        print('Recently closed bar retrieved, features calculated.')

    def get_historical_data(self, days=1):
        print(f'Getting historical data from {days} days ago to now...')
        while True:  # Repeat until historical bars retrieved
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
            # Accept historical data if less than (timeframe) has elapsed since last full historical bar and now
            if pd.to_datetime(datetime.utcnow()).tz_localize('UTC') - self.last_bar < self.timeframe:
                break

        # Get features of historical data
        self.feat_collector = IndicatorCalculator(self.data)
        self.feat_collector.calculate_features(self.indicators_to_get)
        self.data = self.feat_collector.get_data()
        print('Historical data retrieved and features calculated.')
        print('\nWaiting for current candle to close...')

    def on_success(self, tick_time, bid, ask):
        """After successful streaming of tick data, stores data into class DataFrame"""
        success_data = {'bid': bid, 'ask': ask, 'mid': (ask + bid) / 2}
        df = pd.DataFrame(success_data, index=[pd.to_datetime(tick_time)])
        self.tick_data = self.tick_data.append(df)

        # If a time longer than timeframe has elapsed between last full bar and most recent tick, resample to get
        # most recently completed bar data
        recent_tick = pd.to_datetime(tick_time)

        # # NOTE: OANDA time is 4 hours ahead (GMT+0)
        if recent_tick.time() >= pd.to_datetime("17:00").time():  # Stop trading at 1pm EST
            self.terminate_session('Reached end of trading hours.')

        # Check for stop loss/take profit triggers
        self.check_price_triggers()

        # On candle close
        if recent_tick - self.last_bar > self.timeframe:
            print('Candle closed, doing stuff...')
            self.get_recent_bar()  # Get recently closed candle data
            self.check_strategy()  # Check for model/strategy trigger
            self.execute_trades()  # Apply model/strategy decision
            # self.set_stop_loss(trailing=True)  # Set stop loss
            self.update_units()  # Update number of units used
            print(f'Current Position: {self.position} | SL: {self.stop_loss} | TP: {self.take_profit}')

    def get_model(self):
        print('Getting model...')
        yesterday = datetime.now() - timedelta(days=1)
        yesterday = datetime.strftime(yesterday, '%Y-%m-%d')
        ml_model = MLModel(self.instrument, '2014-01-01', yesterday, self.model_timeframe, 0.00007,
                           features=self.model_features)
        ml_model.fit_model()
        self.model = ml_model.model
        print('Model acquired.')

    def check_strategy(self):
        if self.strategy == 'model':
            self.prediction = self.model.predict(self.data[self.model_features].iloc[-1].to_frame().T)[0]
            print('model ooga ooga')
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
        elif self.strategy == 'model_adx':
            curr_adx = self.data['ADX_14'].iloc[-1]
            if curr_adx < 30:
                print('ADX not good')
                self.prediction = 0
            else:
                print('ADX good')
                self.prediction = self.model.predict(self.data[self.model_features].iloc[-1].to_frame().T)[0]

    def check_price_triggers(self):
        # Check for stop loss trigger
        if self.stop_loss is not None:
            if self.position == 1 and self.tick_data['ask'].iloc[-1] <= self.stop_loss:
                self.close_current_position(reason='Stop loss triggered (Long Position)')
            elif self.position == -1 and self.tick_data['bid'].iloc[-1] >= self.stop_loss:
                self.close_current_position(reason='Stop loss triggered (Short Position)')
        # Check for take profit trigger
        if self.take_profit is not None:
            if self.position == 1 and self.tick_data['ask'].iloc[-1] >= self.take_profit:
                self.close_current_position(reason='Take profit triggered (Long Position)')
            elif self.position == -1 and self.tick_data['bid'].iloc[-1] <= self.take_profit:
                self.close_current_position(reason='Take profit triggered (Short Position)')

    def stream(self, stop=None):
        self.stream_data(self.instrument, stop=stop)

    def set_stop_loss(self, trailing=False):
        atr = self.data['ATRr_14'].iloc[-1]
        price = self.data['close'].iloc[-1]
        # Initial stop loss
        if not trailing:
            if self.position == 1:
                self.stop_loss = price - (atr * self.sl_mult)
            elif self.position == -1:
                self.stop_loss = price + (atr * self.sl_mult)
            print(f'Setting initial stop loss to {self.stop_loss} | Position == {self.position}')
        # Trailing stop loss updates if price moves favorably
        else:
            price_back_two_bars = self.data['close'].iloc[-2]
            if self.position == 1 and price > price_back_two_bars:
                self.stop_loss = price - (atr * self.sl_mult)
                print(f'Moving stop loss to {self.stop_loss} | Position == {self.position}')
            elif self.position == -1 and price < price_back_two_bars:
                self.stop_loss = price + (atr * self.sl_mult)
                print(f'Moving stop loss to {self.stop_loss} | Position == {self.position}')

    def execute_trades(self):
        atr = self.data['ATRr_14'].iloc[-1]
        price = self.data['close'].iloc[-1]

        this_hour = datetime.now().hour

        if this_hour < 3 or this_hour > 13:
            self.terminate_session('Reached end of trading hours bazinga')

        if self.prediction == 1:
            if self.position == 0:
                order = self.create_order(self.instrument, self.units, suppress=True, ret=True)
                self.position = 1

                self.take_profit = price + (atr * self.tp_mult)
                self.set_stop_loss(trailing=False)
                self.report_trade(order, "GOING LONG")
            elif self.position == -1:
                order = self.create_order(self.instrument, self.units * 2, suppress=True, ret=True)
                self.position = 1

                self.take_profit = price + (atr * self.tp_mult)
                self.set_stop_loss(trailing=False)
                self.report_trade(order, "GOING LONG")
        elif self.prediction == -1:
            if self.position == 0:
                order = self.create_order(self.instrument, -self.units, suppress=True, ret=True)
                self.position = -1

                self.take_profit = price - (atr * 3)
                self.set_stop_loss(trailing=False)
                self.report_trade(order, "GOING SHORT")
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units * 2, suppress=True, ret=True)
                self.position = -1

                self.take_profit = price - (atr * 3)
                self.set_stop_loss(trailing=False)
                self.report_trade(order, "GOING SHORT")
        elif self.prediction == 0:
            pass
            # self.close_current_position('Prediction == 0')

    def report_trade(self, order, going):
        trade_time = order["time"]
        units = order["units"]
        price = order["price"]
        pl = float(order["pl"])
        self.profits.append(pl)
        cumpl = sum(self.profits)
        print("\n" + 100 * "-")
        print("{} | {}".format(trade_time, going))
        print("{} | units = {} | price = {} | P&L = ${} | Cum P&L = ${}".format(trade_time, units, price, pl, cumpl))
        print(100 * "-" + "\n")

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
                print(e, end=" | ")
            else:
                success = True
                break
            finally:
                attempts += 1
                print("Attempt: {}".format(attempts), end='\n')
                if not success:
                    if attempts >= max_attempts:
                        print("max_attempts reached!")
                        try:  # try to terminate session
                            time.sleep(wait)
                            self.terminate_session(cause="Unexpected Session Stop (too many errors).")
                            self.start_trading(days=1)  # Restart
                        except Exception as e:
                            print(e, end=" | ")
                            print("Could not terminate session properly!")
                        finally:
                            break
                    else:  # try again
                        time.sleep(wait)
                        self.tick_data = pd.DataFrame()

    def close_current_position(self, reason=None):
        if self.position != 0:
            print(f'Closing out current position | Reason: {reason}')
            positions = self.get_positions()
            long_units = round(float(positions[0]['long']['units']))
            short_units = round(float(positions[0]['short']['units']))
            if long_units > 0:
                curr_units = long_units
            else:
                curr_units = short_units
            close_order = self.create_order(self.instrument, units=curr_units*-1, suppress=True, ret=True)
            self.report_trade(close_order, "GOING NEUTRAL")
            self.position = 0
            self.stop_loss = None  # reset stop loss
            self.take_profit = None  # reset take profit

            # Check profit/loss for day
            # profit_loss = self.max_profit_loss_reached()
            # if profit_loss is not False:
            #     self.terminate_session(profit_loss)

    def terminate_session(self, cause):
        self.stop_stream = True
        if self.position != 0:
            print('Closing out order to end trading session...')
            self.close_current_position(reason='Reached end of trading hours.')
        print(cause, end=" | ")
        self.record_daily_trades()  # Write today's transaction history to a .csv
        sys.exit()

    def record_daily_trades(self):
        now = datetime.now()
        last_transaction_id = int(self.get_account_summary()['lastTransactionID'])
        transactions = self.get_transactions(tid=last_transaction_id - 200)
        transactions_today = []
        for transaction in transactions:
            date_string = transaction['time']
            date = datetime.strptime(date_string[0:10], '%Y-%m-%d')
            date = date.replace(hour=int(date_string[11:13])) \
                .replace(minute=int(date_string[14:16])) \
                .replace(second=int(date_string[17:19]))
            if date.date() == now.date():
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
            transaction['tradeOpened'] = transaction['tradeOpened']['tradeID'] if 'tradeOpened' in transaction.keys() else 'N/A'
            transaction['tradesClosed'] = transaction['tradesClosed'][0]['tradeID'] if 'tradesClosed' in transaction.keys() else 'N/A'

            # Clean up the date/time
            date_string = transaction['time']
            date = datetime.strptime(date_string[0:10], '%Y-%m-%d')
            date = date.replace(hour=int(date_string[11:13])) \
                .replace(minute=int(date_string[14:16])) \
                .replace(second=int(date_string[17:19]))
            transaction['date'] = date.date()
            transaction['time'] = date.time()

            # Prettify
            ordered_transaction = {
                'id': transaction['id'],
                'date': transaction['date'],
                'time': transaction['time'],
                'instrument': transaction['instrument'],
                'timeframe': self.model_timeframe,
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
            fp = f'{self.strategy}_demo_trades.csv'
            if exists(fp):
                df_existing = pd.read_csv(fp)
                i = len(df_existing)
                df = pd.DataFrame(ordered_transaction, index=[i])
                df.to_csv(fp, mode='a', index=True, header=False)
            else:
                df = pd.DataFrame(ordered_transaction, index=[0])
                df.index.name = 'transaction_number'
                df.to_csv(fp)


if __name__ == "__main__":
    # TODO: Set trailing stop loss equal to price +- 2x ATR
    curr_hour = datetime.now().hour

    while curr_hour < 3 or curr_hour > 13:
        time.sleep(60)
        print('checking hour again...')
        curr_hour = datetime.now().hour
        print(f'hour is {curr_hour}')

    td = Trader(config_file=r'C:\Users\Nick\Documents\GitHub\AlgoTrader\oanda.cfg',
                instrument='EUR_USD', timeframe='5m', strategy='model_adx')
    td.start_trading(days=5)
