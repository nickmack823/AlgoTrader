import sys
import time
from datetime import datetime, timedelta
from MLModel import MLModel
import tpqoa
import pandas as pd
from indicators import IndicatorCalculator


model_features = ['paverage2close', 'proc15close']


class Trader(tpqoa.tpqoa):

    def __init__(self, config_file, instrument, timeframe):
        super().__init__(config_file)
        self.instrument = instrument
        self.timeframe = pd.to_timedelta(timeframe)  # Convert timeframe to workable object
        self.model_timeframe = timeframe[-1].upper() + timeframe[0:-1]
        self.tick_data = pd.DataFrame()
        self.last_bar = None
        self.data = None
        self.model = None
        self.feat_collector = None
        self.curr_balance = float(self.get_account_summary()['balance'])
        self.starting_balance = self.curr_balance
        self.units = round(self.curr_balance * 10)

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

        self.prediction = None
        self.profits = []

        self.get_model()

    def update_units(self):
        self.curr_balance = float(self.get_account_summary()['balance'])
        self.units = round(self.curr_balance * 10)

    def max_profit_loss_reached(self):
        total_profits = sum(self.profits)
        percent_change = total_profits/self.starting_balance
        if percent_change >= 0.04:
            return 'Max profit reached.'
        elif percent_change <= -0.04:
            return 'Max loss reached.'
        else:
            return False

    def get_model(self):
        # if exists('model.joblib'):
        #     model = joblib.load('model.joblib')
        # else:
        yesterday = datetime.now() - timedelta(days=1)
        yesterday = datetime.strftime(yesterday, '%Y-%m-%d')
        ml_model = MLModel(self.instrument, '2016-01-01', yesterday, self.model_timeframe, 0.00007,
                           features=model_features)
        ml_model.fit_model()
        self.model = ml_model.model
        # joblib.dump(model, 'model.joblib')
        # self.model = model

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
        self.feat_collector.calculate_features(model_features)
        self.data = self.feat_collector.get_data()
        self.last_bar = self.data.index[-1]
        print('Recently closed bar retrieved, features calculated.')

    def get_historical_data(self, days=1):
        print(f'Getting historical data from {days} days ago to now...')
        while True:  # Repeat until historical bars retrieved
            time.sleep(1)
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
        self.feat_collector.calculate_features(model_features)
        self.data = self.feat_collector.get_data()
        print('Historical data retrieved and features calculated.')

    def on_success(self, time, bid, ask):
        """After successful streaming of tick data, stores data into class DataFrame"""
        success_data = {'bid': bid, 'ask': ask, 'mid': (ask + bid) / 2}
        df = pd.DataFrame(success_data, index=[pd.to_datetime(time)])
        self.tick_data = self.tick_data.append(df)

        # If a time longer than timeframe has elapsed between last full bar and most recent tick, resample to get
        # most recently completed bar data
        recent_tick = pd.to_datetime(time)

        # NOTE: OANDA time is 4 hours ahead (GMT+0)
        if recent_tick.time() >= pd.to_datetime("08:00").time():  # Stop trading at 4am EST
            self.terminate_session('Reached end of trading hours.')

        if recent_tick - self.last_bar > self.timeframe:
            # self.resample_and_join()
            self.get_recent_bar()
            self.predict_direction()
            self.execute_trades()
            self.update_units()
            # profit_loss = self.max_profit_loss_reached()
            # if profit_loss is not False:
            #     self.terminate_session(profit_loss)

    def resample_and_join(self):
        # Resample data to timeframe, excluding final data point (incomplete bar)
        resampled = self.tick_data.resample(self.timeframe, label='right').last().ffill().iloc[:-1]
        self.data = self.data.append(resampled)
        self.tick_data = self.tick_data.iloc[-1:]  # Only keep the latest tick
        self.last_bar = self.data.index[-1]  # update time of last full bar

    def stream(self, stop=None):
        self.stream_data(self.instrument, stop=stop)

    def predict_direction(self):
        self.prediction = self.model.predict(self.data[model_features].iloc[-1].to_frame().T)[0]

        # ******************** define your strategy here ************************
        # self.data["returns"] = np.log(self.data.close / self.data.close.shift())
        # self.data["position"] = self.model.predict(self.data[features])
        # ***********************************************************************

    def execute_trades(self):
        if self.prediction == 1:
            if self.position == 0:
                order = self.create_order(self.instrument, self.units, suppress=True, ret=True)
                self.report_trade(order, "GOING LONG")
            elif self.position == -1:
                order = self.create_order(self.instrument, self.units * 2, suppress=True, ret=True)
                self.report_trade(order, "GOING LONG")
            self.position = 1
        elif self.prediction == -1:
            if self.position == 0:
                order = self.create_order(self.instrument, -self.units, suppress=True, ret=True)
                self.report_trade(order, "GOING SHORT")
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units * 2, suppress=True, ret=True)
                self.report_trade(order, "GOING SHORT")
            self.position = -1
        elif self.prediction == 0:
            if self.position == -1:
                order = self.create_order(self.instrument, self.units, suppress=True, ret=True)
                self.report_trade(order, "GOING NEUTRAL")
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units, suppress=True, ret=True)
                self.report_trade(order, "GOING NEUTRAL")
            self.position = 0

    def report_trade(self, order, going):
        time = order["time"]
        units = order["units"]
        price = order["price"]
        pl = float(order["pl"])
        self.profits.append(pl)
        cumpl = sum(self.profits)
        print("\n" + 100 * "-")
        print("{} | {}".format(time, going))
        print("{} | units = {} | price = {} | P&L = ${} | Cum P&L = ${}".format(time, units, price, pl, cumpl))
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

    def terminate_session(self, cause):
        self.stop_stream = True
        if self.position != 0:
            print('Closing out order to end trading session...')
            positions = self.get_positions()
            long_units = round(float(positions[0]['long']['units']))
            short_units = round(float(positions[0]['short']['units']))
            if long_units > 0:
                curr_units = long_units
            else:
                curr_units = short_units
            close_order = self.create_order(self.instrument, units=curr_units*-1,
                                            suppress=True, ret=True)
            self.report_trade(close_order, "GOING NEUTRAL")
            self.position = 0
        print(cause, end=" | ")
        sys.exit()


if __name__ == "__main__":
    # TODO: Trade only from 8:00pm to 4:00am New York Time for USD_JPY
    td = Trader(config_file=r'C:\Users\Nick\Documents\GitHub\AlgoTrader\oanda.cfg',
                instrument='USD_JPY', timeframe='30m')
    td.start_trading(days=5)
