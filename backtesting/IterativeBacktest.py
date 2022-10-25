import itertools
from datetime import datetime, timedelta

import pandas as pd
import winsound

from IterativeBase import *


class IterativeBacktest(IterativeBase):
    ''' Class for iterative (bar-by-bar) backtesting of trading strategies.'''

    def close_position(self, close):

        lot_size = (self.units / 100000)

        if self.position == 1:
            diff = (close - self.price_traded_at)
            profit = diff * lot_size * 100000
        else:
            diff = (self.price_traded_at - close)
            profit = diff * lot_size * 100000

        self.profits.append(profit)

        self.current_balance += profit
        self.update_units()
        self.position = 0
        self.trades += 1

    def buy(self, bar):
        date, close, spread = self.get_values(bar)
        close += spread / 2  # ask price
        if self.position == -1:
            self.close_position(close)  # close current short before opening a buy

        self.price_traded_at = round(self.data.close.iloc[bar], 5)
        self.position = 1
        self.trades += 1
        # print("{} |  Buying {} for {}".format(date, self.units, round(close, 5)))

    def sell(self, bar):
        date, close, spread = self.get_values(bar)
        close -= spread / 2  # bid price
        if self.position == 1:
            self.close_position(close)

        self.price_traded_at = round(self.data.close.iloc[bar], 5)
        self.position = -1
        self.trades += 1
        # print("{} |  Selling {} for {}".format(date, self.units, round(close, 5)))

    def update_units(self):
        self.units = round(self.current_balance * 10)

    def set_stop_loss(self, bar):
        atr = self.data['ATRr_14'].iloc[bar]
        price = self.data['close'].iloc[bar]
        trailing = self.parameters['trailing_sl']
        # Initial stop loss
        if not trailing:
            if self.position == 1:
                self.stop_loss = price - (atr * self.parameters['atr_sl'])
            elif self.position == -1:
                self.stop_loss = price + (atr * self.parameters['atr_sl'])
        # Trailing stop loss updates if price moves favorably
        else:
            price_back_two_bars = self.data['close'].iloc[bar - 1]
            if self.position == 1 and price > price_back_two_bars:
                self.stop_loss = price - (atr * self.parameters['atr_sl'])
                # print(f'Moving stop loss to {self.stop_loss} | Position == {self.position}')
            elif self.position == -1 and price < price_back_two_bars:
                self.stop_loss = price + (atr * self.parameters['atr_sl'])
                # print(f'Moving stop loss to {self.stop_loss} | Position == {self.position}')

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
        if 'model' in self.strategy:
            self.prediction = self.model.predict(self.data[['paverage2close', 'proc15close']].iloc[bar].to_frame().T)[0]

    def trade_conditions_met(self, bar):
        date, close, spread = self.get_values(bar)
        date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

        # Trade only b/w 3:00am and 12:00pm
        if date.hour < 3 or date.hour > 12:
            return False

        # Ensure ADX value is >= the cutoff for trend strength
        if 'adx' in self.strategy:
            curr_adx = self.data['ADX_14'].iloc[bar]
            if curr_adx < self.parameters['adx_cutoff']:
                return False

        return True

    def iterate_through_bars(self):
        # reset
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.current_balance = self.initial_balance  # reset initial capital
        self.get_data()  # reset dataset

        for bar in range(len(self.data) - 1):  # all bars (except the last bar)
            date, close, spread = self.get_values(bar)

            # Check for TP/SL triggers
            self.check_price_triggers(bar)

            # Ensure all necessary conditions are fulfilled before taking a trade
            if not self.trade_conditions_met(bar):
                continue

            # Conditions met, make a prediction
            self.check_strategy(bar)  # Get prediction for current bar

            # Verify that Chaikin Money Flow supports prediction direction ( > 0 for long, < 0 for short
            if 'chaikin' in self.strategy:
                curr_chaikin = self.data['CMF_20'].iloc[bar]
                if (self.prediction == 1 and curr_chaikin <= 0) or (self.prediction == -1 and curr_chaikin >= 0):
                    continue

            # Open a trade
            if self.prediction == 1:  # signal to go long
                if self.position in [0, -1]:
                    self.buy(bar)
                    atr = self.data['ATRr_14'].iloc[bar]
                    self.take_profit = close + (atr * self.parameters['atr_tp'])
                    self.set_stop_loss(bar)
            elif self.prediction == -1:  # signal to go short
                if self.position in [0, 1]:
                    self.sell(bar)
                    atr = self.data['ATRr_14'].iloc[bar]
                    self.take_profit = close + (atr * self.parameters['atr_tp'])
                    self.set_stop_loss(bar)
            elif self.prediction == 0:
                self.close_position(close)

        if self.position != 0:
            self.close_position(self.get_values(bar + 1)[1])

        wins, losses = 0, 0
        total = len(self.profits)
        for p in self.profits:
            if p < 0:
                losses += 1
            else:
                wins += 1

        if total != 0:
            win_ratio = wins/total
        else:
            win_ratio = 0

        results = {
            'symbol': self.symbol,
            'start': self.start,
            'end': self.end,
            'timeframe': self.timeframe,
            'strategy': self.strategy,
            "final_balance": self.current_balance,
            "net_profit": self.current_balance - self.initial_balance,
            "trades": self.trades,
            "wins": wins,
            "losses": losses,
            "win ratio": win_ratio
        }
        results.update(self.parameters)
        #
        # if 'model' in self.strategy:
        #     results['model_details'] = f'{self.model_start}_{self.model_end}'

        return results


def get_model_date_ranges():
    # Get multiple possible date ranges to train model on
    yesterday = datetime.now() - timedelta(days=1)
    years = list(range(2000, yesterday.year + 1, 1))
    date_combos = []
    for year in years:
        d = datetime(year, 1, 1)
        if d < yesterday:
            date_combos.append((datetime.strftime(d, '%Y-%m-%d'), datetime.strftime(yesterday, '%Y-%m-%d')))

    return date_combos


def test_strategy(test_details):
        # Do the test
        test_details['balance'] = 1000
        print(test_details)
        ib = IterativeBacktest(test_details)
        results = ib.iterate_through_bars()
        df = pd.DataFrame(results, index=[test_details['test_number']])
        df.index.name = 'test_number'
        print(df.head())

        file_path = test_details['file_path']
        if exists(file_path):
            df.to_csv(file_path, mode='a', index=True, header=False)
        else:
            df.to_csv(file_path)


def test_strategy_combinations(test_constant_details):
    # Basic parameter options for all strategies
    timeframes = ['M5', 'M15', 'M30', 'H1']
    atr_tps = np.arange(start=2, stop=4, step=0.25)
    atr_sls = np.arange(start=2, stop=4, step=0.25)
    trailings = [True, False]

    # Test-specific details
    strategy = test_constant_details['strategy']
    existing_result_count = test_constant_details['existing_result_count']

    if 'adx' in strategy:
        adx_cutoffs = np.arange(start=25, stop=55, step=5)
        combs = itertools.product(timeframes, atr_tps, atr_sls, trailings, adx_cutoffs)
    else:
        combs = itertools.product(timeframes, atr_tps, atr_sls, trailings)

    i = 0
    for comb in combs:
        # Continue past already completed tests
        if i < existing_result_count:
            i += 1
            continue

        # Collect parameters
        parameters = {
            'atr_tp': comb[1],
            'atr_sl': comb[2],
            'trailing_sl': comb[3]
        }
        if 'adx' in strategy:
            parameters['adx_cutoff'] = comb[4]

        test_dynamic_details = {
            'timeframe': comb[0],
            'parameters': parameters,
            'test_number': i,
        }
        test_dynamic_details.update(test_constant_details)

        test_strategy(test_dynamic_details)
        i += 1


def test_strategies_exhaustive(symbol, start, end, strategies):
    # Universal test details
    test_details = {
        'symbol': symbol,
        'start': start,
        'end': end,
    }

    for strategy in strategies:
        # Specify strategy
        test_details['strategy'] = strategy

        # Test all the combinations
        if 'model' in strategy:
            # Try with multiple models for each range
            date_ranges = get_model_date_ranges()
            # Test all models, each trained on one of the possible date ranges
            for date_range in date_ranges:
                model_start, model_end = date_range[0], date_range[1]

                file_path = os.path.join(os.path.dirname(__file__), f"tests/{start} to {end}/{symbol}-{start} to {end}-"
                                                                    f"{strategy}-model {date_range[0]} "
                                                                    f"to {date_range[1]}-optimization.csv")

                existing_result_count = len(pd.read_csv(file_path).index) if exists(file_path) else 0
                test_details['existing_result_count'] = existing_result_count
                test_details['model_start'] = model_start
                test_details['model_end'] = model_end
                test_details['file_path'] = file_path

                test_strategy_combinations(test_details)
        else:
            file_path = os.path.join(os.path.dirname(__file__), f"tests/{start} to {end}/{symbol}-{start} to {end}-"
                                                                f"{strategy}-optimization.csv")

            existing_result_count = len(pd.read_csv(file_path).index) if exists(file_path) else 0
            test_details['existing_result_count'] = existing_result_count
            test_details['file_path'] = file_path

            test_strategy_combinations(test_details)


def get_results(tests_directory):
    results = []
    for filename in os.listdir(tests_directory):
        f = os.path.join(tests_directory, filename)
        df = pd.read_csv(f)
        if '.txt' in f:
            continue
        best_row = df.iloc[df['net_profit'].idxmax()]
        r = {
            'strategy': best_row['strategy'],
            'timeframe': best_row['timeframe'],
            'trades': best_row['trades'],
            'net_profit': best_row['net_profit'],
            'win_ratio': round(best_row['win ratio'], 3),
            'atr_tp': best_row['atr_tp'],
            'atr_sl': best_row['atr_sl'],
            'trailing_sl': best_row['trailing_sl'],
        }
        if 'model ' in filename:
            i = filename.index('model ')
            model_range = filename[i + 6: i + 30]
            r['model_range'] = model_range
        results.append(r)
        print(results)

    with open(os.path.join(tests_directory, 'top_tests.txt'), 'a') as f:
        for r in results:
            f.write(f'{r}\n')


if __name__ == "__main__":
    start, end = '2012-10-01', '2022-10-21'
    test_dir = f'tests/{start} to {end}'
    # strategies = ['model', 'model_adx', 'model_chaikin', 'model_adx_chaikin']
    strategies = ['model_adx']
    symbol = 'EUR_USD'

    if not exists(test_dir):
        os.mkdir(test_dir)

    # test_strategies_exhaustive(symbol, start, end, strategies)

    # get_results(test_dir)

    file_path = os.path.join(os.path.dirname(__file__), f"tests/{start} to {end}/{symbol}-{start} to {end}-"
                                                        f"model_adx-model 2000-01-01 "
                                                        f"to 2022-10-22-optimization.csv")
    good_one = {
        'timeframe': 'M5',
        'parameters':
            {'atr_tp': 3.25,
             'atr_sl': 2.0,
             'trailing_sl': False,
             'adx_cutoff': 30},
        'test_number': 0,
        'symbol': 'EUR_USD',
        'start': start,
        'end': end,
        'strategy': 'model_adx',
        'existing_result_count': 0,
        'model_start': '2000-01-01',
        'model_end': '2022-10-22',
        'file_path': file_path,
        'balance': 1000}
    test_strategy(good_one)

