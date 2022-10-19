import sys

from IterativeBase import *


class IterativeBacktest(IterativeBase):
    ''' Class for iterative (event-driven) backtesting of trading strategies.
    '''

    # helper method
    def go_long(self, bar, units=None, amount=None):
        if self.position == -1:
            self.buy_instrument(bar, units=-self.units)  # if short position, go neutral first
        if units:
            self.buy_instrument(bar, units=units)
        elif amount:
            if amount == "all":
                amount = self.current_balance
            self.buy_instrument(bar, amount=amount)  # go long

    # helper method
    def go_short(self, bar, units=None, amount=None):
        if self.position == 1:
            self.sell_instrument(bar, units=self.units)  # if long position, go neutral first
        if units:
            self.sell_instrument(bar, units=units)
        elif amount:
            if amount == "all":
                amount = self.current_balance
            self.sell_instrument(bar, amount=amount)  # go short

    def test_strategy(self, features):
        # reset
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.current_balance = self.initial_balance  # reset initial capital
        self.get_data()  # reset dataset

        # sma crossover strategy
        for bar in range(len(self.data) - 1):  # all bars (except the last bar)
            bar_row = self.data[features].iloc[bar].to_frame().T
            prediction = self.model.predict(bar_row)[0]
            if prediction == 1:  # signal to go long
                if self.position in [0, -1]:
                    self.go_long(bar, amount="all")  # go long with full amount
                    self.position = 1  # long position
            elif prediction == -1:  # signal to go short
                if self.position in [0, 1]:
                    self.go_short(bar, amount="all")  # go short with full amount
                    self.position = -1  # short position
            elif prediction == 0:
                self.close_pos(bar)

        return self.close_pos(bar + 1)  # close position at the last bar


if __name__ == "__main__":
    ib = IterativeBacktest('EUR_USD', '2021-01-01', '2022-06-28', 'M5', 1000, use_spread=True)
    ib.test_strategy()

