import datetime
import json
import os
import sys
import webbrowser
from os.path import exists
import backtrader as bt
import joblib
import pandas as pd
import tpqoa
import quantstats
from indicator_getters.indicators import IndicatorCalculator
from model_classes.MLModel import MLModel
from strategies import *


class Data(tpqoa.tpqoa):
    def __init__(self, instrument, timeframe, start, end):
        self.instrument = instrument
        self.timeframe = timeframe
        self.start = start
        self.end = end

        self.dirname = os.path.dirname(__file__)
        self.data_path = f"../forex_data/{self.instrument}_{self.start}_{self.end}_{self.timeframe}.csv"

        self.data = self.get_data()

    def get_data(self):
        print(
            f'Getting data: instrument={self.instrument}, start={self.start}, end={self.end}, timeframe={self.timeframe}')
        if exists(self.data_path):
            return pd.read_csv(self.data_path, parse_dates=['time'], index_col='time').dropna()
        else:
            api = tpqoa.tpqoa(os.path.join(self.dirname, '../oanda.cfg'))
            mid = api.get_history(instrument=self.instrument, start=self.start, end=self.end,
                                  granularity=self.timeframe,
                                  price='M')
            bid = api.get_history(instrument=self.instrument, start=self.start, end=self.end,
                                  granularity=self.timeframe,
                                  price='B')
            ask = api.get_history(instrument=self.instrument, start=self.start, end=self.end,
                                  granularity=self.timeframe,
                                  price='A')
            mid['bid'] = bid.c
            mid['ask'] = ask.c
            mid['spread'] = (bid.c - ask.c).to_frame()
            mid.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': "close"}, inplace=True)
            # mid["returns"] = np.log(mid.close / mid.close.shift(1))
            print('Data retrieved.')
            data = mid.dropna()
            data.to_csv(self.data_path)
            return data

    def get_features_data(self):
        print('Getting features...')
        feat_collector = IndicatorCalculator(self.data)
        feat_collector.calculate_features(features=['paverage2close', 'proc15close'])
        feat_collector.to_file(self.data_path)
        self.data = pd.read_csv(self.data_path, parse_dates=['time'], index_col='time').dropna()
        print('Features acquired.')


class Tester:

    def __init__(self, symbol, timeframe, start, end, logging=True):
        # self.params = params
        self.symbol = symbol
        self.timeframe = timeframe
        self.start = start
        self.end = end
        self.logging = logging

        self.data_getter = Data(symbol, timeframe, start, end)
        self.data = self.data_getter.data
        self.cerebro = None

        print(self.data.columns)

        # self.model = joblib.load("../EUR_USD-2000-01-01-2022-10-22-M5.joblib")

    def build_cerebro(self, optimizing=False):
        # Instantiate Cerebro engine
        if optimizing:
            cerebro = bt.Cerebro(optreturn=False)
        else:
            cerebro = bt.Cerebro()

        data = bt.feeds.PandasData(dataname=self.data)
        cerebro.adddata(data)

        cerebro.broker.setcash(5000)

        # cerebro.broker.set_slippage_fixed(1.5)

        # # Add writer to CSV
        # cerebro.addwriter(bt.WriterFile, csv=True, out='cerebro_tests/')

        return cerebro

    def test(self, strategy, params=None, logging=True):
        # Build cerebro
        self.cerebro = self.build_cerebro(optimizing=False)

        # Add strategy to Cerebro
        if params is None:
            params = {}
        params.update({'logging': logging})
        self.cerebro.addstrategy(strategy, **params)

        # Add metric analyzer
        self.cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')

        # Add trade analyzer
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

        # Test (returns list of strategies tested
        results = self.cerebro.run()

        strat = results[0]  # Get strategy

        analysis = strat.analyzers.trade_analyzer.get_analysis()

        self.parse_analysis(analysis)
        # joblib.dump(analysis, 'analysis3-12.joblib')

        portfolio_stats = strat.analyzers.getbyname('PyFolio')
        returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
        returns.index = returns.index.tz_convert(None)  # Makes returns DF compatible with quantstats

        # Save test report to file
        f = open('TEMP.txt', 'r')
        # strat_name = f.readlines()[0]
        f.close()
        os.remove('TEMP.txt')
        strat_name = ""

        filename = strat_name + f'_{self.symbol}_{self.timeframe}_{self.start} to {self.end}_' + '.html'
        quantstats.reports.html(returns, output=filename, title=strat_name)

        # Rename report file to something fitting
        if exists(filename):
            os.remove(filename)
        os.rename('quantstats-tearsheet.html', filename)

        return filename

    def optimize(self, strategy, param_ranges, logging=False):
        # Build cerebro
        self.cerebro = self.build_cerebro(optimizing=True)

        optstrategy_kwargs = {'strategy': strategy, 'logging': logging}
        optstrategy_kwargs.update(param_ranges)

        # Add strategy to optimize with params, unpacking kwargs for each parameter
        self.cerebro.optstrategy(**optstrategy_kwargs)

        # Run optimizations
        runs = self.cerebro.run()

        self.find_optimal(optimized_runs=runs)

    def get_strategy_params(self, strategy):
        params = dir(strategy.params)
        for param in list(params):
            if '_' in param or param in ['isdefault', 'notdefault']:
                params.remove(param)
        return params

    def find_optimal(self, optimized_runs):
        final_results_list = []
        for run in optimized_runs:
            for strategy in run:
                pl = round(strategy.broker.get_value() - 10000, 2)  # profit/loss
                result = []
                params = self.get_strategy_params(strategy)
                for param in params:
                    result.append(getattr(strategy.params, param))
                result.append(pl)
                final_results_list.append(result)

        # TODO: Consider sorting by TradeAnalyzer analysis results

        sort_by_profit = sorted(final_results_list, key=lambda x: x[-1], reverse=True)  # sort results by p/l
        print('FINAL RESULTS\n=================================================')
        for line in sort_by_profit[:10]:  # Get top 5
            print(line)

    def parse_analysis(self, analysis):
        results = {}
        for key in analysis.keys():
            results[key] = None
        for i in analysis:
            a = analysis[i]
            b = {}
            for key in a.keys():
                b[key] = a[key]
            results[i] = b

        final_results = {'total_trades': results['total']['total'],
                         'longest_win_streak': results['streak']['won']['longest'],
                         'longest_lose_streak': results['streak']['lost']['longest'],
                         'pnl_gross': [(k, v) for k, v in results['pnl']['gross'].items()],
                         'pnl_net': [(k, v) for k, v in results['pnl']['net'].items()],
                         'trades_won': results['won']['total'],
                         'won_pnl': [(k, v) for k, v in results['won']['pnl'].items()],
                         'trades_lost': results['lost']['total'],
                         'lost_pnl': [(k, v) for k, v in results['lost']['pnl'].items()],
                         'long_trades': results['long']['total'],
                         'long_pnl': [(k, v) for k, v in results['long']['pnl'].items()],
                         'short_trades': results['short']['total'],
                         'short_pnl': [(k, v) for k, v in results['short']['pnl'].items()]
                         }

        final_results['win_ratio'] = round(final_results['trades_won'] / final_results['total_trades'], 3)
        # final_results['bars_in_trade'] = [(k, v) for k, v in results['len'].items()]
        for r in final_results:
            rr = final_results[r]
            if isinstance(rr, list):
                new_list = []
                for pair in rr:
                    a = round(pair[0], 3) if isinstance(pair[0], float) else pair[0]
                    b = round(pair[1], 3) if isinstance(pair[1], float) else pair[1]
                    new_list.append((a, b))
                final_results[r] = new_list

        for r in final_results:
            print(r, final_results[r])
            print()

        with open("test.json", "w") as f:
            json.dump(final_results, f)


if __name__ == "__main__":
    symbol, timeframe, start, end = "EUR_USD", "M5", "2022-01-01", "2022-11-04"
    tester = Tester(symbol, timeframe, start, end)

    # ma_opt = {'fast': range(5, 20), 'slow': range(50, 100)}

    result_file = tester.test(Trend, logging=False)
    # # # Open report in chrome
    chrome = webbrowser.get("C:/Program Files/Google/Chrome/Application/chrome.exe %s")
    chrome.open('file://' + os.path.realpath(result_file))
    # tester.cerebro.plot(style='bars', fmt_x_data='%Y-%b-%d %H:%M')


    # ml = MLModel("EUR_USD", "M5", "2000-01-01", "2022-10-22", features=['paverage2close', 'proc15close'])
    # ml.fit_model()
    # # print('Model acquired.')
    # model_path = f"models/{timeframe}/{symbol}-{start}-{end}-{timeframe}.joblib"
    # joblib.dump(ml.model, model_path)