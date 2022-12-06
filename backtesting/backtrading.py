import itertools
import json
import multiprocessing
import os
import random
import sys
import time
import numpy
import pandas as pd
import quantstats
import tpqoa
import logging
import coloredlogs
from timeit import default_timer as timer
from playsound import playsound
from os.path import exists
from indicator_getters.indicators import IndicatorCalculator
from strategies import *

N_CORES = multiprocessing.cpu_count()  # Number of logical cores on pc

LOGGER = None


def get_logger():
    today = datetime.datetime.today()
    year, month, day = today.strftime('%Y'), today.strftime('%B'), today.strftime('%d')

    log_paths = ["logs", f"logs/{year}", f"logs/{year}/{month}", f"logs/{year}/{month}/{day}"]
    log_folder = f"logs/{year}/{month}/{day}"

    # Make log directories
    for lp in log_paths:
        try:
            os.mkdir(lp)
        except FileExistsError:
            pass

    logs_today = len(os.listdir(log_folder))

    log_path_full = f'logs/{year}/{month}/{day}/FULL_CONSOLE_LOG_{logs_today}.log'

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


def build_cerebro(data, optimizing=False):
    # Instantiate Cerebro engine
    if optimizing:
        cerebro = bt.Cerebro(optreturn=False)
    else:
        cerebro = bt.Cerebro()

    data = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data)

    cerebro.broker.setcash(5000)

    # Add trade analyzer
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

    # Add system analyzer
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')

    # cerebro.broker.set_slippage_fixed(1.5)

    # # Add writer to CSV
    # cerebro.addwriter(bt.WriterFile, csv=True, out='cerebro_')

    return cerebro


def combinate(dict_of_items):
    LOGGER.info("Combinating...")
    params, values = zip(*dict_of_items.items())
    possible_combinations = sum(1 for i in itertools.product(*values))

    # If more than 1 million combos, limit the number of combinations made
    if possible_combinations > 1000000:
        LOGGER.info(f'{possible_combinations} possible combinations, reducing to one million...')
        sliced = itertools.islice(itertools.product(*values), 1000000)
        all_combos = [dict(zip(params, val)) for val in sliced]
    else:
        all_combos = [dict(zip(params, val)) for val in itertools.product(*values)]
    LOGGER.info(f"Combinated ({len(all_combos)} combinations).")
    return all_combos


def fill_queue(q, items):
    # Fill queue with combos for workers
    while len(items) > 0:
        combo = items[-1]
        q.put(combo)
        items.remove(combo)
    # Add stop signal to signal end of work
    q.put(None)


def remove_tested_combos(tested_combos, all_combos_subset, out_q):
    for param_combo in all_combos_subset.copy():
        if str(param_combo) in tested_combos:
            all_combos_subset.remove(param_combo)
    out_q.put(all_combos_subset)


def str_to_dict(string):
    return json.loads(string.replace("'", '"'))


def get_strategy_params(strategy):
    params = dir(strategy.params)
    for param in list(params):
        if param[0] == '_' or param in ['isdefault', 'notdefault']:
            params.remove(param)
    return params


def get_param_combos(params_to_test, test_path):
    all_combos = combinate(params_to_test)

    # Get any existing tests
    if exists(test_path):
        param_combos = []
        existing_csv = pd.read_csv(test_path, index_col=0)
        tested_combos = pd.Series(existing_csv['params'])
        tested_combos = [str(x) for x in tested_combos]

        # Remove already tested combos
        LOGGER.info('Removing tested combos...')
        final_q = multiprocessing.Queue()
        removers_done = 0
        combo_subsets = list(chunks(all_combos, N_CORES - 2))
        removers = [multiprocessing.Process(target=remove_tested_combos,
                                            args=(tested_combos, combo_subsets[i], final_q))
                    for i in range(0, N_CORES - 2)]

        for remover in removers:
            remover.start()

        while removers_done < N_CORES - 2:
            param_combos = param_combos + final_q.get()
            removers_done += 1
    else:
        param_combos = all_combos

    # TODO: See why combos not being removed?
    LOGGER.info(f'Removed {len(all_combos) - len(param_combos)} combinations.')
    return param_combos


def parse_analysis(analysis):
    results = {}
    for key in analysis.keys():
        results[key] = None
    for i in analysis:
        a = analysis[i]
        bb = {}
        for key in a.keys():
            bb[key] = a[key]
        results[i] = bb
    try:
        final_results = {'total_trades': results['total']['total'],
                         'longest_win_streak': results['streak']['won']['longest'],
                         'longest_lose_streak': results['streak']['lost']['longest'],
                         'pnl_gross_total': results['pnl']['gross']['total'],
                         'pnl_gross_avg': results['pnl']['gross']['average'],
                         'pnl_net_total': results['pnl']['net']['total'],
                         'pnl_net_avg': results['pnl']['net']['average'],
                         'trades_won': results['won']['total'],
                         'won_pnl_total': results['won']['pnl']['total'],
                         'won_pnl_avg': results['won']['pnl']['average'],
                         'trades_lost': results['lost']['total'],
                         'lost_pnl_total': results['lost']['pnl']['total'],
                         'lost_pnl_avg': results['lost']['pnl']['average'],
                         'long_trades': results['long']['total'],
                         'long_pnl_total': results['long']['pnl']['total'],
                         'long_pnl_avg': results['long']['pnl']['average'],
                         'short_trades': results['short']['total'],
                         'short_pnl_total': results['short']['pnl']['total'],
                         'short_pnl_avg': results['short']['pnl']['average'],
                         }

        final_results['win_ratio'] = round(final_results['trades_won'] / final_results['total_trades'], 3)
    except KeyError as e:
        print(e)
        final_results = {}

    for key, value in final_results.items():
        if isinstance(value, float):
            final_results[key] = round(value, 3)

    return final_results


def test_combo(cerebro, combo):
    args = combo.copy()
    args.update({"logging": False})

    # Update strat params (instead of using addstrategy, which bloats memory)
    cerebro.strats[0][0] = (cerebro.strats[0][0][0], (), args)

    # Test combo
    run = cerebro.run()
    strategy_result = run[0]

    # Get test metrics
    metrics = strategy_result.analyzers.trade_analyzer.get_analysis()
    metrics_parsed = parse_analysis(metrics)
    sqn_analysis = strategy_result.analyzers.sqn.get_analysis()

    # Create results dict for later use as a df
    combo_str = str(combo)
    result_dict = {'params': combo_str, 'sqn': round(sqn_analysis['sqn'], 3)}
    result_dict.update(metrics_parsed)

    return result_dict


def worker_test_combos(in_q, out_q, strategy, data):
    # Build cerebro
    cerebro = build_cerebro(optimizing=False, data=data)
    cerebro.addstrategy(strategy)

    # While queue isn't empty, look for more combos to do
    while not in_q.empty():
        # Get next combo
        combo = in_q.get()

        # Reached end of combos, put stop signal back in queue for next worker and end process
        if combo is None:
            in_q.put(None)
            return

        # logger.info(f"Worker (Core {core_num}) testing {combo}")
        result = test_combo(cerebro, combo)
        out_q.put(result)


def chunks(lst, n):
    """Yield n number of striped chunks from list."""
    for i in range(0, n):
        yield lst[i::n]


class Data(tpqoa.tpqoa):

    def __init__(self):
        self.test_path = None
        self.symbol = symbol
        self.timeframe = timeframe
        self.start = start
        self.end = end

        self.dirname = os.path.dirname(__file__)
        self.data_path = f"forex_data/{self.symbol}_{self.start}_{self.end}_{self.timeframe}.csv"

        self.data = self.get_data()

    def get_data(self):
        LOGGER.info(
            f'(Data Handler) Getting data: instrument={self.symbol}, start={self.start}, end={self.end}, timeframe={self.timeframe}')
        if exists(self.data_path):
            return pd.read_csv(self.data_path, parse_dates=['time'], index_col='time').dropna()
        else:
            api = tpqoa.tpqoa(os.path.join(self.dirname, '../oanda.cfg'))
            mid = api.get_history(instrument=self.symbol, start=self.start, end=self.end,
                                  granularity=self.timeframe,
                                  price='M')
            bid = api.get_history(instrument=self.symbol, start=self.start, end=self.end,
                                  granularity=self.timeframe,
                                  price='B')
            ask = api.get_history(instrument=self.symbol, start=self.start, end=self.end,
                                  granularity=self.timeframe,
                                  price='A')
            mid['bid'] = bid.c
            mid['ask'] = ask.c
            mid['spread'] = (bid.c - ask.c).to_frame()
            mid.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': "close"}, inplace=True)
            # mid["returns"] = np.log(mid.close / mid.close.shift(1))
            LOGGER.info('(Data Handler) Data retrieved.')
            data = mid.dropna()
            data.to_csv(self.data_path)
            return data

    def get_features_data(self):
        LOGGER.info('(Data Handler) Getting features...')
        feat_collector = IndicatorCalculator(self.data)
        feat_collector.calculate_features(features=['paverage2close', 'proc15close'])
        feat_collector.to_file(self.data_path)
        self.data = pd.read_csv(self.data_path, parse_dates=['time'], index_col='time').dropna()
        LOGGER.info('(Data Handler) Features acquired.')

    def test_dir(self):
        # Make necessary directories
        sym_dir = f"tests/{self.symbol}"
        tf_dir = sym_dir + f"/{self.timeframe}"
        dates_dir = tf_dir + f"/{self.start} to {self.end}"
        for d in [sym_dir, tf_dir, dates_dir]:
            if not exists(d):
                os.mkdir(d)

        return dates_dir

    def save_results(self, results):
        # Save results to a df
        LOGGER.info(f'(Data Handler) Saving {len(results)} test results...')
        results_df = None
        for result in results.copy():
            curr_df = pd.DataFrame(data=result, index=[0])
            results_df = curr_df if results_df is None else results_df.append(curr_df, ignore_index=True)
            results.pop(results.index(result))

        # Merge results w/ existing test file, or create a new one
        if exists(self.test_path):
            existing_csv = pd.read_csv(self.test_path, index_col=0)
            merged = existing_csv.append(results_df, ignore_index=True)
            merged = merged.sort_values('sqn', ascending=False)
            merged = merged.drop_duplicates(keep='first')
            merged.to_csv(self.test_path)
        else:
            results_df = results_df.sort_values('sqn', ascending=False)
            results_df.to_csv(self.test_path)
        LOGGER.info('(Data Handler) Save complete.')

    def set_test_path(self, test_path):
        LOGGER.info(f"(Data Handler) Setting test path: {test_path}")
        self.test_path = test_path

    def final_sort_tests(self, sortby):
        # Sort by SQN usually
        final_df = pd.read_csv(self.test_path, index_col=0)
        final_df = final_df.sort_values(sortby, ascending=False)
        final_df.to_csv(self.test_path)


class GeneticOptimizer:
    def __init__(self, params, evolution_rules):

        LOGGER.info(f"=== GENETIC OPTIMIZATION ===")

        self.population_size = evolution_rules['population_size']
        self.generations = evolution_rules['generations']
        self.gen_ancestor_percentage = evolution_rules['gen_ancestor_percentage']
        self.fitness_target = evolution_rules['fitness_target']
        self.do_shakeup = evolution_rules['do_shakeup']
        self.shakeup_percentage = evolution_rules[
            'shakeup_percentage']  # Percentage of each gen to be randomly selected from all combos
        self.evolving = False

        self.all_combos = get_param_combos(params, data_handler.test_path)
        random.shuffle(self.all_combos)
        self.all_combos_set = set(str(x) for x in self.all_combos)

        self.all_previous_solutions = set()

        LOGGER.info(f"=== EVOLUTION PARAMETERS ===")
        LOGGER.info(evolution_rules)

    # Initialize the ground-zero population (randomly select from all combos)
    def initialize_population(self, population_size):
        LOGGER.info("Spawning Generation Zero...")

        population = []
        while len(population) != population_size and len(self.all_combos) > 0:
            combo = random.choice(self.all_combos)
            # self.all_combos.remove(combo)  # Remove combo from possible combo list since it's going to be tested
            population.append(combo)

        LOGGER.info("Generation Zero Spawned.")
        return population

    def fill_population_queue(self, population_queue, items):
        # Fill queue with combos for workers
        while len(items) > 0:
            combo = items[-1]
            population_queue.put(combo)
            items.remove(combo)

    def worker_test_combos_ga(self, population_queue, population_results, strategy, data):
        while self.evolving:
            worker_test_combos(population_queue, population_results, strategy, data)

    # Evaluate the fitness of this combo
    def get_fitness(self, result):
        sqn = result['sqn']
        fitness_value = sqn - self.fitness_target
        if fitness_value == 0:
            # The best solution for target
            return 86753090000000000000000, sqn, result
        else:
            # Smaller result (closer to zero) yields a higher rank value
            return sqn, result
            # return round(abs(1 / fitness_value), 3), sqn, result

    # Gets fitness results of each combo in population from worker threads
    def calculate_population_fitness(self, population_test_results):
        fitnesses = []
        tests_done = 0
        while tests_done != self.population_size:
            # Look for fitness results from the workers
            pop_result = population_test_results.get()
            fitness = self.get_fitness(pop_result)
            fitnesses.append(fitness)
            tests_done += 1
            sys.stdout.write('\r' + f"{tests_done}/{self.population_size} solutions fitted.")
        sys.stdout.write('\r' + "")

        return fitnesses

    # Randomly combinate from solutions
    def get_all_possible_offspring(self, ancestors):
        ranges = {}
        gene_names = ancestors[0].keys()

        # Get each ancestor's gene values
        for gene in gene_names:
            ranges[gene] = []
            for ancestor in ancestors:
                gene_val = ancestor[gene]
                if gene_val not in ranges[gene]:
                    ranges[gene].append(gene_val)

        possible_offspring = combinate(ranges)

        return possible_offspring

    def spawn_next_generation(self, ancestors):

        # See if we need to toss in some random solutions
        if self.do_shakeup:
            solutions_from_genes = round(self.population_size * (1 - self.shakeup_percentage))
        else:
            solutions_from_genes = self.population_size

        # Use a set for faster lookup in while loop (same for incoming all_previous_solutions)
        next_generation = []

        # Get all possible randomly combinated offspring from ancestors
        possible_offspring_list = self.get_all_possible_offspring(ancestors)
        possible_offspring = set(str(o) for o in possible_offspring_list)

        # Filter out offspring used in previous generations
        unused_offspring = possible_offspring.difference(self.all_previous_solutions)
        # TODO
        print(len(possible_offspring), len(unused_offspring))

        # Select from random possible offsprings
        # Entity w/ key-value == gene-value
        for _ in range(0, solutions_from_genes):
            new_descendant = random.choice(tuple(unused_offspring))
            # Add new, not-previously-used entity
            next_generation.append(str_to_dict(new_descendant))

            # Remove it from offspring options
            unused_offspring.remove(new_descendant)

        # Throw some random individuals into the next generation to freshen things (if do_shakeup says to)
        if self.do_shakeup:
            shakeup_count = round(self.population_size * self.shakeup_percentage)
            # Add random sample to next generation
            next_generation.extend(random.sample(self.all_combos, shakeup_count))
            LOGGER.info(f"Tossed in {shakeup_count} random solutions for next generation.")

        return next_generation

    # Go through generations (combos=initial population)
    def evolve(self, strategy):
        LOGGER.info("=== BEGINNING EVOLUTION ===")
        self.evolving = True
        LOGGER.info(f"- Generations: {self.generations}")
        LOGGER.info(f"- Population Size: {self.population_size}")

        generation_zero = self.initialize_population(self.population_size)

        # Add current generation to list of tried solutions (to not have descendants that are repeats of an ancestor)
        self.all_previous_solutions.update(str(x) for x in generation_zero)

        population_queue, population_results = multiprocessing.Queue(maxsize=100), multiprocessing.Queue()

        workers = [
            multiprocessing.Process(target=self.worker_test_combos_ga,
                                    args=(population_queue, population_results, strategy, data_handler.data))
            for core in range(0, N_CORES - 2)]

        queue_filler = multiprocessing.Process(target=self.fill_population_queue,
                                               args=(population_queue, generation_zero))
        queue_filler.start()

        for worker in workers:
            worker.start()

        top_10 = []
        for gen in range(self.generations):

            gen_start = timer()

            LOGGER.info(f"=== Generation {gen}  ===")

            # Use workers to calculate fitness values of each member of the initial population
            ranked_solutions = self.calculate_population_fitness(population_test_results=population_results)

            # Sort all solutions by fitness value
            ranked_solutions.sort(key=lambda x: x[0])
            ranked_solutions.reverse()

            # Add current top ten to overall top ten, then find new top 10
            for top in ranked_solutions[:10]:
                if top not in top_10:
                    top_10.append(top)
            top_10.sort(key=lambda x: x[0])
            top_10.reverse()
            top_10 = top_10[:10]

            # Create new generation from the good ancestors
            LOGGER.info("Spawning next generation...")
            # Get top N% of the generation to use for gene extraction
            num_ancestors = round(self.population_size * self.gen_ancestor_percentage)
            next_gen_ancestors = [str_to_dict(a[1]['params']) for a in ranked_solutions[:num_ancestors]]
            next_generation = self.spawn_next_generation(next_gen_ancestors)
            LOGGER.info("Spawned.")

            # Record usage of next generation
            self.all_previous_solutions.update(str(x) for x in next_generation)

            # Start filling with new population
            queue_filler = multiprocessing.Process(target=self.fill_population_queue,
                                                   args=(population_queue, next_generation))
            queue_filler.start()

            LOGGER.info(f"Current Top 10 SQNs: {[top[0] for top in top_10]}")

            # Save generation to results file
            data_handler.save_results([r[1] for r in ranked_solutions])
            data_handler.final_sort_tests('sqn')

            LOGGER.info(f"Generation Time Elapsed: {datetime.timedelta(seconds=(timer() - gen_start))}")

        # Evolution finished, signal threads to end
        self.evolving = False
        population_queue.put(None)
        queue_filler.kill()
        queue_filler.join()
        for worker in workers:
            worker.kill()
            worker.join()
        LOGGER.info("Terminated all workers.")


class Tester:

    def __init__(self, show_logs=True):
        # self.params = params
        self.dates_dir = None
        self.symbol = symbol
        self.timeframe = timeframe
        self.start = start
        self.end = end
        self.logging = show_logs
        self.cerebro = None

    def opt_test_mp(self, strategy, params_to_test, randomize=False):

        # Get all possible param combinations
        param_combos = get_param_combos(params_to_test=params_to_test, test_path=data_handler.test_path)

        # Test parameter combinations
        LOGGER.info(f"{len(param_combos)} combinations to test")

        if randomize:
            random.shuffle(param_combos)

        # Prepare for multiprocessing
        # (set in_q maximum size to prevent bloating while filling queue and waiting for workers)
        # (queue.put() will pause until space becomes available, i.e. queue.get() called by worker)
        in_q, out_q = multiprocessing.Queue(maxsize=100), multiprocessing.Queue()

        # Create workers to do the work
        # (create number of workers == to number of logical cores minus two (one for main thread, one for filling queue))
        workers = [multiprocessing.Process(target=worker_test_combos, args=(in_q, out_q, strategy, data_handler.data))
                   for core in range(0, N_CORES - 2)]

        # Create a worker to fill the queue...
        queue_filler = multiprocessing.Process(target=fill_queue, args=(in_q, param_combos))
        # ...and start filling it
        queue_filler.start()

        # Start working
        for worker in workers:
            worker.start()

        # Continue looping until all combos tested
        combo_count = len(param_combos)
        results = []
        tests_done = 0
        num_to_save = 100 if combo_count >= 1000 else combo_count
        while tests_done != combo_count:

            if tests_done % 100 == 0:
                time.sleep(1)

            # On each loop around, check if any results have been produced
            # (and pause the thread til a result has been produced)
            try:
                results.append(out_q.get())
                tests_done += 1
                sys.stdout.write('\r' + f"{tests_done}/{combo_count} combos tested.")
                # sys.stdout.write('\r' + "")
                # After 10% of tests done, save results
                if tests_done % num_to_save == 0 and tests_done != 0:
                    LOGGER.info(f"\nSaving previous {num_to_save} tests...")

                    data_handler.save_results(results)
            except:
                pass

        LOGGER.info('Finished.')

        # Save last results
        LOGGER.info("Saving leftovers...")
        data_handler.save_results(results)
        LOGGER.info("Leftovers Saved.")

        # Sort by sqn
        data_handler.final_sort_tests("sqn")

    def test(self, strategy, params=None, show_logs=False):
        # Build cerebro
        self.cerebro = build_cerebro(optimizing=False, data=data_handler.data)

        # Add metric analyzer
        self.cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')

        # Add strategy to Cerebro
        if params is None:
            params = {}
        params.update({'logging': show_logs})
        self.cerebro.addstrategy(strategy, **params)

        # Test (returns list of strategies tested
        results = self.cerebro.run()
        strat = results[0]  # Get strategy

        analysis = strat.analyzers.trade_analyzer.get_analysis()
        sqn_analysis = strat.analyzers.sqn.get_analysis()
        metrics_parsed = parse_analysis(analysis)

        portfolio_stats = strat.analyzers.getbyname('PyFolio')
        returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
        returns.index = returns.index.tz_convert(None)  # Makes returns DF compatible with quantstats

        # Results
        combo_str = str(params)
        result_dict = {'params': combo_str, 'sqn': round(sqn_analysis['sqn'], 3)}
        result_dict.update(metrics_parsed)

        LOGGER.info(result_dict)

        # Make test directory
        params.pop('logging')
        parameters, values = zip(*params.items())
        test_dir = f"tests/{self.symbol}_{self.timeframe}_{self.start} to {self.end}"
        test_path = test_dir + f"/{parameters}.csv"
        if not exists(test_dir):
            os.mkdir(test_dir)
        curr_df = pd.DataFrame(data=result_dict, index=[0])
        # Save to CSV
        if exists(test_path):
            existing_csv = pd.read_csv(test_path, index_col=0)
            merged = existing_csv.append(curr_df, ignore_index=True)
            merged.to_csv(test_path)
        else:
            curr_df.to_csv(test_path)

        strat_name = ""

        filename = strat_name + f'_{self.symbol}_{self.timeframe}_{self.start} to {self.end}_' + '.html'
        quantstats.reports.html(returns, output=filename, title=strat_name)

        # Rename report file to something fitting
        if exists(filename):
            os.remove(filename)
        os.rename('quantstats-tearsheet.html', filename)

        return filename


def do_opt(param_ranges, strat):
    # Set path for tests
    param_names, ___ = zip(*param_ranges.items())
    path = data_handler.test_dir() + f"/{param_names}.csv"
    data_handler.set_test_path(path)

    tester = Tester()
    tester.opt_test_mp(strategy=strat, params_to_test=param_ranges, randomize=True)


def do_evolve(param_ranges, evolve_rules, strat):
    # Set path for tests
    param_names, ___ = zip(*param_ranges.items())
    path = data_handler.test_dir() + f"/{param_names}.csv"
    data_handler.set_test_path(path)

    darwin = GeneticOptimizer(params=param_ranges, evolution_rules=evolve_rules)
    darwin.evolve(strategy=strat)


if __name__ == "__main__":
    # TODO: Build up one piece at a time (baseline, confirmation, etc)
    # Lookup TP/SL from NNFX

    clock_start = timer()

    symbol, timeframe, start, end = "AUD_NZD", "M30", "2022-01-01", "2022-11-04"

    LOGGER = get_logger()
    LOGGER.info((symbol, timeframe, start, end))

    # Tester and GeneticOptimizer get data from this
    data_handler = Data()

    # Params to optimize (FOR TREND)
    param_opt_ranges = {
        'atr': numpy.arange(4, 32, step=2),
        'atr_sl': numpy.arange(1, 4.25, step=0.25),
        'atr_tp': numpy.arange(1, 4.25, step=0.25),
        'rsi': numpy.arange(4, 30, step=2),
        'adx': numpy.arange(4, 30, step=2),
        'adx_cutoff': numpy.arange(25, 40, step=5),
        'vidya': numpy.arange(20, 100, step=2)
    }

    evolution_parameters = {
        "population_size": 1000,  # Minimum pop size is 100 for proper result formatting
        "generations": 10,
        "gen_ancestor_percentage": 0.1,  # Use top n percent of generation for gene extraction
        "fitness_target": 7,
        "do_shakeup": False,  # Do or don't throw in some random entities each generation
        "shakeup_percentage": 0.05
    }

    # differential_evolution()

    for b in [['sma'], ['ema'], ['ama']]:
        baseline_param_ranges = {
            'atr': numpy.arange(4, 32, step=2),
            'atr_sl': numpy.arange(1, 4.25, step=0.25),
            'atr_tp': numpy.arange(1, 4.25, step=0.25),
            'baseline': b,
            b[0]: numpy.arange(10, 52, step=2)
        }
        do_evolve(param_ranges=baseline_param_ranges, evolve_rules=evolution_parameters, strat=NNFX)

    playsound('C:\\Users\\Nick\\Documents\\GitHub\\AlgoTrader\\guh_huh.mp3')

    LOGGER.info("\n=== TIME ELAPSED ===")
    LOGGER.info(datetime.timedelta(seconds=timer() - clock_start))

    sys.exit()

    # tester.test(Trend, test, logging=False)
    # result_file = tester.test(Trend, logging=False)
    # # # Open report in chrome
    # chrome = webbrowser.get("C:/Program Files/Google/Chrome/Application/chrome.exe %s")
    # chrome.open('file://' + os.path.realpath(result_file))
