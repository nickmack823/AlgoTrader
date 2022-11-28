import itertools
import multiprocessing
import os
import random
import time
import numpy
import quantstats
import tpqoa
import logging
from playsound import playsound
from os.path import exists
from indicator_getters.indicators import IndicatorCalculator
from strategies import *

n_cores = multiprocessing.cpu_count()  # Number of logical cores on pc
logger = None


def get_logger():
    today = datetime.datetime.today()
    year, month, day = today.strftime('%Y'), today.strftime('%B'), today.strftime('%d')

    log_paths = ["logs", f"logs/{year}", f"logs/{year}/{month}", f"logs/{year}/{month}/{day}"]
    log_folder = f"logs/{year}/{month}/{day}"

    # Make log directories
    for lp in log_paths:
        try:
            os.mkdir(lp)
        except FileExistsError as e:
            pass

    logs_today = len(os.listdir(log_folder))

    log_path_full = f'logs/{year}/{month}/{day}/FULL_CONSOLE_LOG_{logs_today}.log'

    # Logging
    logger = logging.getLogger()
    logger.setLevel('INFO')

    file_log_handler = logging.FileHandler(log_path_full)
    logger.addHandler(file_log_handler)

    stderr_log_handler = logging.StreamHandler()
    logger.addHandler(stderr_log_handler)

    # nice output format
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_log_handler.setFormatter(formatter)
    stderr_log_handler.setFormatter(formatter)

    return logger


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


def get_strategy_params(strategy):
    params = dir(strategy.params)
    for param in list(params):
        if param[0] == '_' or param in ['isdefault', 'notdefault']:
            params.remove(param)
    return params


# Combinate params and remove already tested ones
def get_param_combos(params_to_test, test_path):
    all_combos = combinate(params_to_test)

    # Get any existing tests
    if exists(test_path):
        param_combos = []
        existing_csv = pd.read_csv(test_path, index_col=0)
        tested_combos = pd.Series(existing_csv['params'])
        tested_combos = [str(x) for x in tested_combos]

        # Remove already tested combos
        logger.info('Removing tested combos...')
        final_q = multiprocessing.Queue()
        removers_done = 0
        combo_subsets = list(chunks(all_combos, n_cores - 2))
        removers = [multiprocessing.Process(target=remove_tested_combos,
                                            args=(tested_combos, combo_subsets[i], final_q))
                    for i in range(0, n_cores - 2)]

        for remover in removers:
            remover.start()

        while removers_done < n_cores - 2:
            param_combos = param_combos + final_q.get()
            removers_done += 1
    else:
        param_combos = all_combos

    # TODO: See why combos not being removed?
    logger.info(f'Removed {len(all_combos) - len(param_combos)} combinations.')
    return param_combos


def parse_analysis(analysis):
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

    for key, value in final_results.items():
        if isinstance(value, float):
            final_results[key] = round(value, 3)

    return final_results


def test_combo(combo, cerebro):
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
    print(result_dict)
    return result_dict


def worker_test_combos(in_q, out_q, data, strategy):
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
        result = test_combo(combo, cerebro)
        out_q.put(result)


def chunks(lst, n):
    """Yield n number of striped chunks from list."""
    for i in range(0, n):
        yield lst[i::n]


def combinate(dict_of_items):
    logger.info("Combinating...")
    params, values = zip(*dict_of_items.items())
    possible_combinations = sum(1 for i in itertools.product(*values))

    # If more than 1 million combos, limit the number of combinations made
    if possible_combinations > 1000000:
        sliced = itertools.islice(itertools.product(*values), 1000000)
        all_combos = [dict(zip(params, val)) for val in sliced]
    else:
        all_combos = [dict(zip(params, val)) for val in itertools.product(*values)]
    logger.info(f"Combinated ({len(all_combos)} combinations).")
    return all_combos


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
        logger.info(
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
            logger.info('(Data Handler) Data retrieved.')
            data = mid.dropna()
            data.to_csv(self.data_path)
            return data

    def get_features_data(self):
        logger.info('(Data Handler) Getting features...')
        feat_collector = IndicatorCalculator(self.data)
        feat_collector.calculate_features(features=['paverage2close', 'proc15close'])
        feat_collector.to_file(self.data_path)
        self.data = pd.read_csv(self.data_path, parse_dates=['time'], index_col='time').dropna()
        logger.info('(Data Handler) Features acquired.')

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
        logger.info(f'(Data Handler) Saving {len(results)} test results...')
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
            merged.to_csv(self.test_path)
        else:
            results_df.to_csv(self.test_path)
        logger.info('(Data Handler) Save complete.')

    def set_test_path(self, path):
        logger.info(f"(Data Handler) Setting test path: {path}")
        self.test_path = path

    def final_sort_tests(self, sortby):
        # Sort by SQN usually
        final_df = pd.read_csv(self.test_path, index_col=0)
        final_df = final_df.sort_values(sortby, ascending=False)
        final_df.to_csv(self.test_path)


class GeneticOptimizer:
    def __init__(self, strategy, params, evolution_rules):

        self.data = data_handler.data
        self.strategy = strategy
        self.params = params
        self.population_size = evolution_rules['population_size']
        self.generations = evolution_rules['generations']
        self.gen_ancestor_percentage = evolution_rules['gen_ancestor_percentage']
        self.fitness_target = evolution_rules['fitness_target']
        self.shakeup = evolution_rules['shakeup']
        self.shakeup_percentage = evolution_rules[
            'shakeup_percentage']  # Percentage of each gen to be randomly selected from all combos
        self.evolving = False

        # Get test path
        params, values = zip(*self.params.items())
        path = data_handler.test_dir() + f"/{params}.csv"
        data_handler.set_test_path(path)

        self.all_combos = get_param_combos(self.params, path)
        random.shuffle(self.all_combos)

    # Initialize the ground-zero population (randomly select from all combos)
    def initialize_population(self, population_size):
        logger.info("Spawning Generation Zero...")

        population = []
        for n in range(population_size):
            combo = random.choice(self.all_combos)
            population.append(combo)
            # self.all_combos.remove(combo)

        logger.info("Generation Zero Spawned.")
        return population

    def fill_population_queue(self, population_queue, items):
        # Fill queue with combos for workers
        while len(items) > 0:
            combo = items[-1]
            population_queue.put(combo)
            items.remove(combo)

    # Evaluate the fitness of this combo
    def get_fitness(self, combo):
        results = test_combo(combo, self.data, self.strategy)
        sqn = results['sqn']
        fitness_value = sqn - self.fitness_target
        if fitness_value == 0:
            # The best solution for target
            return 86753090000000000000000, sqn, results
        else:
            # Smaller result (closer to zero) yields a higher rank value
            return round(abs(1 / fitness_value), 3), sqn, results

    # This is run by the worker threads
    def get_fitness_of_queue(self, population_queue, fitness_results):
        # Look for work to do while evolving
        while self.evolving:

            # Get fitness of members of the population
            while not population_queue.empty():
                combo = population_queue.get()
                if combo is None:
                    population_queue.put(None)
                    return

                # Send out the result
                fitness_results.put(self.get_fitness(combo))

    # Gets fitness results of each combo in population from worker threads
    def calculate_population_fitness(self, fitness_results):
        results = []
        tests_done = 0
        while tests_done != self.population_size:
            # Look for fitness results from the workers
            results.append(fitness_results.get())
            tests_done += 1
            sys.stdout.write('\r' + f"{tests_done}/{self.population_size} fitted.")
        sys.stdout.write('\r' + "")

        return results

    def spawn_original_descendant(self, ancestors, all_previous_solutions):

        # Extract "genes" (parameter values) from the ancestors
        genes = []
        # Extract genes (params) and their values from the top solutions
        for solution in ancestors:
            params = solution[2]
            for key, value in params.items():
                genes.append((key, value))

        # Get name of each gene
        gene_names = []
        gene_names = [g[0] for g in genes if g not in gene_names]

        # Entity w/ key-value == gene-value
        new_descendant = {}
        while new_descendant == {} or new_descendant in all_previous_solutions:
            # For each gene we're using, get a random value from incoming set of genes
            for gene in gene_names:

                # Get random gene-value tuple
                random_gene = random.choice(genes)

                # Make sure the random selection matches the gene we're setting a value for (if not, get another)
                while random_gene[0] != gene:
                    random_gene = random.choice(genes)

                # Set gene value
                random_gene_value = random_gene[1]

                # Mutate the gene???
                # gene_range = self.params[gene]
                # mutated_value = random_gene_value + random.choice(gene_range)

                # Create new solution with random gene value
                new_descendant[gene] = random_gene_value

        return new_descendant

    # Go through generations (combos=initial population)
    def evolve(self):
        logger.info("=== BEGINNING EVOLUTION ===")
        self.evolving = True
        logger.info(f"- Generations: {self.generations}")
        logger.info(f"- Population Size: {self.population_size}")

        # For holding previous solutions
        all_previous_solutions = []

        # Spawn first generation
        current_generation = self.initialize_population(self.population_size)

        population_queue, fitness_results = multiprocessing.Queue(maxsize=100), multiprocessing.Queue()

        workers = [multiprocessing.Process(target=self.get_fitness_of_queue, args=(population_queue, fitness_results))
                   for core in range(0, n_cores - 2)]

        queue_filler = multiprocessing.Process(target=self.fill_population_queue,
                                               args=(population_queue, current_generation))
        queue_filler.start()

        for worker in workers:
            worker.start()

        top_10 = []

        for gen in range(self.generations):
            logger.info(f"=== Generation {gen}  ===")

            # Add current generation to list of tried solutions (to not have descendants that are repeats of an ancestor)
            all_previous_solutions.extend(current_generation)

            # Use workers to calculate fitness values of each member of the initial population
            ranked_solutions = self.calculate_population_fitness(fitness_results=fitness_results)

            # Save generation to test file
            results_to_save = [solution[2] for solution in ranked_solutions]
            data_handler.save_results(results_to_save)

            # Sort all solutions by fitness value
            ranked_solutions.sort(key=lambda x: x[0])
            ranked_solutions.reverse()

            # Get top N% of the generation to use for gene extraction
            num_ancestors = round(self.population_size * self.gen_ancestor_percentage)
            next_gen_ancestors = ranked_solutions[:num_ancestors]

            # Add current top ten to overall top ten, then find new top 10
            for top in next_gen_ancestors[:10]:
                if top not in top_10:
                    top_10.append(top)
            top_10.sort(key=lambda x: x[0])
            top_10.reverse()
            top_10 = top_10[:10]

            # See if we need to toss in some random solutions
            if self.shakeup:
                solutions_from_genes = round(self.population_size * (1 - self.shakeup_percentage))
            else:
                solutions_from_genes = self.population_size

            # Create new generation from the good genes
            logger.info("Spawning next generation...")
            next_generation = []
            for _ in range(solutions_from_genes):
                # Add new solution to the next generation
                next_generation.append(self.spawn_original_descendant(next_gen_ancestors, all_previous_solutions))

            logger.info("Spawned.")

            # Throw some random individuals into the next generation to freshen things
            if self.shakeup:
                shakeup_count = round(self.population_size * self.shakeup_percentage)
                next_generation.extend(
                    random.sample(self.all_combos, shakeup_count))  # Add random sample to next generation
                logger.info(f"Tossed in {shakeup_count} random solutions for next generation.")

            # Start filling with new population
            queue_filler = multiprocessing.Process(target=self.fill_population_queue,
                                                   args=(population_queue, next_generation))
            queue_filler.start()

            logger.info(f"Current Top 10: {top_10}")

        # Evolution finished, signal threads to end
        self.evolving = False
        population_queue.put(None)


class Tester:

    def __init__(self, strategy, logging=True):
        self.symbol = symbol
        self.timeframe = timeframe
        self.start = start
        self.end = end
        self.logging = logging

        self.data = data_handler.data
        self.strategy = strategy
        self.cerebro = None

    def opt_test_mp(self, strategy, params_to_test, randomize=False):
        # Set path for tests
        params, values = zip(*params_to_test.items())
        path = data_handler.test_dir() + f"/{params}.csv"
        data_handler.set_test_path(path)

        param_combos = get_param_combos(params_to_test=params_to_test, test_path=path)

        # Test parameter combinations
        logger.info(f"{len(param_combos)} combinations to test")

        # If Monte Carlo, shuffle
        if randomize:
            random.shuffle(param_combos)

        # Prepare for multiprocessing
        # (set in_q maximum size to prevent bloating while filling queue and waiting for workers)
        # (queue.put() will pause until space becomes available, i.e. queue.get() called by worker)
        in_q, out_q = multiprocessing.Queue(maxsize=100), multiprocessing.Queue()

        # Create workers
        # (create number of workers == to number of logical cores minus two (one for main thread, one for filling queue))
        workers = [multiprocessing.Process(target=worker_test_combos, args=(in_q, out_q, self.data, strategy))
                   for core in range(0, n_cores - 2)]

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
                ten_percent = round(combo_count * 0.1) if combo_count >= 1000 else combo_count
                if tests_done % ten_percent == 0 and tests_done != 0:
                    logger.info(f"\nSaving previous {ten_percent} tests...")

                    data_handler.save_results(results)
            except:
                pass

        logger.info('Finished.')

        # Save last results
        logger.info("Saving leftovers...")
        data_handler.save_results(results)
        logger.info("Leftovers Saved.")

        # Sort by sqn
        data_handler.final_sort_tests("sqn")

    def test(self, params=None, logging=False):
        # Build cerebro
        self.cerebro = build_cerebro(optimizing=False, data=self.data)

        # Add metric analyzer
        self.cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')

        # Add strategy to Cerebro
        if params is None:
            params = {}
        params.update({'logging': logging})
        self.cerebro.addstrategy(self.strategy, **params)

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

        logger.info(result_dict)

        # Make test directory
        params.pop('logging')
        param_names, values = zip(*params.items())
        path = data_handler.test_dir() + f"/{param_names}.csv"
        curr_df = pd.DataFrame(data=result_dict, index=[0])
        # Save to CSV
        if exists(path):
            existing_csv = pd.read_csv(path, index_col=0)
            merged = existing_csv.append(curr_df, ignore_index=True)
            merged.to_csv(path)
        else:
            curr_df.to_csv(path)

        strat_name = ""

        filename = strat_name + f'_{self.symbol}_{self.timeframe}_{self.start} to {self.end}_' + '.html'
        quantstats.reports.html(returns, output=filename, title=strat_name)

        # Rename report file to something fitting
        if exists(filename):
            os.remove(filename)
        os.rename('quantstats-tearsheet.html', filename)

        return filename


if __name__ == "__main__":
    # TODO: RESEARCH MORE INDICATORS (entry, confirmation, volume, trend, volatility)
    # TODO: Look into Aroon (for direction of trend), DPO (for entries?)

    logger = get_logger()
    # strategy = Trend
    symbol, timeframe, start, end = "AUD_NZD", "M30", "2020-01-01", "2022-11-04"
    data_handler = Data()
    logger.info((symbol, timeframe, start, end))

    # Params to optimize
    # param_opt_ranges = {
    #     'atr': numpy.arange(4, 32, step=2),
    #     'atr_sl': numpy.arange(1, 4.25, step=0.25),
    #     'atr_tp': numpy.arange(1, 4.25, step=0.25),
    #     'rsi': numpy.arange(4, 30, step=2),
    #     'adx': numpy.arange(4, 30, step=2),
    #     'adx_cutoff': numpy.arange(25, 40, step=5),
    #     'sma': numpy.arange(20, 150, step=5)
    # }

    param_opt_ranges = {
        'atr': [10],
        'atr_sl': [2.25],
        'atr_tp': [1.0],
        'rsi': [12],
        'adx': [14],
        'adx_cutoff': [30],
        'sma': numpy.arange(20, 102, step=2)
    }

    test_details = {
        'strategy': Trend,
        'symbol': 'AUD_NZD',
        'timeframe': 'M30',
        'start': '2020-01-01',
        'end': '2022-11-04'
    }

    tester = Tester(Trend)
    tester.opt_test_mp(strategy=Trend, params_to_test=param_opt_ranges, randomize=True)
    # test_one = {'atr': 10, 'atr_sl': 2.25, 'atr_tp': 1.0, 'rsi': 12, 'adx': 14, 'adx_cutoff': 30}
    # tester.test(Trend, test_one, logging=False)
    # optimize(details=test_details, param_settings=param_opt_ranges)
    #

    # Genetic Algorithm
    evolution_parameters = {
        "population_size": 1250,
        "generations": 10,
        "gen_ancestor_percentage": 0.1,  # Use top n percent of generation for gene extraction
        "fitness_target": 7,
        "shakeup": False,  # Do or don't throw in some random entities each generation
        "shakeup_percentage": 0.05
    }
    # darwin = GeneticOptimizer(strategy=Trend, params=param_opt_ranges, evolution_rules=evolution_parameters)
    # darwin.evolve()

    # TODO: Look into more MAs for Baseline (NNFX MA video), save each generation to corresponding .csv file

    playsound('C:\\Users\\Nick\\Documents\\GitHub\\AlgoTrader\\guh_huh.mp3')

    sys.exit()

    # result_file = tester.test(Trend, logging=False)
    # # # Open report in chrome
    # chrome = webbrowser.get("C:/Program Files/Google/Chrome/Application/chrome.exe %s")
    # chrome.open('file://' + os.path.realpath(result_file))
