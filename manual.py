import pandas as pd

timeframes = ["M5", "M15", "M30"]
paths = []
for t in timeframes:
    test_path = f"C:\\Users\\sm598\\OneDrive\\Documents\\GitHub\\AlgoTrader\\backtesting\\tests\\{t}\\2020-01-01 to " \
                "2022-11-04\\"

    results = pd.read_csv(test_path + "('atr', 'atr_sl', 'atr_tp', 'rsi', 'adx', 'adx_cutoff').csv")

    best_df = pd.DataFrame()

    best = {}
    for index, row in results.iterrows():
        symbol = row['symbol']
        sqn = row['sqn']
        if sqn < 2:
            continue
        if symbol not in best.keys():
            best[symbol] = row
        else:
            if best[symbol]['sqn'] < sqn:
                best[symbol] = row

    for k, v in best.items():
        best_df = best_df.append(v, ignore_index=True)

    best_df = best_df.drop('Unnamed: 0', axis=1)

    path = f"('atr', 'atr_sl', 'atr_tp', 'rsi', 'adx', 'adx_cutoff')_BEST_{t}.csv"
    best_df.to_csv(path)
    paths.append(path)

final_results_df = pd.DataFrame()
best = dict()
for t in timeframes:
    path = f"('atr', 'atr_sl', 'atr_tp', 'rsi', 'adx', 'adx_cutoff')_BEST_{t}.csv"
    results = pd.read_csv(path)

    for index, row in results.iterrows():
        symbol = row['symbol']
        sqn = row['sqn']
        row['timeframe'] = t

        if sqn < 2:
            continue

        if symbol not in best.keys():
            best[symbol] = row
        else:
            if best[symbol]['sqn'] < sqn:
                best.pop(symbol)
                best[symbol] = row

    for k, v in best.items():
        final_results_df = final_results_df.append(v, ignore_index=True)

    # final_results_df.columns = ['s', 'sqn', 't']
    final_results_df = final_results_df.drop_duplicates()
    final_results_df = final_results_df.sort_values(by=['sqn'], ascending=False)
    final_results_df.to_csv("BEST_PAIR_SETTINGS.csv")