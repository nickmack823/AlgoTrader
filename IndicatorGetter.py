import numpy as np

def vidya(close_prices, period=9, hist_period=30):
    """
    Calculates the VIDYA indicator for forex prices.

    Parameters:
        close_prices (list): A list of closing prices of the forex pair you want to calculate the indicator for.
        period (int): the number of periods for the standard deviation of the fast moving average (default is 9)
        hist_period (int): the number of periods for the standard deviation of the slow moving average (default is 30)

    Returns:
        vidya (numpy.ndarray): The VIDYA indicator as a list of values.
    """
    vidya = np.zeros(len(close_prices))
    for i in range(len(close_prices)):
        if i < len(close_prices) - hist_period:
            k_coefficient = np.std(close_prices[i - period:i], ddof=0) / np.std(close_prices[i - hist_period:i], ddof=0)
            smooth_coefficient = (2.0 / (period + 1))
            vidya[i] = k_coefficient * smooth_coefficient * close_prices[i] + (1 - k_coefficient * smooth_coefficient) * \
                       vidya[i + 1]
        else:
            vidya[i] = close_prices[i]
    return vidya


def absolute_strength_histogram(prices, length=3, smooth=3, signal=3, mode='EMA'):
    bulls = np.zeros(len(prices))
    bears = np.zeros(len(prices))
    avg_bulls = np.zeros(len(prices))
    avg_bears = np.zeros(len(prices))
    smth_bulls = np.zeros(len(prices))
    smth_bears = np.zeros(len(prices))
    sig_bulls = np.zeros(len(prices))
    sig_bears = np.zeros(len(prices))

    # Calculation loop
    for i in range(len(prices)):
        if i < length + smooth + signal:
            continue
        price1 = prices[i - 1]
        price2 = prices[i - 2]
        hhb = max(price1, price2)
        llb = min(price1, price2)
        bulls[i] = hhb - prices[i]
        bears[i] = prices[i] - llb

        if mode == 'EMA':
            avg_bulls[i] = (bulls[i] + (length - 1) * avg_bulls[i - 1]) / length
            avg_bears[i] = (bears[i] + (length - 1) * avg_bears[i - 1]) / length
        elif mode == 'SMA':
            avg_bulls[i] = np.mean(bulls[i - length + 1:i + 1])
            avg_bears[i] = np.mean(bears[i - length + 1:i + 1])
        else:
            raise ValueError("Invalid mode. Use 'EMA' or 'SMA'")

        smth_bulls[i] = (avg_bulls[i] + (smooth - 1) * smth_bulls[i - 1]) / smooth
        smth_bears[i] = (avg_bears[i] + (smooth - 1) * smth_bears[i - 1]) / smooth
        sig_bulls[i] = (smth_bulls[i] + (signal - 1) * sig_bulls[i - 1]) / signal
        sig_bears[i] = (smth_bears[i] + (signal - 1) * sig_bears[i - 1]) / signal
    return sig_bulls, sig_bears

