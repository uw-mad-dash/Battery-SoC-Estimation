import pandas as pd

from hampel import hampel
from scipy.signal import butter, lfilter


def get_hampel(col_values: pd.Series, window_size: int, n: int):

    return hampel(col_values, window_size, n, imputation=True)


def get_ma(col_values: pd.Series, window_size: int):
    res = col_values.shift(1).rolling(window_size, min_periods=1).mean()
    res.fillna(col_values, inplace=True)

    return res


def get_bwf(col_values: pd.Series, cutoff_fs: float, order: int):
    nyq = 0.5
    normal_cutoff = cutoff_fs / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    return lfilter(b, a, col_values)
