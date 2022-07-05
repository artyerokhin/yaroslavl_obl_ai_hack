"""
useful utils.
"""
import re
from typing import Dict

import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.metrics import recall_score
from sklearn.preprocessing import FunctionTransformer

from constants import CNT_EDGES, DROP_COLS, DROP_COLS_TEST, MAPPING, TIME_COLS


def fixed_recall(y_true: np.array, y_pred: np.array) -> float:
    """Compute competition metric - mean of recall for positive and negative labels.

    Args:
        y_true (np.array) - true labels
        y_pred (np.array) - predicted labels

    Returns:
        score (float) - mean of recalls for positive and negative labels
    """
    return np.mean([recall_score(y_true, y_pred, pos_label=1, zero_division=0),
                    recall_score(y_true, y_pred, pos_label=0, zero_division=1)])


def cramers_v(x, y):
    """Compute Cramer's V."""

    confusion_matrix = pd.crosstab(x, y)

    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()

    phi2 = chi2 / n

    r, k = confusion_matrix.shape

    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))

    rcorr = r - ((r - 1)**2) / (n - 1)
    kcorr = k - ((k - 1)**2) / (n - 1)

    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def map_cols(df: pd.DataFrame, mapping: dict, fill_val: int) -> pd.DataFrame:
    """Map values from all dicts mapping for columns and fillna with fill_valю

    Args:
        df (pd.DataFrame) - initial dataframe
        mapping (dict) - dictionary of mapped values
        fill_val (int) - integer value to fill nan's

    Returns:
        df (pd.DataFrame) - copy of initial dataframe with mapped and filled values
    """
    df = df.copy()

    for key in mapping.keys():
        df[key] = df[key].map(mapping[key]).fillna(0)

    return df


def sin_transformer(period: int) -> FunctionTransformer:
    """Transform value into sinusoidal form."""
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period: int) -> FunctionTransformer:
    """Transform value into cosinusoidal form."""
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def diff_col(df: pd.DataFrame, start_col: str, end_col: str,
             shift: int = 0) -> pd.DataFrame:
    """Get difference between start and end time columns.
    Columns are equal formatted (hh:mm:ss, 24 hours day), so we need to shift our difference (ex. 6AM - 10PM = 8, but != -16)

    Args:
        df (pd.DataFrame) - initial dataframe
        start_col (str) - column to diff
        end_col (str) - column to diff from
        shift (int) - shift difference parameter

    Returns:
        df (pd.DataFrame) - copy of initial dataframe with new diff feature
    """
    df = df.copy()

    df[f'{start_col}_{end_col}_diff'] = np.abs(
        df[end_col] - df[start_col] + shift)
    df[f'{start_col}_{end_col}_diff'] = np.where(df[f'{start_col}_{end_col}_diff'] < 24,
                                                 df[f'{start_col}_{end_col}_diff'],
                                                 df[f'{start_col}_{end_col}_diff'] + shift)

    return df


def trigonometric_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Compute trigonometric features for hourly data."""
    df = df.copy()

    df[f'{col}_sin'] = sin_transformer(24).fit_transform(df[col])
    df[f'{col}_cos'] = cos_transformer(24).fit_transform(df[col])

    return df


def process_time_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Process time features."""
    df = df.copy()

    for col in cols:
        # get datetime and hour (sleep and wake up time)
        df[col] = pd.to_datetime(df[col])
        df[f'{col}_hour'] = df[col].dt.hour

    # difference between going to sleep and waking up
    df = diff_col(df, f'{cols[0]}_hour', f'{cols[1]}_hour', -24)

    return df


def filter_by_col_cnts(df: pd.DataFrame, col_cnt_dict: dict = None) -> dict:
    """Filter columns, that >= of constant rareness value.

    Args:
        df (pd.DataFrame) - initial dataframe
        col_cnt_dict (dict) - dictionary with edge values for columns

    Returns:
        passed_cnt_dict (dict) - dict of passed column items
    """
    df = df.copy()

    if col_cnt_dict is None:
        raise ValueError('col_cnt_dict mustn\'t be empty')

    passed_cnt_dict = {}

    for col, cnt in col_cnt_dict.items():
        val_cnt = df[col].value_counts()
        passed_cnt_dict[col] = val_cnt[val_cnt >= cnt].index.tolist()

    return passed_cnt_dict


def replace_nan_col_values(
        df: pd.DataFrame, col_dict: Dict[str, int] = None) -> pd.DataFrame:
    """Replace values out of col_dict with nan's.

    Args:
        df (pd.DataFrame) - initial dataframe
        col_dict (dict) - dict of column list
    Returns:
        df (pd.DataFrame) - dataframe with replaced with nan values
    """
    df = df.copy()

    if col_dict is None:
        raise ValueError('col_dict mustn\'t be empty')

    for col, cnt in col_dict.items():
        df.loc[~df[col].isin(col_dict[col]), col] = np.nan

    return df


def process_df(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    """Process dataframe for training or for making predictions."""
    df = df.copy()

    df = process_time_columns(df, TIME_COLS)
    if is_test:
        df = df.drop(columns=DROP_COLS_TEST).copy()
    else:
        df = df.drop(columns=DROP_COLS).copy()
    df = map_cols(df, MAPPING, 0)

    passed_dict = filter_by_col_cnts(df, CNT_EDGES)
    df = replace_nan_col_values(df, passed_dict)
    # https://stackoverflow.com/questions/60582050/lightgbmerror-do-not-support-special-json-characters-in-feature-name-the-same
    df = df.rename(columns=lambda x: re.sub('[^А-Яа-яA-Za-z0-9_]+', ' ', x))

    return df
