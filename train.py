"""
training functions.
"""
from typing import Callable, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd


def train_algo(df: pd.DataFrame, target: pd.Series, splitter: Callable, scorer: Callable, border_prob: Optional[list] = None,
               add_split: Optional[pd.Series] = None, lgb_params: Optional[dict] = None, silent: bool = True):
    """Train only one algorithm.

    Args:
        df (pd.DataFrame) - initial dataframe
        target (pd.Series) - target data series
        splitter (Callable) - Fold split class
        scorer (Callable) - score function
        border_prob (list or None) - list of probs to binarize predictions
        add_split (pd.Series or None) - another target entity to split by
        lgb_params (dict or None) - lightgbm parameters
        silent (bool) - silence flag

    Returns:
        (algos, stats) - tuple of (list of trained models, list of scores)
    """
    algos = []
    stats = []

    if add_split is None:
        add_split = target

    for train_ind, test_ind in splitter.split(df, add_split):
        X_train, y_train = df.iloc[train_ind], target[train_ind]
        X_test, y_test = df.iloc[test_ind], target[test_ind]

        alg = lgb.LGBMClassifier(**lgb_params)

        alg.fit(X_train, y_train)
        algos.append(alg)

        if border_prob is not None:
            stats.append(
                scorer(
                    y_test,
                    (alg.predict_proba(X_test)[
                        :,
                        1] > border_prob).astype(int)))
        else:
            stats.append(scorer(y_test, alg.predict_proba(X_test)[:, 1]))
    if not silent:
        print(stats)
        print(np.mean(stats), np.std(stats))

    return (algos, stats)


def train_multi_algo(df: pd.DataFrame, target_cols: list, splitter: Callable, scorer: Callable, border_probs: Optional[list] = None,
                     add_split: Optional[pd.Series] = None, lgb_params: Optional[dict] = None, silent: bool = True):
    """Train n algorithms. One for any target from target_cols

    Args:
        df (pd.DataFrame) - initial dataframe
        target_cols (list) - list of target column names
        splitter (Callable) - Fold split class
        scorer (Callable) - score function
        border_prob (list or None) - list of probs to binarize predictions
        add_split (pd.Series or None) - another target entity to split by
        lgb_params (dict or None) - lightgbm parameters
        silent (bool) - silence flag

    Returns:
        (algos, stats) - tuple of (list of trained models, list of scores)
    """
    results = {}
    if border_probs is None:
        border_probs = [None] * 5

    if lgb_params is None:
        lgb_params = {'random_state': 2022}

    for col_num, col in enumerate(target_cols):
        if not silent:
            print(col)

        if isinstance(lgb_params, list):
            params = lgb_params[col_num]
        else:
            params = lgb_params

        if add_split is None:
            results[f'{col}'] = train_algo(df.drop(target_cols, axis=1), df[col], splitter, scorer,
                                           border_probs[col_num], lgb_params=params, silent=silent)
        else:
            results[f'{col}'] = train_algo(df.drop(target_cols + [add_split], axis=1), df[col], splitter, scorer,
                                           border_probs[col_num], df[add_split], lgb_params=params, silent=silent)

    return results
