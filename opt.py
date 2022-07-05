"""
optimization tools.
"""
import pickle
from typing import Callable, Dict

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from constants import (N_REPEATS, N_SPLITS, OUTPUT_PATH, PRED_CLASSES,
                       RANDOM_STATE)
from train import train_algo, train_multi_algo
from utils import fixed_recall

skf = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE)


def lgb_opt(trial, train_params: dict) -> float:
    """Optimize lightGBM parameters.

    Args:
        trial - optuna's trial object
        train_params (dict) - initial information for train_algo function

    Returns:
        mean_stats (float) - mean by Folds target metric
    """
    param_grid = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 4, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "max_bin": trial.suggest_int("max_bin", 4, 256),
        "reg_alpha": trial.suggest_int("reg_alpha", 0, 100),
        "reg_lambda": trial.suggest_int("reg_lambda", 0, 100),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.2, 0.95, step=0.05
        )
    }

    train_params['lgb_params'] = param_grid

    _, stats = train_algo(**train_params)

    return -np.mean(stats)


def objective(trial, df: pd.DataFrame, splitter: Callable,
              scorer: Callable, opt_params: dict) -> float:
    """Optimize probability cut-off for all 5 classes.

    Args:
        trial - optuna trial object
        opt_params (dict) - dictionary with optimal algorithm parameters
    Returns:
        mean_stats (float) - mean by Folds target metric
    """
    prob1 = trial.suggest_float('prob1', 0.2, 0.6, log=True)
    prob2 = trial.suggest_float('prob2', 0.01, 0.3, log=True)
    prob3 = trial.suggest_float('prob3', 0.01, 0.3, log=True)
    prob4 = trial.suggest_float('prob4', 0.01, 0.3, log=True)
    prob5 = trial.suggest_float('prob5', 0.01, 0.3, log=True)

    probs = [prob1, prob2, prob3, prob4, prob5]

    # fixed_recall as target metric
    res = train_multi_algo(df, PRED_CLASSES, splitter, scorer,
                           border_probs=probs, lgb_params=[param for _, param in opt_params.items()])

    scores = []
    for key, results in res.items():
        scores.append(np.mean(results[1]))

    return -np.mean(scores)


def find_best_params(df: pd.DataFrame, splitter: Callable, scorer: Callable,
                     silent: bool = False, load_pickle: bool = False) -> Dict[str, dict]:
    """Find best lgb params for all predicted classes.

    Args:
        df (pd.DataFrame) - initial dataframe
        splitter (callable) - function to split dataset into folds
        scorer (callable) - target metric function
        silent (bool) - print additional info flag
        load_picle (bool) - load params from file flag

    Returns:
        opt_params (dict) - optimal parameters for all predicted classes
    """
    if not load_pickle:
        opt_params = {}

    for pred_class in PRED_CLASSES:
        print(pred_class)
        # Roc-AUC used as target metric for lgb optimization
        params = {'df': df.drop(PRED_CLASSES, axis=1), 'target': df[pred_class], 'splitter': splitter,
                  'scorer': scorer, 'border_prob': None, 'add_split': None, 'silent': silent}

        study = optuna.create_study()
        study.optimize(lambda trial: lgb_opt(trial, params), n_trials=100)

        params['lgb_params'] = study.best_params
        params['lgb_params']['random_state'] = RANDOM_STATE

        _, stats = train_algo(**params)

        opt_params[pred_class] = params['lgb_params']

        print(np.mean(stats))

    return opt_params


def get_lgb_params(df: pd.DataFrame, splitter: Callable = skf, scorer: Callable = roc_auc_score,
                   silent: bool = False, load_pickle: bool = False) -> Dict[str, dict]:
    """Get lightgbm params - from file or with optuna search."""

    if not load_pickle:
        opt_params = find_best_params(
            df, splitter, scorer, silent, load_pickle)

        # save fined parameters
        with open(f'{OUTPUT_PATH}/lgb_params.pickle', 'wb') as file:
            pickle.dump(opt_params, file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(f'{OUTPUT_PATH}/lgb_params.pickle', 'rb') as file:
            opt_params = pickle.load(file)

    return opt_params


def get_cutoff_params(df: pd.DataFrame, params: dict, splitter: Callable = skf, scorer: Callable = fixed_recall,
                      load_pickle: bool = False) -> list:
    """Get cutoff params - from file or with optuna search."""

    if not load_pickle:
        study = optuna.create_study()
        study.optimize(
            lambda trial: objective(
                trial,
                df,
                splitter,
                scorer,
                params),
            n_trials=200)

        probs = study.best_params
        print(probs)

        probs = [prob for _, prob in probs.items()]

        with open(f'{OUTPUT_PATH}/probs.pickle', 'wb') as file:
            pickle.dump(probs, file, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(f'{OUTPUT_PATH}/probs.pickle', 'rb') as file:
            probs = pickle.load(file)

    return probs
