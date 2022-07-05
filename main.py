"""
main script.
"""
import argparse
import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from constants import (N_REPEATS, N_SPLITS, OUTPUT_FILENAME, OUTPUT_PATH,
                       PATHS, PRED_CLASSES, PRED_CLASSES_MAPPING, RANDOM_STATE)
from opt import get_cutoff_params, get_lgb_params
from train import train_multi_algo
from utils import fixed_recall, process_df

parser = argparse.ArgumentParser()
parser.add_argument(
    "--load_pickle",
    help="Find optimal parameters or load them from pickle file",
    type=bool,
    default=True,
)
args = parser.parse_args()


if __name__ == "__main__":

    skf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE)

    train = pd.read_csv(PATHS['train'])
    test = pd.read_csv(PATHS['test'])
    solution = pd.read_csv(PATHS['sample'])

    train_df = process_df(train)
    test_df = process_df(test, is_test=True)
    
    opt_params = get_lgb_params(train_df, load_pickle=args.load_pickle)
    probs = get_cutoff_params(train_df, opt_params, load_pickle=args.load_pickle)

    res = train_multi_algo(train_df, PRED_CLASSES, skf, fixed_recall,
                           border_probs=probs, lgb_params=[param for _, param in opt_params.items()], silent=True)

    scores = []
    for key, results in res.items():
        scores.append(np.mean(results[1]))

    print(np.mean(scores), np.std(scores))

    step_cnt = 0

    for key, results in res.items():
        solution[PRED_CLASSES_MAPPING[key]] = (
            np.mean([alg.predict_proba(test_df)[:, 1] for alg in results[0]], axis=0) > probs[step_cnt]).astype(int)
        step_cnt += 1

    now_dt = str(datetime.datetime.now()).replace(' ', '-')
    output_filename = f'{OUTPUT_PATH}/{now_dt}_{OUTPUT_FILENAME}.csv'

    solution.to_csv(output_filename, index=False)
