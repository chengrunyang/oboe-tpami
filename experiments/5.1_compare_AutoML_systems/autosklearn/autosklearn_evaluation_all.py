from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import balanced_accuracy
import pandas as pd
import numpy as np
import time
import re
from sklearn.model_selection import KFold, StratifiedKFold
from subprocess import check_output
import sys
import itertools
import os
import pickle
import multiprocessing as mp
from scipy.stats import mode
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

import glob
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
# import mkl
# mkl.set_num_threads(1)


automl_path = '../../../package/automl'
dataset_path = '../4_gather_datasets/gathered_datasets'
result_path = 'autosklearn_results'
num_outer_folds = 3
RANDOM_STATE = 0

sys.path.append(automl_path)
import util

with open(os.path.join(automl_path, 'defaults', 'training_index.pkl'), 'rb') as handle:
    dataset_indices = pickle.load(handle)

# get available OpenML datasets

def get_dataset_ids_from_folder(raw_data_path):
    return [int(re.findall(r'\d+', os.path.basename(f))[0]) for f in os.listdir(raw_data_path) if f.endswith('_features.csv')]

available_OpenML_dataset_indices = get_dataset_ids_from_folder(dataset_path)



def autosklearn_evaluation(total_runtime, train_features, train_labels, seed=1):
    clf = AutoSklearnClassifier(
            time_left_for_this_task=total_runtime,
            include_estimators = ["adaboost","gaussian_nb", "extra_trees", "gradient_boosting", 
                                 "liblinear_svc", "random_forest",
                                 "k_nearest_neighbors","decision_tree"],
            seed=seed, # random seed
    )
        
    clf.fit(train_features, train_labels, metric=balanced_accuracy)
    return clf

def kfolderror(x, y, runtime, num_outer_folds=num_outer_folds, seed=1):
    training_error = []
    test_error = []
    time_elapsed = []
    kf = StratifiedKFold(num_outer_folds, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, test_idx in kf.split(x, y):
        x_tr = x[train_idx, :]
        y_tr = y[train_idx]
        x_te = x[test_idx, :]
        y_te = y[test_idx]
        
        start = time.time()
        try:
            clf = autosklearn_evaluation(runtime, x_tr, y_tr, seed=seed)
            y_tr_pred = clf.predict(x_tr)
            y_pred = clf.predict(x_te)
        except:
            y_tr_mode = mode(y_tr.flatten())[0][0]
            y_tr_pred = np.full(x_tr.shape[0], y_tr_mode)
            y_pred = np.full(x_te.shape[0], y_tr_mode)
#         clf.refit(x_tr, y_tr)
        time_elapsed.append(time.time() - start)

        training_error.append(util.error(y_tr, y_tr_pred, 'classification', 'BER'))
        test_error.append(util.error(y_te, y_pred, 'classification', 'BER'))
    
    time_elapsed = np.array(time_elapsed).mean()
    training_error = np.array(training_error)
    test_error = np.array(test_error)
    return training_error, test_error, time_elapsed


def run_experiment_on_single_combination(comb, seed=1):
    dataset_id = comb[0]
    runtime_limit = comb[1]

    # read dataset
    X = pd.read_csv(os.path.join(dataset_path, "dataset_{}_features.csv".format(dataset_id)), index_col=None, header=None)
    y = pd.read_csv(os.path.join(dataset_path, "dataset_{}_labels.csv".format(dataset_id)), index_col=None, header=None)
    categorical = pd.read_csv(os.path.join(dataset_path, "dataset_"+str(dataset_id)+"_categorical.csv"), index_col=None, header=None).values.T[0]
    X.columns = categorical
    # encode the categorical entries that are not missing
    X = X.apply(lambda series: pd.Series(le.fit_transform(series[series.notnull()]), 
                                         index=series[series.notnull()].index) if series.name else series)
    y = y.apply(lambda x: le.fit_transform(x))
    X = X.values
    y = y.values.T[0]
    
    print("Running auto-sklearn on dataset {} with preset runtime limit {}".format(dataset_id, runtime_limit))
    #         error, time_elapsed = kfolderror(x, y, int(runtime_limit))
    try:
        print("tried runtime limit {}".format(runtime_limit))
        training_error, test_error, time_elapsed = kfolderror(X, y, int(runtime_limit), seed=seed)
        average_training_error = training_error.mean()
        average_test_error = test_error.mean()
    except:
    #         with open(os.path.join(dirname, configs_file), 'a') as log:
    #             log.write(str(e))
        training_error = np.full(num_outer_folds, np.nan)
        test_error = np.full(num_outer_folds, np.nan)
        time_elapsed = np.nan
        average_training_error = np.nan
        average_test_error = np.nan
        pass


    result_path_single_dataset = os.path.join(result_path, str(dataset_id))

    if not os.path.exists(result_path_single_dataset):
        print("creating folder {}".format(result_path_single_dataset))
        os.makedirs(result_path_single_dataset)

    results = np.concatenate((np.array([runtime_limit, average_training_error, average_test_error, time_elapsed]), training_error, test_error)).reshape(1, -1)
    columns = ['set_runtime_limit_per_fold', 'average_training_error', 'average_test_error', 'actual_runtime_per_fold'] + ['training_error_fold_{}'.format(i+1) for i in range(num_outer_folds)] + ['test_error_fold_{}'.format(i+1) for i in range(num_outer_folds)]
    
    pd.DataFrame(results, index=[dataset_id], columns=columns).to_csv(
        os.path.join(result_path_single_dataset, 'autosklearn_dataset_{}_time_{}.csv'.format(dataset_id, runtime_limit)), index=True, header=True)



dataset_id_all = sorted(list(set(dataset_indices).intersection(set(available_OpenML_dataset_indices))))
runtime_limit_all = np.arange(15, 150, 15)

combinations = list(itertools.product(dataset_id_all, runtime_limit_all))


def main(i):
    run_experiment_on_single_combination(comb=combinations[i], seed=i)


if __name__ == '__main__':
    i = int(sys.argv[1])
    main(i)
