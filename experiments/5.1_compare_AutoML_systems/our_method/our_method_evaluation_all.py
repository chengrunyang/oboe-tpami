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
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

import glob
from datetime import datetime
import mkl
mkl.set_num_threads(1)

import warnings
warnings.filterwarnings("ignore")

num_outer_folds = 3
random_state_kfold = 0
random_state_autolearner = 1

automl_path = '../../../package/automl'
dataset_path = '../4_gather_datasets/gathered_datasets'
result_path = 'our_method_results'

sys.path.append(automl_path)
from auto_learner import AutoLearner
import util

with open(os.path.join(automl_path, 'defaults', 'training_index.pkl'), 'rb') as handle:
    dataset_indices = pickle.load(handle)

# get available preprocessed OpenML datasets

def get_dataset_ids_from_folder(raw_data_path):
    return [int(re.findall(r'\d+', os.path.basename(f))[0]) for f in os.listdir(raw_data_path) if f.endswith('_features.csv')]

available_OpenML_dataset_indices = get_dataset_ids_from_folder(dataset_path)


def evaluate_our_method(total_runtime, train_features, train_labels):
    clf = AutoLearner(p_type='classification', runtime_limit=total_runtime, n_cores=1, load_imputed=True, verbose=False, random_state=random_state_autolearner)        
    r = clf.fit(train_features, train_labels)    
    return clf, r

def kfolderror(x, y, runtime, num_outer_folds=num_outer_folds):
    training_error = []
    test_error = []
    time_elapsed = []
    kf = StratifiedKFold(num_outer_folds, shuffle=True, random_state=random_state_kfold)
    r_all = []
    for train_idx, test_idx in kf.split(x, y):
        x_tr = x[train_idx, :]
        y_tr = y[train_idx]
        x_te = x[test_idx, :]
        y_te = y[test_idx]
        
        start = time.time()
        clf, r = evaluate_our_method(runtime, x_tr, y_tr)
        time_elapsed.append(time.time() - start)
        y_tr_pred = clf.predict(x_tr)
        y_pred = clf.predict(x_te)
        training_error.append(util.error(y_tr, y_tr_pred, 'classification', 'BER'))
        test_error.append(util.error(y_te, y_pred, 'classification', 'BER'))
        r_all.append(r)
    
    time_elapsed = np.array(time_elapsed).mean()
    training_error = np.array(training_error)
    test_error = np.array(test_error)
    return training_error, test_error, time_elapsed, r_all

def run_experiment_on_single_combination(comb):
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
    
    print("Running our method on dataset {} with preset runtime limit {}".format(dataset_id, runtime_limit))
    try:
        training_error, test_error, time_elapsed, r_all = kfolderror(X, y, int(runtime_limit))
        average_training_error = training_error.mean()
        average_test_error = test_error.mean()
    except:
        training_error = np.full(num_outer_folds, np.nan)
        test_error = np.full(num_outer_folds, np.nan)
        time_elapsed = np.nan
        average_training_error = np.nan
        average_test_error = np.nan
        r_all = []
        pass

    
    result_path_single_dataset = os.path.join(result_path, str(dataset_id))

    if not os.path.exists(result_path_single_dataset):
        print("creating folder {}".format(result_path_single_dataset))
        os.makedirs(result_path_single_dataset)

    results = np.concatenate((np.array([runtime_limit, average_training_error, average_test_error, time_elapsed]), training_error, test_error)).reshape(1, -1)
    columns = ['set_runtime_limit_per_fold', 'average_training_error', 'average_test_error', 'actual_runtime_per_fold'] + ['training_error_fold_{}'.format(i+1) for i in range(num_outer_folds)] + ['test_error_fold_{}'.format(i+1) for i in range(num_outer_folds)]
    
    pd.DataFrame(results, index=[dataset_id], columns=columns).to_csv(
        os.path.join(result_path_single_dataset, 'our_method_dataset_{}_time_{}.csv'.format(dataset_id, runtime_limit)), index=True, header=True)
    
    results_fitting = {}
    results_fitting['fitting_details'] = r_all
    
    with open(os.path.join(result_path_single_dataset, 'our_method_fitting_results_dataset_{}_time_{}.pkl'.format(dataset_id, runtime_limit)), 'wb') as f:
        pickle.dump(results_fitting, f)

dataset_id_all = list(set(dataset_indices).intersection(set(available_OpenML_dataset_indices)))
runtime_limit_all = np.arange(15, 150, 15)

combinations = list(itertools.product(dataset_id_all, runtime_limit_all))


def main(i):
    run_experiment_on_single_combination(comb=combinations[i])


if __name__ == '__main__':
    i = int(sys.argv[1])
    main(i)
