from tpot import TPOTClassifier
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


automl_path = '../../../package/automl'
dataset_path = '../4_gather_datasets/gathered_datasets'
result_path = 'tpot_results'
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


classifier_config_dict_custom = {
    # Classifiers
    'sklearn.neighbors.KNeighborsClassifier': {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
        'p': [1, 2]
    },

    'sklearn.tree.DecisionTreeClassifier': {
        'min_samples_split': [2,4,8,16,32,64,128,256,512,1024,0.01,0.001,0.0001,1e-05]
    },

    'sklearn.ensemble.RandomForestClassifier': {
        'min_samples_split': [2,4,8,16,32,64,128,256,512,1024,0.01,0.001,0.0001,1e-05],
        'criterion': ["gini", "entropy"],
    },
    
    'sklearn.ensemble.GradientBoostingClassifier': {
        'learning_rate': [0.001,0.01,0.025,0.05,0.1,0.25,0.5],
        'max_depth': [3, 6],
        'max_features': [None, "log2"],
    },
    
    'sklearn.ensemble.AdaBoostClassifier': {
        'n_estimators': [50, 100],
        'learning_rate': [1.0, 1.5, 2.0, 2.5, 3.0],
    },
    
    'sklearn.svm.LinearSVC': {
        'C': [0.125,0.25,0.5,0.75,1,2,4,8,16],
    },

    'sklearn.linear_model.LogisticRegression': {
        'penalty': ["l1", "l2"],
        'C': [0.25,0.5,0.75,1,1.5,2,3,4],
        'solver': ["liblinear", "saga"],
    },
    
    'sklearn.linear_model.Perceptron': {
    },

    'sklearn.naive_bayes.GaussianNB': {
    },
    
    'sklearn.neural_network.MLPClassifier': {
        'learning_rate_init': [0.0001,0.001,0.01],
        'learning_rate': ["adaptive"],
        'solver': ["sgd", "adam"],
        'alpha': [0.0001, 0.01],
    },
    
     'sklearn.ensemble.ExtraTreesClassifier': {
        'min_samples_split': [2,4,8,16,32,64,128,256,512,1024,0.1,0.01,0.001,0.0001,1e-05],
        'criterion': ["gini", "entropy"],
    },

    # Preprocesssors

    'sklearn.decomposition.PCA': {
        'svd_solver': ['randomized'],
        'n_components': [0.2, 0.4, 0.6, 0.8],
        'iterated_power': range(1, 11)
    },

    'sklearn.preprocessing.StandardScaler': {
    },
    
    'sklearn.preprocessing.OneHotEncoder': {
        'handle_unknown': ["ignore"],
        'sparse': [0],
    },
    


    # Selectors

    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },
    
    'sklearn.feature_selection.SelectKBest': {
    },
    
    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(1, 100),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },
}



def tpot_evaluation(total_runtime, train_features, train_labels):
    clf = TPOTClassifier(
            max_time_mins=total_runtime/60,
            scoring='balanced_accuracy',
            config_dict=classifier_config_dict_custom,
    )        
    clf.fit(train_features, train_labels)    
    return clf

def kfolderror(x, y, runtime, num_outer_folds=num_outer_folds):
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
        clf = tpot_evaluation(runtime, x_tr, y_tr)
        time_elapsed.append(time.time() - start)
        y_tr_pred = clf.predict(x_tr)
        y_pred = clf.predict(x_te)
        training_error.append(util.error(y_tr, y_tr_pred, 'classification', 'BER'))
        test_error.append(util.error(y_te, y_pred, 'classification', 'BER'))
    
    time_elapsed = np.array(time_elapsed).mean()
    training_error = np.array(training_error)
    test_error = np.array(test_error)
    return training_error, test_error, time_elapsed



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
    
    print("Running TPOT on dataset {} with preset runtime limit {}".format(dataset_id, runtime_limit))
    #         error, time_elapsed = kfolderror(x, y, int(runtime_limit))
    try:
        training_error, test_error, time_elapsed = kfolderror(X, y, int(runtime_limit))
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
        os.path.join(result_path_single_dataset, 'tpot_dataset_{}_time_{}.csv'.format(dataset_id, runtime_limit)), index=True, header=True)




dataset_id_all = list(set(dataset_indices).intersection(set(available_OpenML_dataset_indices)))
runtime_limit_all = np.arange(15, 150, 15)

combinations = list(itertools.product(dataset_id_all, runtime_limit_all))



p1 = mp.Pool(90)
result = [p1.apply_async(run_experiment_on_single_combination, args=[comb]) 
          for comb in combinations]
p1.close()
p1.join()

