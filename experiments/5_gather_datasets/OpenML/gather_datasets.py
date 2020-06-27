import numpy as np
import scipy as sp
import scipy.sparse as sps
import openml
import random
import pandas as pd
import sys
import os

i = sys.argv[1]

# output directory
output_directory = 'gathered_datasets'
target_files_path = os.path.join(os.getcwd(), output_directory)
if not os.path.exists(target_files_path):
    os.makedirs(target_files_path)

selected_datasets = pd.read_csv(os.path.join(os.getcwd(), "selected_OpenML_classification_dataset_indices.csv"), index_col=None, header=None).values.T[0]
dataset_id = int(selected_datasets[int(i)])
print(dataset_id)

try:
    dataset = openml.datasets.get_dataset(dataset_id)
    data_features, data_labels, data_categorical, _ = dataset.get_data(target=dataset.default_target_attribute)
except:
    directory = '{}/datasets_with_error'.format(target_files_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.system('touch '+str(directory)+'/'+str(dataset_id)+'.txt')

if sps.issparse(data_features):
    data_features=data_features.todense()


if type(data_features) is pd.core.frame.DataFrame or pd.core.sparse.frame.SparseDataFrame:
    data_features.to_csv(str(target_files_path)+'/dataset_'+str(dataset_id)+'_features.csv', header=False, index=False)
elif type(data_features) == np.ndarray:
    pd.DataFrame(data_features, index=None, columns=None).to_csv(str(target_files_path)+'/dataset_'+str(dataset_id)+'_features.csv', header=False, index=False)


if type(data_labels) == pd.core.series.Series:
    data_labels.to_csv(str(target_files_path)+'/dataset_'+str(dataset_id)+'_labels.csv', header=False, index=False)
elif type(data_labels) == np.ndarray:
    pd.DataFrame(data_labels.reshape(-1, 1), index=None, columns=None).to_csv(str(target_files_path)+'/dataset_'+str(dataset_id)+'_labels.csv', header=False, index=False)
                                
pd.DataFrame(np.array(data_categorical).reshape(-1, 1), index=None, columns=None).to_csv(str(target_files_path)+'/dataset_'+str(dataset_id)+'_categorical.csv', header=False, index=False)

print("dataset "+str(dataset_id)+" finished")