import numpy as np
import pickle
import itertools
import os
from sklearn.model_selection import LeaveOneOut
import tensorly as tl
from scipy.optimize import minimize
import time

import sys
from factorizations import tucker_on_error_tensor, get_omega, matrix_completion_by_EM
from util import missing_ratio, relative_error


def test_tensor_completion(error_tensor, runtime_tensor, ranks_for_imputation, runtime_threshold, verbose=False):
    print("ranks: {}, runtime_threshold: {}".format(ranks_for_imputation, runtime_threshold))
    masking_criteria = "runtime_tensor >= {}".format(runtime_threshold)
    if verbose:
        print("masking entries that satisfy: {}".format(masking_criteria))
        
    masked_indices = np.where(eval(masking_criteria))
    error_tensor_masked = error_tensor.copy()
    error_tensor_masked[eval(masking_criteria)] = np.nan
    
    _, _, error_tensor_pred, errors = tucker_on_error_tensor(error_tensor_masked, ranks_for_imputation, save_results=False, verbose=verbose)
    
    return relative_error(error_tensor[eval(masking_criteria)], error_tensor_pred[eval(masking_criteria)])
    

def test_matrix_completion(error_matrix, runtime_matrix, rank_for_imputation, runtime_threshold, verbose=False):
    print("rank: {}".format(rank_for_imputation))
    masking_criteria = "runtime_matrix >= {}".format(runtime_threshold)
    if verbose:
        print("masking entries that satisfy: {}".format(masking_criteria))
        
    masked_indices = np.where(eval(masking_criteria))
    
    error_matrix_masked = error_matrix.copy()
    error_matrix_masked[eval(masking_criteria)] = np.nan
    
    error_matrix_pred, errors_matrix = matrix_completion_by_EM(error_matrix_masked, rank=rank_for_imputation, verbose=verbose)
    
    return relative_error(error_matrix[eval(masking_criteria)], error_matrix_pred[eval(masking_criteria)])
    
    
