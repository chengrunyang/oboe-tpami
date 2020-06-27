import numpy as np
from tensorly.decomposition import tucker, parafac, non_negative_tucker
from tensorly import tucker_to_tensor

def get_omega(tensor):
    Ω = np.ones(tensor.shape)
    missing_index = np.where(np.isnan(tensor))
    missing_index = list(zip(*missing_index))
    for index in missing_index:
        Ω[index] = 0
    return Ω

# Tucker decomposition

def tucker_on_error_tensor(error_tensor, ranks=[15, 4, 2, 2, 8, 15], save_results=False, verbose=False):
    
    tensor_pred = np.nan_to_num(error_tensor)
    tensor_from_fac = np.zeros(error_tensor.shape)
    errors = []
    num_iterations = 0
    Ω = get_omega(error_tensor)

    # while(not stopping_condition(tensor, tensor_from_fac, threshold)):
    while(len(errors) <= 2 or errors[-1] < errors[-2] - 0.01):
        
        num_iterations += 1
        core, factors = tucker(tensor_pred, ranks=ranks)
        tensor_from_fac = tucker_to_tensor((core, factors))
        error = np.linalg.norm(np.multiply(Ω, np.nan_to_num(error_tensor - tensor_from_fac)))
        
        if verbose:
            if not num_iterations % 5:
                print("ranks: {}, iteration {}, error: {}".format(ranks, num_iterations, error))

        errors.append(error)
        tensor_pred = np.nan_to_num(error_tensor) + np.multiply(1-Ω, tensor_from_fac)
    
    core, factors = tucker(tensor_pred, ranks=ranks)
    
    if save_results:
        save_dir = os.path.join('results_tucker', "dataset_rank_{}".format(ranks[0]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        result = {}
        result["ranks"] = ranks
        result["tensor_pred"] = tensor_pred
        result["errors"] = errors

        with open(os.path.join(save_dir, 'tucker_{}.pkl'.format(ranks)), 'wb') as f:
            pickle.dump(result, f)
    else:
        return core, factors, tensor_pred, errors
    
# def imputation_by_tucker(error_tensor, ranks=[15, 4, 2, 2, 8, 15], save_results=False, verbose=False):
#     Ω = get_omega(error_tensor)
#     core, factors, errors = tucker_on_error_tensor(error_tensor=error_tensor, ranks=ranks, save_results=save_results, verbose=verbose)
#     tensor_from_fac = tucker_to_tensor((core, factors))
#     tensor_pred = np.nan_to_num(error_tensor) + np.multiply(1-Ω, tensor_from_fac)
#     return tensor_pred
    
    
def matrix_completion_by_EM(error_matrix, rank=50, verbose=False):
    
    error_matrix_pred = np.nan_to_num(error_matrix)
    matrix_from_fac = np.zeros(error_matrix.shape)
    errors = []
    num_iterations = 0
    Ω = get_omega(error_matrix)

    while(len(errors) <= 2 or errors[-1] < errors[-2] - 0.01):
        
        num_iterations += 1
        U, Σ, Vt = np.linalg.svd(error_matrix_pred, full_matrices=False)
        Uk = U[:, :rank]
        Σk = Σ[:rank]
        Vtk = Vt[:rank, :]
        
        error_matrix_from_fac = Uk.dot(np.diag(Σk)).dot(Vtk)
    
        error = np.linalg.norm(np.multiply(Ω, np.nan_to_num(error_matrix - error_matrix_from_fac)))
        
        if verbose:
            if not num_iterations % 5:
                print("rank: {}, iteration {}, error: {}".format(rank, num_iterations, error))

        errors.append(error)
        error_matrix_pred = np.nan_to_num(error_matrix) + np.multiply(1-Ω, error_matrix_from_fac)

    else:
        return error_matrix_pred, errors
