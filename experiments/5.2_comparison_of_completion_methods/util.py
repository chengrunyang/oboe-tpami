import numpy as np
from scipy.linalg import qr

def missing_ratio(tensor):
    return np.sum(np.isnan(tensor)) / np.prod(tensor.shape)
    
def relative_error(array_true, array_pred):
    return np.linalg.norm(array_pred - array_true) / np.linalg.norm(array_true)

def pivot_columns(a, rank=None, columns_to_avoid=None, threshold=None):
    """Computes the QR decomposition of a matrix with column pivoting, i.e. solves the equation AP=QR such that Q is
    orthogonal, R is upper triangular, and P is a permutation matrix.
    Args:
        a (np.ndarray):    Matrix for which to compute QR decomposition.
        threshold (float): Threshold specifying approximate rank of a. All singular values less than threshold * (largest singular value) will be set to 0
        rank (int):        The approximate rank.
    Returns:
        np.array: The permutation p.
    """
    if columns_to_avoid:
        set_of_columns_to_avoid = set(columns_to_avoid)
    else:
        set_of_columns_to_avoid = set()
    
    assert (threshold is None) != (rank is None), "Exactly one of threshold and rank should be specified."
    if threshold is not None:
        rank = approx_rank(a, threshold)
    
    qr_columns = qr(a, pivoting=True)[2]
    
    r = []
    i = 0
    while(len(r) < rank):
        if qr_columns[i] not in set_of_columns_to_avoid:
            r.append(qr_columns[i])
        i += 1
    return r