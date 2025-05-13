from scipy.sparse import random as sparse_random
import numpy as np


def strategy_dropout_global(mat, p_drop=0.5, random_state=42):
    """
    Dropout strategy for sparse matrices.
    Randomly drops elements from the matrix with a given probability.
    Args:
        - mat (scipy.sparse.csr_matrix): Input sparse matrix.
        - p_drop (float): Probability of dropping an element. Default is 0.5.
        - random_state (int): Random seed for reproducibility. Default is 42.
    Returns:
        - mat2 (scipy.sparse.csr_matrix): Sparse matrix with elements dropped.
    """
    rng = np.random.default_rng(random_state)
    # Create a binary mask: 1=keep, 0=drop
    mask_data = rng.random(mat.data.shape) > p_drop
    mat2 = mat.copy()
    mat2.data *= mask_data
    mat2.eliminate_zeros()
    return mat2.tocsr()

import numpy as np
from scipy.sparse import csr_matrix

def strategy_add_global(mat, p_add=0.5, random_state=42):
    """
    Add-noise strategy for sparse matrices.
    Randomly adds +1 to some of the existing elements in the matrix with probability p_add.
    Args:
        - mat (scipy.sparse.csr_matrix): Input sparse matrix.
        - p_add (float): Probability of adding one comment to an element. Default is 0.5.
        - random_state (int): Random seed for reproducibility. Default is 42.
    Returns:
        - mat2 (scipy.sparse.csr_matrix): Sparse matrix with extra comments added.
    """
    rng = np.random.default_rng(random_state)

    mat2 = mat.copy().tocsr()

    # For each non null element in the sparse matrix, add +1 with probability p_add
    mask = rng.random(mat2.data.shape) < p_add
    mat2.data += mask.astype(int)

    return mat2


def strategy_add_comments_by_group(mat, types_array, p_add_minority=0.05, p_add_nimby=0.05, p_add_other=0.05, ran_st_nim=1, ran_st_min=11, ran_st_oth=111):
    """
    For each user u and each proposal j of type <group>,
    with probability p_add_{group}, increment the number of comments by 1.
    Args:
        - mat (scipy.sparse.csr_matrix): Input sparse matrix.
        - types_array (np.ndarray): Array of proposal types (e.g., 'minority', 'nimby').
        - p_add_minority (float): Probability of adding a comment for 'minority' proposals. Default is 0.05.
        - p_add_nimby (float): Probability of adding a comment for 'nimby' proposals. Default is 0.05.
        - p_add_other (float): Probability of adding a comment for 'other' proposals. Default is 0.05.
        - ran_st_nim (int): Random state for nimby group. Default is 1.
        - ran_st_min (int): Random state for minority group. Default is 11.
        - ran_st_oth (int): Random state for other group. Default is 111.
    """
    # Convert into lil for efficient row-wise operations
    # and copy to avoid modifying the original matrix
    mat2 = mat.tolil().copy().astype(int)
    rng_nim = np.random.default_rng(ran_st_nim)
    rng_min = np.random.default_rng(ran_st_min)
    rng_oth = np.random.default_rng(ran_st_oth)
    
    # Pre-compute the candidate columns for each type
    # (minority, nimby, other)
    mask_nim = np.isin(types_array, ['nimby'])
    candidate_cols_nim = np.nonzero(mask_nim)[0]  # index j array for nimby
    mask_min = np.isin(types_array, ['minority'])
    candidate_cols_min = np.nonzero(mask_min)[0]  # index j array for minority
    mask_other = np.isin(types_array, ['other'])
    candidate_cols_other = np.nonzero(mask_other)[0]  # index j array for other
    
    n_users = mat2.shape[0]
    for u in range(n_users):
        # rand vector for nimby and minority
        probs_nim = rng_nim.random(candidate_cols_nim.shape[0])
        probs_min = rng_min.random(candidate_cols_min.shape[0])
        probs_other = rng_oth.random(candidate_cols_other.shape[0])
        to_add_nim = candidate_cols_nim[probs_nim < p_add_nimby]
        to_add_min = candidate_cols_min[probs_min < p_add_minority]
        to_add_other = candidate_cols_other[probs_other < p_add_other]
        # for each candidate column, we add 1 to the user u
        for j in to_add_nim:
            mat2[u, j] = mat2[u, j] + 1
        for j in to_add_min:
            mat2[u, j] = mat2[u, j] + 1
        for j in to_add_other:
            mat2[u, j] = mat2[u, j] + 1

    return mat2.tocsr()


def strategy_drop_comments_by_group(mat, types_array, p_drop_minority=0.05, p_drop_nimby=0.05, p_drop_other=0.05, ran_st_nim=3, ran_st_min=33, ran_st_oth=333):
    """
    Drop comments (drop them to 0) from proposals of type '<group>' with a given probability if there is at least one comment.
    Args:
        - mat (scipy.sparse.csr_matrix): Input sparse matrix.
        - types_array (np.ndarray): Array of proposal types (e.g., 'other').
        - p_drop_other (float): Probability of dropping a comment for 'other' proposals. Default is 0.05.
        - p_drop_minority (float): Probability of dropping a comment for 'minority' proposals. Default is 0.05.
        - p_drop_nimby (float): Probability of dropping a comment for 'nimby' proposals. Default is 0.05.
        - ran_st_nim (int): Random state for nimby group. Default is 3.
        - ran_st_min (int): Random state for minority group. Default is 33.
        - ran_st_oth (int): Random state for other group. Default is 333.
    """
    
    # Convert into lil for efficient row-wise operations
    # and copy to avoid modifying the original matrix
    mat2 = mat.tolil().copy().astype(int)
    rng_nim = np.random.default_rng(ran_st_nim)
    rng_min = np.random.default_rng(ran_st_min)
    rng_oth = np.random.default_rng(ran_st_oth)
    
    # Pre-compute the candidate columns for each type
    # (minority, nimby, other)
    mask_nim = np.isin(types_array, ['nimby'])
    candidate_cols_nim = np.nonzero(mask_nim)[0]  # index j array for nimby
    mask_min = np.isin(types_array, ['minority'])
    candidate_cols_min = np.nonzero(mask_min)[0]  # index j array for minority
    mask_other = np.isin(types_array, ['other'])
    candidate_cols_other = np.nonzero(mask_other)[0]  # index j array for other
    
    n_users = mat2.shape[0]
    for u in range(n_users):
        # -- minority --
        active_minority = [j for j in candidate_cols_min if mat2[u, j] > 0]
        if active_minority:
            probs = rng_min.random(len(active_minority))
            to_drop = np.array(active_minority)[probs < p_drop_minority]
            for j in to_drop:
                mat2[u, j] = 0
        # -- nimby --
        active_nimby = [j for j in candidate_cols_nim if mat2[u, j] > 0]
        if active_nimby:
            probs = rng_nim.random(len(active_nimby))
            to_drop = np.array(active_nimby)[probs < p_drop_nimby]
            for j in to_drop:
                mat2[u, j] = 0
        # -- other --
        active_other = [j for j in candidate_cols_other if mat2[u, j] > 0]
        if active_other:
            probs = rng_oth.random(len(active_other))
            to_drop = np.array(active_other)[probs < p_drop_other]
            for j in to_drop:
                mat2[u, j] = 0

    return mat2.tocsr()


def strategy_put_one_comment_by_group(mat, types_array, p_minority=0.05, p_nimby=0.05, p_other=0.05, ran_st_nim=5, ran_st_min=55, ran_st_oth=555):
    """
    Put one comment (0->1) from proposals of type '<group>' with a given probability if there are still no comments.
    Args:
        - mat (scipy.sparse.csr_matrix): Input sparse matrix.
        - types_array (np.ndarray): Array of proposal types (e.g., 'other').
        - p_other (float): Probability of adding a comment for 'other' proposals. Default is 0.05.
        - p_minority (float): Probability of adding a comment for 'minority' proposals. Default is 0.05.
        - p_nimby (float): Probability of adding a comment for 'nimby' proposals. Default is 0.05.
        - ran_st_nim (int): Random state for nimby group. Default is 2.
        - ran_st (int): Random state for reproducibility. Default is 42.
    """
    
    # Convert into lil for efficient row-wise operations
    # and copy to avoid modifying the original matrix
    mat2 = mat.tolil().copy().astype(int)
    rng_nim = np.random.default_rng(ran_st_nim)
    rng_min = np.random.default_rng(ran_st_min)
    rng_oth = np.random.default_rng(ran_st_oth)
    
    # Pre-compute the candidate columns for each type
    # (minority, nimby, other)
    mask_nim = np.isin(types_array, ['nimby'])
    candidate_cols_nim = np.nonzero(mask_nim)[0]  # index j array for nimby
    mask_min = np.isin(types_array, ['minority'])
    candidate_cols_min = np.nonzero(mask_min)[0]  # index j array for minority
    mask_other = np.isin(types_array, ['other'])
    candidate_cols_other = np.nonzero(mask_other)[0]  # index j array for other
    
    n_users = mat2.shape[0]
    for u in range(n_users):
        # -- minority --
        active_minority = [j for j in candidate_cols_min if mat2[u, j] == 0]
        if active_minority:
            probs = rng_min.random(len(active_minority))
            to_add = np.array(active_minority)[probs < p_minority]
            for j in to_add:
                mat2[u, j] = 1
        # -- nimby --
        active_nimby = [j for j in candidate_cols_nim if mat2[u, j] == 0]
        if active_nimby:
            probs = rng_nim.random(len(active_nimby))
            to_add = np.array(active_nimby)[probs < p_nimby]
            for j in to_add:
                mat2[u, j] = 1
        # -- other --
        active_other = [j for j in candidate_cols_other if mat2[u, j] == 0]
        if active_other:
            probs = rng_oth.random(len(active_other))
            to_add = np.array(active_other)[probs < p_other]
            for j in to_add:
                mat2[u, j] = 1

    return mat2.tocsr()