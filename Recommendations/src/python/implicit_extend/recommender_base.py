from abc import ABCMeta
from implicit.recommender_base import RecommenderBase
import numpy as np
from scipy.sparse import csr_matrix

class RecommenderBaseExt(RecommenderBase, metaclass=ABCMeta):

    @staticmethod
    def filter_already_liked_items(score_matrix, user_items):
        user_items.data = user_items.data/user_items.data
        opposed = np.ones((user_items.shape[0], user_items.shape[1])) - user_items
        opposed = score_matrix.multiply(opposed)

        mask = opposed.data < 0
        opposed.data[mask] = 0

        opposed.eliminate_zeros()

        return csr_matrix(opposed)

    @staticmethod
    def top_n_idx(N, score_matrix, divide_scores=False, sort=True):
        # Return index and values of top n values in each row of a sparse matrix
        top_n_ids = []
        top_n_scores = []

        if divide_scores:
            divide = score_matrix.shape[0]
        else:
            divide = 1

        for le, ri in zip(score_matrix.indptr[:-1], score_matrix.indptr[1:]):
            N_row_pick = min(N, ri - le)
            idx = le + np.argpartition(score_matrix.data[le:ri], -N_row_pick)[-N_row_pick:]
            top_n_ids = [*top_n_ids,
                         np.concatenate((score_matrix.indices[idx],
                                         np.repeat([-1], N - N_row_pick, axis=0)))]
            top_n_scores = [*top_n_scores,
                            np.concatenate((score_matrix.data[idx] / divide,
                                            np.repeat([-1], N - N_row_pick, axis=0)))]

        ids = np.array(top_n_ids, dtype='int32')
        scores = np.array(top_n_scores, dtype='float32')

        # We sort ids and scores from maximum to minimum score
        if sort:
            sorted_scores = np.argsort(scores, axis=1, kind='stable')
            sorted_scores = np.flip(sorted_scores, axis=1)
            ids = ids[np.arange(ids.shape[0])[:, None], sorted_scores]
            scores = scores[np.arange(scores.shape[0])[:, None], sorted_scores]

        return ids, scores