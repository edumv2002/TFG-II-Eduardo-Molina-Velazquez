from implicit_extend.recommender_base import RecommenderBaseExt
from scipy.sparse import csr_matrix
import numpy as np
import random

class RandomRecommender(RecommenderBaseExt):

    def __init__(self):
        self.unrated_matrix = None

    def _shuffle_and_score_items(self):
        self.unrated_matrix = csr_matrix(np.ones((self.unrated_matrix.shape[0], self.unrated_matrix.shape[1]))\
                         - self.unrated_matrix)
        for le, ri in zip(self.unrated_matrix.indptr[:-1], self.unrated_matrix.indptr[1:]):
            indices = self.unrated_matrix.indices[le:ri]
            random.shuffle(indices)
            self.unrated_matrix.data[le:ri] = [1 / n for n in range(1, len(indices) + 1)]

    def fit(self, user_items):
        self.unrated_matrix = user_items.copy()
        self.unrated_matrix.data = self.unrated_matrix.data / self.unrated_matrix.data
        self._shuffle_and_score_items()

    def recommend(self, userid, user_items, N=10, filter_already_liked_items=True,
                  filter_items=None, recalculate_user=False, items=None):
        # Filters already liked items by default. Not implemented for filter_already_liked_items=False
        if not isinstance(user_items, csr_matrix):
            raise ValueError("user_items needs to be a CSR sparse matrix")

        if not np.isscalar(userid):
            if user_items.shape[0] != len(userid):
                raise ValueError("user_items must contain 1 row for every user in userids")

        ids, scores = self.top_n_idx(N, self.unrated_matrix, divide_scores=False)

        return ids, scores

    def similar_users(self, userid, N=10, filter_users=None, users=None):
        raise NotImplementedError("similar_users is not supported with this model")

    def similar_items(
            self, itemid, N=10, recalculate_item=False, item_users=None, filter_items=None, items=None
    ):
        raise NotImplementedError("similar_items is not supported with this model")

    def save(self, file):
        raise NotImplementedError("save is not supported with this model")

    def load(cls, fileobj_or_path):
        raise NotImplementedError("load is not supported with this model")
