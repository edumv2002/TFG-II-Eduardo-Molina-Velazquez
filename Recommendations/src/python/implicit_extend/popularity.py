from implicit_extend.recommender_base import RecommenderBaseExt
from scipy.sparse import csr_matrix, diags
import numpy as np


class PopularityRecommender(RecommenderBaseExt):
    """ Popularity Recommender:
        - Each user is recommended a list of size K of the K most popular items in the system, unseen by the user.
        - The popularity of an item corresponds to the number of users that have ever commented on it.
    """

    def __init__(self, num_threads=0):
        self.pop_scores = None
        self.num_threads = num_threads
        self.pop_matrix = None

    def fit(self, train_user_items, show_process=True):
        train_user_items = train_user_items.copy()
        train_user_items.data = train_user_items.data / train_user_items.data
        self.pop_scores = csr_matrix(train_user_items.sum(axis=0))
        return self.pop_scores.copy()

    def recommend(self, userid, user_items, N=10, filter_already_liked_items=True,
                  filter_items=None, recalculate_user=False, items=None):
       
        if not isinstance(user_items, csr_matrix):
            raise ValueError("user_items needs to be a CSR sparse matrix")

        if not np.isscalar(userid):
            if user_items.shape[0] != len(userid):
                raise ValueError("user_items must contain 1 row for every user in userids")

        user_items = user_items.copy()
        user_items.data = user_items.data/user_items.data
        self.pop_matrix = diags(self.pop_scores.toarray()[0], 0).tocsr()

        if filter_already_liked_items:
            self.pop_matrix = (csr_matrix(np.ones((user_items.shape[0], user_items.shape[1]))) - user_items)\
                              * self.pop_matrix
        else:
            self.pop_matrix = csr_matrix(np.ones((user_items.shape[0], user_items.shape[1])))\
                              * self.pop_matrix

        ids, scores = self.top_n_idx(N, self.pop_matrix, divide_scores=True)

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


class PopularityNumCommentsRecommender(PopularityRecommender):
    
    """ Popularity Recommender:
        - Each user is recommended a list of size K of the K most popular items in the system, unseen by the user.
        - The popularity of an item corresponds to the total number of comments made on that item.
    """

    def fit(self, train_user_items, show_process=True):
        self.pop_scores = csr_matrix(train_user_items.sum(axis=0))
        return self.pop_scores.copy()