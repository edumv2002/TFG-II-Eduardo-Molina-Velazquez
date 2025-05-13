from scipy.sparse import csr_matrix
import numpy as np
from implicit.nearest_neighbours import normalize
from implicit_extend.recommender_base import RecommenderBaseExt

class ContentBasedRecommender(RecommenderBaseExt):
    """Base class for Content Based recommender models.

    Attributes
    ----------
    item_profiles : csr_matrix
        A sparse matrix of shape (items, number_tags). That contains the weights
        given to each tag per item.
    user_profiles: csr_matrix
        A sparse matrix of shape (users, number_tags). That contains the weights
        given to each tag per user. Will be calculated on the fit method.
    similarity_scores : csr_matrix
        A sparse matrix of shape (users, items). That contains the score (cosine similarity)
        given to each item per user. Will be calculated during the fit method.
    tag: str
        name of the tag ("category", "topic" or "location").
    """

    def __init__(self, it_cat_matrix, tag='category'):
        self.item_profiles = it_cat_matrix
        self.user_profiles = None
        self.similarity_scores = None
        self.tag = tag

    def fit(self, user_items):
        self.user_profiles = self._generate_user_profiles(user_items[:])
        self.similarity_scores = normalize(self.user_profiles) * normalize(self.item_profiles).T

    def _generate_user_profiles(self, rm):
        rm.data = rm.data/rm.data
        counts = np.diff(rm.indptr)
        val = np.repeat(counts, rm.getnnz(axis=1))
        rm.data /= val

        return (rm * self.item_profiles).tocsr()

    def recommend(self, userid, user_items, N=10, filter_already_liked_items=True, filter_items=None,
                  recalculate_user=False, items=None, sort=False):

        if (np.isscalar(userid) and userid >= self.similarity_scores.shape[0]) or \
           (~np.isscalar(userid) and max(userid) >= self.similarity_scores.shape[0]):
            raise ValueError("userid must be a valid scalar or array for the user indexes"
                             " in the rating matrix.")

        if filter_already_liked_items:
            score_matrix = self.filter_already_liked_items(self.similarity_scores[userid, :],
                                                           user_items)
        else:
            score_matrix = self.similarity_scores[userid, :]

        ids, scores = self.top_n_idx(N, score_matrix, sort=sort)

        return ids, scores

    def similar_users(self, userid, N=10, filter_users=None, users=None):
        raise NotImplementedError("similar_users is not supported with this model.")

    def similar_items(self, itemid, N=10, recalculate_item=False, item_users=None,
                      filter_items=None, items=None):
        raise NotImplementedError("similar_items is not supported with this model.")

    def save(self, file):
        raise NotImplementedError("save is not supported with this model.")

    def load(cls, fileobj_or_path):
        raise NotImplementedError("load is not supported with this model.")
