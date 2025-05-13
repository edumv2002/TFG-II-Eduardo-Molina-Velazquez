from scipy.sparse import csr_matrix
import numpy as np
from tqdm.auto import tqdm

from implicit.nearest_neighbours import normalize
from implicit_extend.recommender_base import RecommenderBaseExt
from implicit._nearest_neighbours import all_pairs_knn


class UserUserRecommender(RecommenderBaseExt):
    """Base class for User-User Nearest Neighbour recommender models.

    Attributes
    ----------

    similarity: csr_matrix
        A sparse matrix of shape (users, users). That contains the score (cosine similarity)
        between user profiles. Will be calculated on the fit method.

    K : int, optional
        The number of neighbours to include when calculating the user-user
        similarity matrix.
    """

    def __init__(self, K=20):
        self.similarity = None
        self.K = K

    def fit(self, weighted, show_progress=True):
        """Computes and stores the similarity matrix.
        It is not exactly a similarity matrix, because only the K
        highest similarities for each user, excluding its similarity
        with their own selves, are non-zero. The rest of the entries
        are set to zero so that they are not taken into account when
        calculating the recommendations for each user.

        Parameters
        ----------
        weighted: csr_matrix
            A sparse matrix of shape (user, items), already precomputed to be turned into a
            similarity matrix here.
        show_progress: bool, optional
            Whether to show a progress bar or not.
    """

        #similarity = weighted*weighted.T # se quita al usar el all_pairs de implicit
        if weighted.shape[0] < self.K:
            raise ValueError("Parameter K, number of neighbours, is bigger than the " +\
                             "total number of users. Create a model with a smaller K.")

        self.similarity = all_pairs_knn(weighted.T, K=self.K+1, show_progress=show_progress).tocsr()
        self.similarity.setdiag(0)
        self.similarity.eliminate_zeros()

        return self.similarity

    def recommend(self, userid, user_items, N=10, filter_already_liked_items=True, filter_items=None,
                  recalculate_user=False, items=None, sort=False):
        """ Recommends items for users following a User Based Nearest Neighbours
        algorithm.

        This method allows you to calculate the top N recommendations for a user or
        batch of users. Passing an array of userids instead of a single userid will
        tend to be more efficient, and allows multi-thread processing on the CPU.

        This method has options for filtering out items that have already been liked
        by the user with the filter_already_liked_items parameter.
        The filter_items and items parameters in the RecommenderBase parent class have
        not been implemented and are ignored.

        Example usage:

            # calculate the top recommendations for a single user
            ids, scores = model.recommend(0, user_items[0])

            # calculate the top recommendations for a batch of users
            userids = np.arange(10)
            ids, scores = model.recommend(userids, user_items[userids])

        Parameters
        ----------
        userid : Union[int, array_like]
            The userid or array of userids to calculate recommendations for
        user_items : csr_matrix
            A sparse matrix of shape (users, number_items). This lets us look
            up the liked items and their weights for the user. This is used to filter out
            items that have already been liked from the output, and to also potentially
            recalculate the user representation. Each row in this sparse matrix corresponds
            to a row in the userid parameter: that is the first row in this matrix contains
            the liked items for the first user in the userid array.
        N : int, optional
            The number of results to return
        filter_already_liked_items: bool, optional
            When true, don't return items present in the training set that were rated
            by the specified user.
        filter_items : array_like, optional
            Ignored
        recalculate_user : bool, optional
            Ignored
        items: array_like, optional
            Ignored

        Returns
        -------
        tuple
            Tuple of (itemids, scores) arrays. When calculating for a single user these array will
            be 1-dimensional with N items. When passed an array of userids, these will be
            2-dimensional arrays with a row for each user.
        """
        if not isinstance(user_items, csr_matrix):
            raise ValueError("user_items needs to be a CSR sparse matrix")

        if user_items.shape[0] != self.similarity.shape[0]:
            raise ValueError("user_items must contain the same number of users as "+\
                             "the matrix used for fitting.")

        # Each entry of the matrix is the new-calculated score
        score_matrix = self.similarity * user_items

        if filter_already_liked_items:
            score_matrix = self.filter_already_liked_items(score_matrix[userid, :],
                                                           user_items[userid, :])
        else:
            score_matrix = score_matrix[userid, :]

        ids, scores = self.top_n_idx(N, score_matrix, sort=sort)

        return ids, scores

    def similar_users(self, userid, N=10, filter_users=None, users=None):
        raise NotImplementedError("similar_users is not supported with this model")

    def similar_items(self, itemid, N=10, recalculate_item=False, item_users=None,
                      filter_items=None, items=None):
        raise NotImplementedError("similar_items is not supported with this model")

    def save(self, file):
        raise NotImplementedError("save is not supported with this model")

    def load(cls, fileobj_or_path):
        raise NotImplementedError("load is not supported with this model")

class CosineRecommenderUB(UserUserRecommender):
    """An Item-Item Recommender on Cosine distances between items"""

    def fit(self, counts, show_progress=True):
        # cosine distance is just the dot-product of a normalized matrix
        return super().fit(normalize(counts), show_progress)