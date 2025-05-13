from scipy.sparse import csr_matrix
from implicit.nearest_neighbours import CosineRecommender
from implicit_extend.nearest_neighbours_ub import CosineRecommenderUB
from implicit_extend.content_based import ContentBasedRecommender

class HybridRecommenderUB(CosineRecommenderUB, ContentBasedRecommender):
    """Base class for Hybrid recommender. It is like the User-Based Nearest Neighbours recommender,
    but the similarity matrix is the cosine similarity matrix of the user profiles based on the
    content.

    Attributes
    ----------
    item_profiles : csr_matrix
        A sparse matrix of shape (items, number_tags). That contains the weights
        given to each tag per item.
    user_profiles: csr_matrix
        A sparse matrix of shape (users, number_tags). That contains the weights
        given to each tag per user. Will be calculated on the fit method.
    similarity: csr_matrix
        A sparse matrix of shape (users, users). That contains the score (cosine similarity)
        between user profiles. Will be calculated on the fit method.
    tag: str
        name of the tag ("category", "topic" or "location").
    """
    def __init__(self, it_cat_matrix, tag='category', K=20):
        self.item_profiles = it_cat_matrix
        self.user_profiles = None
        self.tag = tag
        super().__init__(K=K)

    def fit(self, user_items):
        self.user_profiles = super()._generate_user_profiles(user_items)
        return super().fit(self.user_profiles)

class HybridRecommenderIB(CosineRecommender):
    """Base class for Hybrid recommender. It is like the Item-Based Nearest Neighbours recommender,
    but the similarity matrix is the cosine similarity matrix of the item profiles based on the
    content.
        Parameters
        ----------
        item_profiles : csr_matrix
            A sparse matrix of shape (items, number_tags). That contains the weights
            given to each tag per item
        similarity: csr_matrix
            A sparse matrix of shape (items, items). That contains the (cosine similarity)
            between item profiles. Will be calculated on the fit method.
        tag: str
            name of the tag ("category", "topic" or "location").
    """

    def __init__(self, it_cat_matrix, tag='category', K=20):
        self.item_profiles = it_cat_matrix
        self.tag = tag
        super().__init__(K=K)

    def fit(self, user_items):
        return super().fit(self.item_profiles.T)

    def similar_users(self, userid, N=10, filter_users=None, users=None):
        raise NotImplementedError('similar_users is not supported with this model')