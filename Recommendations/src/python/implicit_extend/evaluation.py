import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
from ipywidgets import IntProgress
from IPython.display import display


def ranking_metrics_at_k(model, train_user_items, test_user_items, K=10, show_progress=True,
                         filter_already_liked_items=True):
    """ Calculates ranking metrics for a given trained model
    Parameters
    ----------
    model : RecommenderBase
        The fitted recommendation model to test
    train_user_items : csr_matrix
        Sparse matrix of user by item that contains elements that were used
            in training the model
    test_user_items : csr_matrix
        Sparse matrix of user by item that contains withheld elements to
        test on
    K : int
        Number of items to test on
    show_progress : bool, optional
        Whether to show a progress bar
    filter_already_liked_items: bool, optional
        Wether to filter already liked items or not in recommendations.
    Returns
    -------
    dict
        A dictionary with the calculated metrics @ K:
        - Precision
        - Recall
        - MAP
        - NDCG
        - AUC
        - MRR
        - F1
    """
    if not isinstance(train_user_items, csr_matrix):
        train_user_items = train_user_items.tocsr()

    if not isinstance(test_user_items, csr_matrix):
        test_user_items = test_user_items.tocsr()

    users, items = test_user_items.shape[0], test_user_items.shape[1]

    user_metrics = pd.DataFrame(data={"userId": [i for i in range(0, users)],
                                      "precision": [None]*users,
                                      "recall": [None]*users,
                                      "map": [None]*users,
                                      f"ndcg@{K}": [None]*users,
                                      "auc": [None]*users,
                                      "mrr": [None]*users,
                                      "f1": [None]*users}).set_index("userId")

    # precision
    relevant, pr, rc = 0, 0, 0
    # map
    mean_ap, ap = 0, 0
    # ndcg
    cg = (1.0 / np.log2(np.arange(2, K + 2)))
    cg_sum = np.cumsum(cg)
    ndcg = 0
    # auc
    mean_auc = 0
    # mrr
    mrr = 0

    test_indptr = test_user_items.indptr
    test_indices = test_user_items.indices
    likes = set()

    if (K < 1) or (K > items):
        raise ValueError("The 'K' must in between [1, num items]")
    batch_size = int(np.ceil(0.25*users))
    start_idx = 0

    # get an array of userids that have at least one item in the test set
    to_generate = np.arange(users, dtype="int32")
    to_generate = to_generate[np.ediff1d(test_user_items.indptr) > 0]
    total_users = len(to_generate)
    progress = IntProgress(min=0, max=total_users)
    display(progress)

    recommend_flag = (model.__class__.__name__ == "CosineRecommenderUB") or \
                     (model.__class__.__name__ == "HybridRecommenderUB")

    print(model.__class__.__name__)
    while start_idx < total_users:
        batch = to_generate[start_idx: start_idx + batch_size]

        if recommend_flag:
            ids, scores = model.recommend(batch,
                                          train_user_items,
                                          N=K,
                                          filter_already_liked_items=filter_already_liked_items)
        else:
            ids, scores = model.recommend(batch,
                                          train_user_items[batch],
                                          N=K,
                                          filter_already_liked_items=filter_already_liked_items)

        start_idx += batch_size

        sorted_scores = np.argsort(scores, axis=1, kind='stable')
        sorted_scores = np.flip(sorted_scores, axis=1)
        ids = ids[np.arange(ids.shape[0])[:, None], sorted_scores]

        for batch_idx in range(len(batch)):
            u = batch[batch_idx]
            likes.clear()
            for i in range(test_indptr[u], test_indptr[u+1]):
                likes.add(test_indices[i])

            relevant = 0
            ap = 0
            hit = 0
            miss = 0
            auc = 0
            ndcg_u = 0
            mrr_u = 0

            idcg = cg_sum[min(K, len(likes)) - 1]
            num_pos_items = len(likes)
            num_neg_items = items - num_pos_items
            for i in range(K):
                if ids[batch_idx, i] in likes:
                    relevant += 1
                    hit += 1
                    ap += hit / (i + 1.0)
                    ndcg_u += cg[i] / idcg
                    if hit == 1:
                        mrr_u += 1 / (i + 1.0)
                elif ids[batch_idx, i] != -1 and scores[batch_idx, i] != 0:
                    miss += 1
                    auc += hit # Accumulated hit of previous positions in ranking

            user_metrics.loc[u, "precision"] = relevant / K
            user_metrics.loc[u, "recall"] = relevant / len(likes)
            user_metrics.loc[u, "map"] = ap / (1e-7 + min(K, len(likes)))
            user_metrics.loc[u, f"ndcg@{K}"] = ndcg_u
            user_metrics.loc[u, "auc"] = auc / (1e-7 + num_pos_items * num_neg_items)
            user_metrics.loc[u, "mrr"] = mrr_u
            if (relevant / K) * (relevant / len(likes)) == 0:
                user_metrics.loc[u, "f1"] = 0
            else:
                user_metrics.loc[u, "f1"] = 2 * ((relevant / K) * (relevant / len(likes))) / ((relevant / K) + (relevant / len(likes)))

        progress.value += len(batch)

    progress.close()
    return user_metrics