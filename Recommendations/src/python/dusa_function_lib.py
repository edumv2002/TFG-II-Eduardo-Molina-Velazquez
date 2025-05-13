import pandas as pd
import csv
from scipy.sparse import load_npz
import scipy.sparse as sps
from implicit_extend.evaluation import ranking_metrics_at_k
from implicit.evaluation import train_test_split
from itertools import chain
from datetime import datetime
import random
from sklearn.model_selection import ParameterGrid
import numpy as np
import os
import re

################################################################
##### DATABASE NAMES AUXILIARY FUNCTIONS ###############################
################################################################
def build_db_name(city, year):
    db_name = ""
    if city == "New York" or city == "New York City" or city == "New-York":
        db_name = f"NewYorkCity-{year}"
    elif city == "Cambridge":
        db_name = f"Cambridge-{year}"
    elif city == "Miami":
        db_name = f"Miami-{year}"
    return db_name

def build_directory_city_name(city):
    directory_city_name = ""
    if city == "New York" or city == "New York City" or city == "New-York":
        directory_city_name = "newyork"
    elif city == "Cambridge":
        directory_city_name = "cambridge"
    elif city == "Miami":
        directory_city_name = "miami"
    return directory_city_name

################################################################
#### DECIDING PARAMETERS FOR RECOMMENDERS METRICS ##############
################################################################
def get_n_for_ndcg(min_n, max_n, num_proposals, percentage=0.15):
    """Decide the number of proposals to take into account in ndcg@N

    Args:
        min_n (int): _description_
        max_n (int): _description_
        num_proposals (int): _description_
        percentage (float, optional): Percentage of proposals. Defaults to 0.15.

    Returns:
        int: number of proposals N to take into account in ndcg@N
    """
    n = round(percentage * num_proposals)
    if n < min_n:
        n = min_n
    elif n > max_n:
        n = max_n
    return n


def get_k_from_results(dataframe, n_ndcg):
    
    try:
        best_index = dataframe[f'ndcg@{n_ndcg}'].idxmax()
        best_k = int(re.search(r'\d+', best_index).group())
        return best_k
    except:
        return None

################################################################
#### SAVING RECOMMENDATIONS AND CALCULATING METRICS ############
################################################################
def get_rm_train_test_info(city, year, strategy='basic'):
    directory_city_name = build_directory_city_name(city)
    
    data_dir = f"../../data/rm/{directory_city_name}/{year}"
    
    rm_info = dict()
    
    rm_info['rm'] = load_npz(f"{data_dir}/basic/num_comm_matrix.npz")
    rm_info['user_mapping'] = pd.read_csv(f"{data_dir}/{strategy}/user_mapping.csv", sep="|")
    rm_info['item_mapping'] = pd.read_csv(f"{data_dir}/{strategy}/item_mapping.csv", sep="|")
    
    rm_train = load_npz(f"{data_dir}/{strategy}/num_comm_train.npz").astype(np.int32)
    rm_test = load_npz(f"{data_dir}/basic/num_comm_test.npz").astype(np.int32)
    
    return rm_info, rm_train, rm_test

def get_item_category_info(city, year):
    directory_city_name = build_directory_city_name(city)
    
    data_dir = f"../../data/category/{directory_city_name}/{year}"
    
    it_cat_info = dict()
    
    it_cat_info["it_cat"] = load_npz(f"{data_dir}/item_category_matrix.npz")
    it_cat_info["it_mapping"] = pd.read_csv(f"{data_dir}/it_cat_item_mapping.csv", sep="|")
    it_cat_info["cat_mapping"] = pd.read_csv(f"{data_dir}/it_cat_category_mapping.csv", sep="|")
    
    return it_cat_info

def get_item_location_info(city, year):
    db_name = build_db_name(city, year)
    directory_city_name = build_directory_city_name(city)
    
    data_dir = f"../../data/location/{directory_city_name}/{year}"
    
    it_loc_info = dict()
    
    it_loc_info["it_loc"] = load_npz(f"{data_dir}/item_cluster_matrix.npz")
    it_loc_info["it_mapping"] = pd.read_csv(f"{data_dir}/it_loc_item_mapping.csv", sep="|")
    it_loc_info["loc_mapping"] = pd.read_csv(f"{data_dir}/it_loc_cluster_mapping.csv", sep="|")
    
    return it_loc_info

def get_ratings_df(rm, rm_info):
    df = pd.DataFrame(data={'userId': list(chain(*[[i] * (j_1 - j_0)
                                                   for j_0, j_1, i in zip(rm.indptr[0:-1],
                                                                          rm.indptr[1:],
                                                                          range(len(rm.indptr)))])),
                            'itemId': rm.indices,
                            'numComments': rm.data})\
        .replace(to_replace={'userId': dict(zip(rm_info['user_mapping'].new_userId,
                                                rm_info['user_mapping'].userId)),
                             'itemId': dict(zip(rm_info['item_mapping'].new_itemId,
                                                    rm_info['item_mapping'].itemId))})
    df.numComments = df.numComments.astype('int64')
    return df
    
################################################################
#### HYPERPARAMETER TUNNING ####################################
################################################################
def tunning_and_metrics(rm_train, rm_test, model={}, cvk=5, N=50, Nf=None, check_overfitting=False):

    if Nf is None:
        Nf=N
    model_name = list(model.keys())[0]
    algorithm = list(model.values())[0]
    params = algorithm['params']

    all_ht_metric_results_train = None
    all_ht_metric_results_test = None

    if algorithm['m'].__name__ == "ContentBasedRecommender":
        best_params = dict(zip(params.keys(), [v[0] for v in params.values()]))
        model_name2 = model_name
        model = algorithm['m'](**best_params)

    elif algorithm['m'].__name__ == "PopularityRecommender" \
            or algorithm['m'].__name__ == "PopularityNumCommentsRecommender"\
            or algorithm['m'].__name__ == "RandomRecommender":
        model_name2 = model_name
        model = algorithm['m']()

    else:
        if check_overfitting:
            send_test = rm_test
        else:
            send_test = None
        all_ht_metric_results_train, all_ht_metric_results_test, best_params =\
            hyperparameter_tunning_CV(rm_train, model_name, algorithm['m'],
                                      params=params, cvk=cvk, N=N,
                                      check_overfitting=check_overfitting, rm_test=send_test)
        model_name2 = model_name + ''.join(list(chain.from_iterable([str(v) for v, k in zip(best_params.values(),
                                                                                            best_params.keys())
                                                                     if k != 'prop_tag_matrix' and k != 'tag'])))
        model = algorithm['m'](**best_params)

    model.fit(rm_train)

    final_metrics_users = ranking_metrics_at_k(model, rm_train, rm_test, K=Nf)
    final_metric_results = pd.DataFrame(data=final_metrics_users.mean().to_dict(),
                                 index=[model_name2])

    return final_metric_results, all_ht_metric_results_train, all_ht_metric_results_test, final_metrics_users

def hyperparameter_tunning_CV(rm_train, model_name, algorithm, params=None, cvk=1, N=50,
                              check_overfitting=False, rm_test=None):

    random_seeds = random.sample(range(0, 2 ** 32 - 1), cvk)
    print('Random Seeds', random_seeds)
    metric_results = []
    metrics_test = []

    param_grid = ParameterGrid(params)
    best_ndcg, best_params = -1, {}

    for params in param_grid:
        model_name2 = model_name + ''.join(list(chain.from_iterable([str(v) for v, k in zip(params.values(),
                                                                                            params.keys())
                                                                     if k != 'prop_tag_matrix' and k != 'tag'])))
        model = algorithm(**params)
        metrics = cv_recsys(model, model_name2, rm_train, random_seeds, cvk, N)
        metric_results.append(metrics)
        if best_ndcg < metrics[f'ndcg@{N}'].values[0]:
            best_ndcg = metrics[f'ndcg@{N}'].values[0]
            best_params = params

        if check_overfitting:
            model.fit(rm_train)
            metrics_test.append(pd.DataFrame(data=ranking_metrics_at_k(model, rm_train, rm_test, K=N).mean().to_dict(),
                                             index=[model_name2]))

    metric_results = pd.concat(metric_results)
    if check_overfitting:
        metrics_test_results = pd.concat(metrics_test)
    else:
        metrics_test_results = None

    print("Best params are -> ", best_params)

    return metric_results, metrics_test_results, best_params


################# SAVING RECOMMENDATIONS AND CALCULATING METRICS #################

def gen_recommendations(rm_info, rm_train=None, rm_train_perturbed=None, rm_test=None, model_name='pop', model=None, params={},
                        city="Cambridge", year='2014', N=50, strategy='basic', save=False):
    citypath = build_directory_city_name(city)
    dataset_name = build_db_name(city, year)
    
    m = model(**params)
    if rm_train is None or rm_test is None:
        print("ERROR: must provide train and test")
    if rm_train_perturbed is not None:
        rm_train_perturbed = sps.csr_matrix(rm_train_perturbed.astype(np.float32))
        m.fit(rm_train_perturbed)
    else:
        m.fit(rm_train)
    
    ids, scores = m.recommend(userid=range(0, rm_train.shape[0]),
                              user_items=rm_train.astype(float),
                              N=N,
                              filter_already_liked_items=True)

    recs = pd.DataFrame(data={'userId': list(chain(*[[i] * ids.shape[1] for i in range(0, ids.shape[0])])),
                              'itemId': ids.flatten(),
                              'scores': scores.flatten()}) \
        .sort_values(by=['userId', 'scores'], ascending=[True, False]) \
        .reset_index(drop=True) \
        .replace(to_replace={'userId': dict(zip(rm_info['user_mapping'].new_userId,
                                                rm_info['user_mapping'].userId)),
                             'itemId': dict(zip(rm_info['item_mapping'].new_itemId,
                                                    rm_info['item_mapping'].itemId))})
    recs = recs[recs.itemId != -1].reset_index(drop=True)
    if save:
        params.pop('prop_tag_matrix', None)
        params.pop('tag', None)

        initial_save_path = f"../../data/recommendations/{citypath}/{year}"
        os.makedirs(f"{initial_save_path}/{strategy}", exist_ok=True)
        
        recs.to_csv(f"{initial_save_path}/{strategy}/rec_{model_name}.csv",
                    sep='|', encoding='utf-8', index_label=False, index=False)

        try:
            if strategy == 'basic':
                f = f'../../data/recommendations/{citypath}/model_history.csv'
            else:
                f = f'../../data/recommendations/{citypath}/model_history_{strategy}.csv'
            
            history = pd.read_csv(f, sep='|', encoding='utf-8', parse_dates=[3])
            history = pd.concat([history,
                                pd.DataFrame(data={'model_name': [model_name],
                                                    'year':[year],
                                                    'params':
                                                        [params[list(params.keys())[0]]] if len(params)>0 else [''],
                                                    'date': [datetime.now()]})])\
                .sort_values(by=['model_name', 'date', 'year'])
            history.to_csv(f, sep='|', encoding='utf-8', index_label=False, index=False)

        except FileNotFoundError:
            pd.DataFrame(data={'model_name': [model_name],
                            'year':[year],
                            'params': [params[list(params.keys())[0]]] if len(params)>0 else [''],
                            'date': [datetime.now()]})\
                .to_csv(f, sep='|', encoding='utf-8', index_label=False, index=False)

    return recs


def cv_recsys(model, model_name, rm, random_seeds, cvk=5, N=50):
    """ Cross Validation experiment to get mean metrics"""

    metrics = pd.DataFrame(data={'precision': 0.0,
                                 'recall': 0.0,
                                 'f1': 0.0,
                                 'map': 0.0,
                                 f'ndcg@{N}': 0.0,
                                 'auc': 0.0,
                                 'mrr': 0.0}, index=[model_name])

    ndcg = []
    print(f"{model_name}")
    for i in range(0, cvk):
        print(f"Iter {i+1} ", end="")
        rm_train, rm_test = train_test_split(rm, train_percentage=0.8, random_state=random_seeds[i])
        rm_train = rm_train.astype(np.float32)
        rm_test = rm_test.astype(np.float32)
        
        model.fit(rm_train)
        
        metrics_small = ranking_metrics_at_k(model, rm_train, rm_test, K=N).mean().to_dict()
        metrics = metrics + pd.DataFrame(data=metrics_small, index=[model_name])

        ndcg.append(metrics_small[f'ndcg@{N}'])
    metrics = metrics / cvk
    metrics[f'std_ndcg@{N}'] = np.std(ndcg)
    if metrics[f'ndcg@{N}'].values[0] != 0:
        metrics[f'var_coef_ndcg@{N}'] = metrics[f'std_ndcg@{N}'] / abs(metrics[f'ndcg@{N}'])
    else:
        metrics[f'var_coef_ndcg@{N}'] = float('inf')

    return metrics