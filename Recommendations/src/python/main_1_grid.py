from implicit.cpu.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import CosineRecommender

from implicit_extend.popularity import PopularityRecommender, PopularityNumCommentsRecommender
from implicit_extend.random import RandomRecommender
from implicit_extend.nearest_neighbours_ub import CosineRecommenderUB
from implicit_extend.content_based import ContentBasedRecommender
from implicit_extend.hybrid import HybridRecommenderUB, HybridRecommenderIB

from dusa_function_lib import (
    build_db_name, get_n_for_ndcg,
    get_rm_train_test_info,
    get_item_category_info,
    get_item_location_info,
    tunning_and_metrics,
    ranking_metrics_at_k,
    build_directory_city_name,
    get_k_from_results,
)

# auxiliary functions
import pandas as pd
import os
import scipy.sparse as sps
import re
import numpy as np


# we won't use bpr for now
MODEL_ORDER = {
    "pop": 0, "pop_nc":1, "rand":2,
    "ib": 3, "ub":4, "mf": 5, "bpr": 6,
    "cb_cat": 7, "cb_loc": 8,
    "cbub_cat":9, "cbub_loc":10,
    "cbib_cat":11, "cbib_loc":12,
}
    

def main():
    cities = ['Cambridge', 'Miami', 'New York']
    years = ['2014', '2015', '2016', '2017']
    cvk = 5
    
    n_ndcg_dict = dict()
    nf_ndcg_dict = dict()
    
    all_data = dict()
    it_cat_data = dict()
    it_loc_data = dict()
    
    # strategy basic is already done
    strategies = ['add_global', 'add', 'add_in_zeros', 'drop', 'drop_add']
    
    tun_metrics = dict()
    final_metrics = dict()
    
    for city in cities:
        n_ndcg_dict[city] = dict()
        nf_ndcg_dict[city] = dict()
        city_directory_name = build_directory_city_name(city)
        for year in years:
            db_name = build_db_name(city, year)
            city_year_directory_name = f"{city_directory_name}/{year}"
            all_data[db_name] = dict()
            
            # We store the category data matrix in a dict
            it_cat_data[db_name] = dict()
            it_cat_info = get_item_category_info(city, year)
            it_cat_data[db_name]['it_cat_info'] = it_cat_info
            
            # same for Location data
            it_loc_data[db_name] = dict()
            it_loc_info = get_item_location_info(city, year)
            it_loc_data[db_name]['it_loc_info'] = it_loc_info
            
            tun_metrics[db_name] = dict()
            print("#" * 50)
            print(f"Starting {db_name}")
            print("#" * 50)
            for strateg in strategies:
                print(f"\t- Starting strategy: {strateg}")
                ######################################################################
                # starting the data structures for each strategy
                ######################################################################
                # all data dict
                all_data[db_name][strateg] = dict()
                rm_info, rm_train, rm_test = get_rm_train_test_info(city, year, strategy=strateg)
                all_data[db_name][strateg]['rm_info'] = rm_info
                all_data[db_name][strateg]['rm_train'] = rm_train
                all_data[db_name][strateg]['rm_test'] = rm_test  ##### AQUÍ #####
                
                
                
                # n_ndcg and nf_ndcg dicts
                number_of_proposals = all_data[db_name][strateg]['rm_info']['rm'].shape[1]
                n_ndcg = get_n_for_ndcg(30, 100, number_of_proposals, 0.15)
                nf_ndcg = get_n_for_ndcg(30, 100, number_of_proposals, 0.20)
                n_ndcg_dict[city][year] = n_ndcg
                nf_ndcg_dict[city][year] = nf_ndcg
                
                tun_metrics[db_name][strateg] = dict()
                #######################################################################
                # DEFINING THE MODELS
                #######################################################################
                num_users = rm_info['rm'].shape[0]
                
                MODEL_CONFIGS = {
                    "pop": {
                        "pop": {
                            'm': PopularityRecommender, 
                            'params': {}
                        },
                    },
                    "pop_nc": {
                        "pop_nc": {
                            'm': PopularityNumCommentsRecommender,
                            'params': {},
                        },
                    },
                    "rand": {
                        "rand": {
                            'm': RandomRecommender,
                            'params': {},
                        }
                    },
                    "mf": {
                        "mf": {
                            'm':LogisticMatrixFactorization,
                            'params':{
                                'factors': [i for i in range(5, 41, 5)]
                            },
                        },
                    },
                    "ib": {
                        "ib": {
                            'm':CosineRecommender,
                            'params':{
                                "K": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100]
                            },
                        },
                    },
                    "ub": {
                        "ub": {
                            'm': CosineRecommenderUB,
                            'params':{
                                "K": [5, 10, 20, 30, 50, 100]+[i for i in range(150, 301, 50)]
                            }
                        },
                    },
                    "cb_cat": {
                        "cb_cat": {
                            'm':ContentBasedRecommender, 
                            'params':{
                                'it_cat_matrix': [it_cat_data[db_name]['it_cat_info']['it_cat']],
                            },
                        },
                    },
                    "cb_loc": {
                        "cb_loc": {
                            'm':ContentBasedRecommender, 
                            'params':{
                                'it_cat_matrix': [it_loc_data[db_name]['it_loc_info']['it_loc']],
                            },
                        },
                    },
                    "cbub_cat": {
                        "cbub_cat": {
                            'm':HybridRecommenderUB, 
                            'params':{
                                'it_cat_matrix': [it_cat_data[db_name]['it_cat_info']['it_cat']],
                                'tag': ['category'],
                                "K": [k for k in [5] + list(range(50, num_users, 50)) + [1000, 2000, 6000] if k < num_users],
                            },
                        },
                    },
                    "cbub_loc": {
                        "cbub_loc": {
                            'm':HybridRecommenderUB, 
                            'params':{
                                'it_cat_matrix': [it_loc_data[db_name]['it_loc_info']['it_loc']],
                                'tag': ['category'],
                                "K": [k for k in [5] + list(range(50, num_users, 50)) + [1000, 2000, 6000] if k < num_users],
                            },
                        },
                    },
                    "cbib_cat": {
                        'cbib_cat': {
                            'm':HybridRecommenderIB, 
                            'params':{
                                'it_cat_matrix': [it_cat_data[db_name]['it_cat_info']['it_cat']],
                                'tag': ['category'],
                                "K": [k for k in list(range(10, 100, 10)) + list(range(100, 201, 20)) if k < num_users],
                            },
                        },
                    },
                    "cbib_loc": {
                        'cbib_loc': {
                            'm':HybridRecommenderIB, 
                            'params':{
                                'it_cat_matrix': [it_loc_data[db_name]['it_loc_info']['it_loc']],
                                'tag': ['category'],
                                "K": [k for k in list(range(10, 100, 10)) + list(range(100, 201, 20)) if k < num_users],
                            },
                        },
                    },
                }
                
                for model_name, model_dict in MODEL_CONFIGS.items():
                    tun_metrics[db_name][strateg][model_name] = dict()
                    
                    if model_name in ['ib', 'ub']:
                        rm_train = sps.csr_matrix(rm_train.astype(np.float32))
                        rm_test = sps.csr_matrix(rm_test.astype(np.float32))
                    
                    _, train_res, test_res, _ = \
                        tunning_and_metrics(
                            rm_train,
                            rm_test,
                            cvk=cvk,
                            N=n_ndcg_dict[city][year],
                            model=model_dict,
                            check_overfitting=True,
                        )
                    
                    tun_metrics[db_name][strateg][model_name]['test_res'] = test_res
                    tun_metrics[db_name][strateg][model_name]['train_res'] = train_res
                    tun_directory = f"../../data/tuning_results/{city_year_directory_name}/{strateg}"
                    os.makedirs(tun_directory, exist_ok=True)
                    if test_res is not None:
                        test_res.to_csv(f"{tun_directory}/test_res_{model_name}.csv", sep='|')
                    if train_res is not None:
                        train_res.to_csv(f"{tun_directory}/train_res_{model_name}.csv", sep='|')
                    

                    if model_name in ['mf', 'ib', 'ub', 'cbub_cat', 'cbub_loc', 'cbib_cat', 'cbib_loc']:
                        best_k = get_k_from_results(test_res, n_ndcg_dict[city][year])
                        if best_k is None:
                            try:
                                best_k = int(test_res.index[0][len(model_name):])
                            except:
                                best_k = 100
                            print(f"\t\033[31m No best_k found for {db_name}-->{model_name}-->{strateg}\033[0m")
                            print(f"\t\033[31m We take instead {best_k} \033[0m")
                        
                        if model_name in ['mf']:
                            NF=nf_ndcg_dict[city][year]
                            m = LogisticMatrixFactorization(factors=best_k)
                        elif model_name in ['cbub_cat', 'cbub_loc', 'cbib_cat', 'cbib_loc']:
                            NF = nf_ndcg_dict[city][year]
                            m = model_dict[model_name]['m'](
                                K=best_k,
                                it_cat_matrix=model_dict[model_name]['params']['it_cat_matrix'][0],
                                tag=model_dict[model_name]['params']['tag'][0],
                            )
                        elif model_name in ['ib', 'ub']:
                            NF = min(max(round(1.2*best_k), 5), 100) # BEST_NF
                            m = model_dict[model_name]['m'](K=best_k)
                    
                    elif model_name in ['cb_cat', 'cb_loc']:
                        NF=nf_ndcg_dict[city][year]
                        m = model_dict[model_name]['m'](model_dict[model_name]['params']['it_cat_matrix'][0])
                    else:
                        NF=nf_ndcg_dict[city][year]
                        m = model_dict[model_name]['m']()
                        
                    m.fit(rm_train)
                    final_metrics = pd.DataFrame(
                        data=ranking_metrics_at_k(m, rm_train, rm_test, K=NF).mean().to_dict(),
                        index=[model_name],
                    )
                        
                    final_directory = f"../../data/final_metrics/{city_year_directory_name}/{strateg}"
                    os.makedirs(final_directory, exist_ok=True)
                    final_metrics.to_csv(f"{final_directory}/{model_name}_final_metrics.csv", sep='|')  

if __name__ == "__main__":
    main()