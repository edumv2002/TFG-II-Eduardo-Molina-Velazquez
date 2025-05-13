from dusa_function_lib import (
    build_db_name, 
    build_directory_city_name,
)

from fairness_evaluation import (
    create_dataframes,
    fairness_results,
)

import pandas as pd
import os

def main():
    TYPES_ORDER = {"minority":0, "nimby":1, "no_type":2}
    MODEL_ORDER = {"rand":0, "pop": 1, "pop_nc":2,
               "ib": 3, "ub":4, "mf": 5,
               "cb_cat": 6, "cb_loc":7,
               "cbib_cat":8, "cbib_loc":9,
               "cbub_cat":10, "cbub_loc":11}
    
    cities = ['Cambridge', 'Miami', 'New York']
    #cities = ['Cambridge']
    years = ['2014', '2015', '2016', '2017']
    #years = ['2014']
    # strategy basic is already done
    strategies = ['add_global', 'add', 'add_in_zeros', 'drop', 'drop_add']
    #strategies = ['add']
    types = dict()
    item_mapping = dict()
    types_prop = dict()
    df_test_types = dict()
    df_train_types = dict()
    gce_results = dict()
    pm = dict()
    pf = dict()
    
    for city in cities:
        city_directory_name = build_directory_city_name(city)
        types[city] = dict()
        item_mapping[city] = dict()
        types_prop[city] = dict()
        df_test_types[city] = dict()
        df_train_types[city] = dict()
        
        gce_results[city] = dict()
        pm[city] = dict()
        pf[city] = dict()
        
        for year in years:
            types[city][year] = pd.read_excel(f'../../../Code/GeneratedExcels/{city_directory_name}_{year}.xlsx', sheet_name='cl_gpt', decimal=',')[['id', 'type', 'ranking']]
            types[city][year] = types[city][year].rename(columns={'id':'itemId'})
            types[city][year]['type'] = types[city][year]['type'].str.strip()
            
            gce_results[city][year] = dict()
            pm[city][year] = dict()
            pf[city][year] = dict()
            
            item_mapping[city][year], types_prop[city][year], df_test_types[city][year], df_train_types[city][year] =\
                create_dataframes(city, year, types[city][year])
                
            for strateg in strategies:
                print(f"City: {city}, Year: {year}, Strategy: {strateg}")
                gce_results[city][year][strateg], pm[city][year][strateg], pf[city][year][strateg] =\
                    fairness_results(city, 
                                     year,
                                     item_mapping[city][year],
                                     types_prop[city][year], 
                                     df_test_types[city][year],
                                     TYPES_ORDER,
                                     MODEL_ORDER,
                                     strategy=strateg
                                    )
                fin_metrics_dir = f"../../data/final_metrics/{city_directory_name}/{year}/{strateg}"
                gce_results[city][year][strateg]['nDCG'] = pd.read_csv(f'{fin_metrics_dir}/final_metrics.csv', index_col=0, sep='|')[['nDCG']]
                gce_results[city][year][strateg] = gce_results[city][year][strateg][['nDCG', 'p_uniform', 'p_test', 'p_minority', 'p_nimby', 'p_min_nimby']]
                fairness_directory = f"../../data/fairness/{city_directory_name}/{year}"
                
                os.makedirs(f"{fairness_directory}/{strateg}", exist_ok=True)
                
                gce_results[city][year][strateg].to_csv(f"{fairness_directory}/{strateg}/final_fairness.csv", sep="|")
                gce_results_styled = gce_results[city][year][strateg].style.background_gradient().format('{:.5f}')
                gce_results_styled.to_html(f"{fairness_directory}/final_fairness_{strateg}.html")
                
                pm[city][year][strateg].to_csv(f"{fairness_directory}/{strateg}/pm.csv", sep="|")
                pm_results_styled = pm[city][year][strateg].style.background_gradient().format('{:.5f}')
                pm_results_styled.to_html(f"{fairness_directory}/pm_{strateg}.html")
                
                
                    
                    
                      

if __name__ == "__main__":
    main()