import pandas as pd
import re

from dusa_function_lib import (
    build_directory_city_name,
)

def main():
    cities = ['Cambridge', 'Miami', 'New York']
    years = ['2014', '2015', '2016', '2017']
    algorithms = ['rand','pop','pop_nc','ib','ub','mf','cb_cat','cb_loc','cbub_cat','cbub_loc','cbib_cat','cbib_loc']
    strategies = ['add_global', 'add', 'add_in_zeros', 'drop', 'drop_add']

    final_metrics_styled_dict = {}
    for city in cities:
        final_metrics_styled_dict[city] = {}
        city_directory_name = build_directory_city_name(city)
        for year in years:
            city_year_directory_name = f"{city_directory_name}/{year}"
            final_metrics_styled_dict[city][year] = {}
            for strateg in strategies:
                city_list_df = []
                city_year_strategy_directory_name = f"{city_year_directory_name}/{strateg}"
                for alg in algorithms:
                    dataframe = pd.read_csv(f"../../data/final_metrics/{city_year_strategy_directory_name}/{alg}_final_metrics.csv", sep="|")
                    ndcg_col = next(col for col in dataframe.columns if re.match(r'ndcg@\d+', col))
                    dataframe.rename(columns={'precision':'Precision',
                                      'recall':'Recall',
                                      'map':'MAP',
                                      ndcg_col:'nDCG',
                                      'mrr':'MRR',
                                      'f1':'F1'}, 
                                     inplace=True)
                    dataframe.drop(columns=['auc'], inplace=True)
                    city_list_df.append(dataframe)
                
                final_metrics_df = pd.concat(city_list_df)
                final_metrics_df.rename(columns={'Unnamed: 0':'Algorithm'}, inplace=True)
                final_metrics_df = pd.concat(city_list_df)
                final_metrics_df.rename(columns={'Unnamed: 0':'Algorithm'}, inplace=True)
                final_metrics_df.set_index('Algorithm', inplace=True)
                final_metrics_df.to_csv(f"../../data/final_metrics/{city_year_strategy_directory_name}/final_metrics.csv", sep="|")
                final_metrics_styled_dict[city][year][strateg] = final_metrics_df.style.background_gradient().format('{:.3f}')
                final_metrics_styled_dict[city][year][strateg].to_html(f"../../data/final_metrics/{city_year_directory_name}/final_metrics_{strateg}.html")
                print(f"City: {city}, Year: {year}, Strategy: {strateg} - Final Metrics Done")
                    
    
if __name__ == '__main__':
    main()