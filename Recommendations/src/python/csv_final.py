import pandas as pd
from dusa_function_lib import (
    build_directory_city_name,
)
def main():
    df = pd.DataFrame(
        columns=[
            'city', 'year', 'recommender', 'mitigation',
            'Precision', 'Recall', 'MAP', 'nDCG', 'MRR', 'F1',
            'p_uniform', 'p_test', 'p_minority', 'p_nimby', 'p_min_nimby',
            'pm_minority', 'pm_nimby', 'pm_other',
            ]
    )
    #cities = ['Cambridge', 'Miami', 'New York']
    cities = ['New York']
    # years = ['2014', '2015', '2016', '2017']
    years = ['2015']
    strategies = ['basic', 'add_global', 'add', 'add_in_zeros', 'drop', 'drop_add']
    
    for city in cities:
        city_dir = build_directory_city_name(city)
        for year in years:
            print(f"Processing {city} {year}")
            city_year_dir = f"{city_dir}/{year}"
            for strat in strategies:
                city_year_str_dir = f"{city_year_dir}/{strat}"
                metrics_csv = f"../../data/final_metrics/{city_year_str_dir}/final_metrics.csv"
                fairness_csv = f"../../data/fairness/{city_year_str_dir}/final_fairness.csv"
                pm_csv = f"../../data/fairness/{city_year_str_dir}/pm.csv"

                mets = pd.read_csv(metrics_csv, sep='|', index_col=0)
                fair = pd.read_csv(fairness_csv, sep='|', index_col=0).drop(columns=['nDCG'])
                pm = pd.read_csv(pm_csv, sep='|', index_col=0)

                pm.index = pm.index.str.replace(r'^pm_', '', regex=True)
                pm.rename(columns={'minority':'pm_minority',
                                    'nimby':'pm_nimby',
                                    'no_type':'pm_other'}, inplace=True)
                combined = mets.join(fair).join(pm)
                idx_name = combined.index.name if combined.index.name else 'index'
                combined = combined.reset_index().rename(columns={idx_name:'recommender'})
                
                for _, row in combined.iterrows():
                    df.loc[len(df)] = [
                        city,
                        int(year),
                        row['recommender'],
                        strat,
                        row['Precision'],
                        row['Recall'],
                        row['MAP'],
                        row['nDCG'],
                        row['MRR'],
                        row['F1'],
                        row['p_uniform'],
                        row['p_test'],
                        row['p_minority'],
                        row['p_nimby'],
                        row['p_min_nimby'],
                        row['pm_minority'],
                        row['pm_nimby'],
                        row['pm_other']
                    ]
    # 
    df.to_csv('../../data/all_results_newyork_2015.csv', index=False, sep='|', decimal=',')
    print("Datos guardados en '../../data/all_results_newyork_2015.csv'")
    

if __name__ == "__main__":
    main()