import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import os
from dusa_function_lib import (
    build_directory_city_name,
    get_ratings_df,
)
########################################################################
#### Fairness dataframes creation
########################################################################
def create_dataframes(city, year, types, strategy='basic'):
    dir_city = build_directory_city_name(city)
    data_dir = f"../../data/rm/{dir_city}/{year}/{strategy}"
    item_cnv = pd.read_csv(f"{data_dir}/item_mapping.csv", sep="|")
    types_prop = pd.concat([types[types.itemId.isin(item_cnv.itemId)],
                            pd.DataFrame(data={"itemId":
                                        item_cnv[~item_cnv['itemId'].isin(types['itemId'])].itemId,
                                        'type': ['no_type']*item_cnv[~item_cnv['itemId']\
                                                                                .isin(types['itemId'])].shape[0],
                                               'ranking':[0]*item_cnv[~item_cnv['itemId']\
                                                                       .isin(types['itemId'])].shape[0]})
                            ]).sort_values('itemId').reset_index(drop=True)
    
    # delete duplicates
    types_prop = \
        types_prop.loc[types_prop.groupby('itemId').ranking.idxmin().reset_index().set_index('ranking').index]\
            .reset_index(drop=True)
    
    df_test_types_prop = get_ratings_df(load_npz(f'{data_dir}/num_comm_test.npz'),
                                        rm_info={'user_mapping':
                                                 pd.read_csv(f'{data_dir}/user_mapping.csv', sep='|'),
                                                 'item_mapping': item_cnv})\
    .merge(types_prop, how='left', on='itemId')
    
    df_train_types_prop = get_ratings_df(load_npz(f'{data_dir}/num_comm_train.npz'),
                                         rm_info={'user_mapping':
                                                  pd.read_csv(f'{data_dir}/user_mapping.csv', sep='|'),
                                                  'item_mapping': item_cnv})\
        .merge(types_prop, how='left', on='itemId')
    
    return item_cnv, types_prop, df_test_types_prop, df_train_types_prop


def fairness_results(city, year, item_mapping, types_prop, df_test_types_prop, TYPES_ORDER, MODEL_ORDER, strategy='basic'):
    dir_city = build_directory_city_name(city)
    data_dir = f"../../data/recommendations/{dir_city}/{year}/{strategy}/"
    model_list = os.listdir(f'{data_dir}')
    model_list = [f for f in model_list if f.endswith('.csv')]
    # model_list.remove('model_history.csv')

    gce_results = []
    pm_t = []
    for m in model_list:
        
        m_name = m.split('.')[0].split('rec_')[1]
        df_rec_m = pd.read_csv(f'{data_dir}{m}', sep='|')
        df_rec_m_types_prop = df_rec_m.merge(types_prop, how='left', on='itemId')

        types_prop_m = types_prop[(types_prop.itemId.isin(item_mapping.itemId)) & \
                              (types_prop.itemId.isin(df_rec_m.itemId))].reset_index(drop=True)

        # types_prop_m_values= list(types_prop_m['type'].unique())
        types_prop_m_values = ['minority', 'no_type', 'nimby']
        types_prop_m_values.sort(key=lambda val: TYPES_ORDER[val])
        
        # Uniform
        p0 = dict(zip(types_prop_m_values, [1 / len(types_prop_m_values) for i in range(0, len(types_prop_m_values))]))
        # each = num_likes_test_cat/num_likes_test
        d =  df_test_types_prop.groupby('type').count()['itemId'].to_dict()
        d = [d[k] for k in sorted(TYPES_ORDER, key=TYPES_ORDER.get)]
        p1 = dict(zip(types_prop_m_values, [v/np.sum(d) for v in d]))
        # minority = 0.5, rest = 0.1
        p2 = dict(zip(types_prop_m_values, [0.8, 0.1, 0.1]))
        # nimby = 0.5, rest = 0.1
        p3 = dict(zip(types_prop_m_values, [0.1, 0.8, 0.1]))
        # no_type = 0.1, rest = 0.9/num_rest
        p7 = dict(zip(types_prop_m_values, [0.9 / (len(types_prop_m_values)-1) 
                                          for i in range(0, len(types_prop_m_values)-1)]+[0.1]))
        
        pf = {'p_uniform':p0, 'p_test':p1, 'p_minority':p2, 'p_nimby':p3, 'p_min_nimby':p7}

        gce_df = pd.DataFrame(index=[m_name])
        
        for n, p in zip(pf.keys(), pf.values()):

            gce_df[n], pm = GCE(itemIds=item_mapping.itemId.unique(),
                            df_rec_attributes=df_rec_m_types_prop, 
                            df_test_attributes=df_test_types_prop,
                            item_attributes=types_prop_m,
                            p_f=p,
                            fun='ndcg', beta=2, h=0.95, pc=0.0001)
        pm_t.append(pd.DataFrame(data=pm, index=[f'pm_{m_name}']))
        gce_results.append(gce_df)

    gce_results = pd.concat(gce_results)
    pm_t = pd.concat(pm_t)
    pm_t = pm_t[TYPES_ORDER.keys()]
    return gce_results.loc[[s for s in MODEL_ORDER.keys()]], pm_t.loc[[f'pm_{s}' for s in MODEL_ORDER.keys()]], pf

#########################################################################
#### fairness evaluation functions
#########################################################################
def estimate_model_distribution(all_items, df_rec, df_test, item_attributes, attribute_values,
                                attribute_name, h=0.95, pc=0.0001, fun='count'):
    rgis_df = pd.DataFrame(data={'itemId': all_items})

    # Get rank of item for user
    df_rec_r = df_rec.sort_values(by=['userId', 'itemId', 'scores'], ascending=False)

    df_rec_r = df_rec_r.merge(df_rec_r[['userId']].reset_index().rename(columns={'index': 'r_0'}) \
                              .groupby('userId').min().reset_index()[['userId', 'r_0']],
                              how='left', on=['userId'])

    df_rec_r['r'] = df_rec_r.index - df_rec_r.r_0 + 1
    df_rec_r.drop(columns=['r_0', 'scores'], inplace=True)

    rgis_df = rgis_df.merge(df_rec_r, how='left', on='itemId')
    rgis_df['r'] = rgis_df['r'].fillna(0)

    if fun == 'count':
        rgis_df = rgis_df.groupby('itemId').count()[['userId']].reset_index()
        rgis = dict(zip(rgis_df['itemId'], rgis_df['userId']))
    else:
        rgis_df = rgis_df.merge(df_test, how='left', on=['itemId', 'userId']).rename(
            columns={'numComments': 'rel_u_i'})
        rgis_df.loc[(rgis_df.rel_u_i.isnull()) & (rgis_df.r == 0), 'rel_u_i'] = 0
        rgis_df.loc[~(rgis_df.rel_u_i.isnull()) | (rgis_df.r > 0), 'rel_u_i'] = 1

        if fun == 'bin':
            rgis_df = rgis_df.groupby('itemId').sum()[['rel_u_i']].reset_index()
            rgis = dict(zip(rgis_df['itemId'], rgis_df['rel_u_i']))

        elif fun == 'ndcg':
            rgis_df['ndcg'] = 0.0
            rgis_df.loc[rgis_df.r > 0, 'ndcg'] = (2 ** rgis_df.loc[rgis_df.r > 0, 'rel_u_i'] - 1) / np.log2(
                rgis_df.loc[rgis_df.r > 0, 'r'] + 1)
            rgis_df = rgis_df.groupby('itemId').sum()[['ndcg']].reset_index()
            rgis = dict(zip(rgis_df['itemId'], rgis_df['ndcg']))
        else:
            print("ERROR: fun is not defined.")
            return

    Z = np.sum(list(rgis.values()))

    pm_i = {}
    for a in attribute_values:
        pm_i[a] = 0
        # For each item with that value for the atribute a
        for i in item_attributes[item_attributes[attribute_name] == a].itemId:
            pm_i[a] += rgis[i]
        pm_i[a] = h * pm_i[a] / Z + (1 - h) * pc

    Z_hat = np.sum(list(pm_i.values()))
    for a in attribute_values:
        pm_i[a] = pm_i[a]/Z_hat

    return pm_i


def GCE(itemIds, df_rec_attributes, df_test_attributes, item_attributes, p_f, fun='count',
        beta=2, h=0.95, pc=0.0001):
    GCE = 0

    attribute_name = [i for i in item_attributes.columns if i != 'itemId'][0]
    # attribute_values_uniq = item_attributes[attribute_name].unique()
    attribute_values = ['minority', 'no_type', 'nimby']

    p_m = estimate_model_distribution(all_items=itemIds,
                                      df_rec=df_rec_attributes, df_test=df_test_attributes,
                                      item_attributes=item_attributes,
                                      attribute_values=attribute_values,
                                      attribute_name=attribute_name,
                                      h=h, pc=pc, fun=fun)
    for a in attribute_values:
        if (p_m[a] == 0):
            continue  # If a topic value has never been assigned to the items, it is skipped
        GCE += (p_f[a] ** beta) * (p_m[a] ** (1 - beta))
    GCE -= 1
    GCE *= 1 / (beta * (1 - beta))

    return GCE, p_m