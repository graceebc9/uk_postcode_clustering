import pandas as pd 
import logging
from .column_settings import settings_dict

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import os 


cols = settings_dict[71][1]
cols += ['residuals', 'PCI', 'gas_per_hh']

def load_and_prepare_postcode_data( data,  cols, subset=None):
    logging.info('Load postcode data and aggregate variables')
    # data = pd.read_csv(input_path)
    data = pre_process_pc(data)
    X = data[cols].copy() 
 
    print('X shape: ', X.shape)
    X = X.dropna()
    print('X shape: ', X.shape) 
    data_cols = X.columns.tolist()
    if subset is None:
        return  X, data_cols
    else:
        return  X.iloc[0:subset], data_cols
    



def econ_settings():
   econ_act = ['economic_activity_perc_Economically active (excluding full-time students): In employment: Employee: Part-time',
 'economic_activity_perc_Economically active (excluding full-time students): In employment: Employee: Full-time',
 'economic_activity_perc_Economically active (excluding full-time students): In employment: Self-employed with employees: Part-time',
 'economic_activity_perc_Economically active (excluding full-time students): In employment: Self-employed with employees: Full-time',
 'economic_activity_perc_Economically active (excluding full-time students): In employment: Self-employed without employees: Part-time',
 'economic_activity_perc_Economically active (excluding full-time students): In employment: Self-employed without employees: Full-time',
 
 'economic_activity_perc_Economically active and a full-time student: In employment: Employee: Part-time',
 'economic_activity_perc_Economically active and a full-time student: In employment: Employee: Full-time',
 'economic_activity_perc_Economically active and a full-time student: In employment: Self-employed with employees: Part-time',
 'economic_activity_perc_Economically active and a full-time student: In employment: Self-employed with employees: Full-time',
 'economic_activity_perc_Economically active and a full-time student: In employment: Self-employed without employees: Part-time',
 'economic_activity_perc_Economically active and a full-time student: In employment: Self-employed without employees: Full-time',
   ]
   econ_inac = ['economic_activity_perc_Economically inactive: Retired',
 'economic_activity_perc_Economically inactive: Student',
 'economic_activity_perc_Economically inactive: Looking after home or family',
 'economic_activity_perc_Economically inactive: Long-term sick or disabled',
 'economic_activity_perc_Economically inactive: Other' ] 
   
   unemp = ['economic_activity_perc_Economically active and a full-time student: Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks',
   'economic_activity_perc_Economically active (excluding full-time students): Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks' 
   ]
   other = ['economic_activity_perc_Does not apply']

   list_cols = [econ_act, unemp, econ_inac, other] 
   names = ['Perc_econ_employed', 'perc_econ_unemployed', 'perc_econ_inactive', 'perc_econ_other' ] 
   return list_cols, names 


def eth_setting():
    white = [ 'ethnic_group_perc_White: English, Welsh, Scottish, Northern Irish or British', 
    'ethnic_group_perc_White: Irish', 
    'ethnic_group_perc_White: Gypsy or Irish Traveller',
    'ethnic_group_perc_White: Roma', 'ethnic_group_perc_White: Other White',]
    black = [ 'ethnic_group_perc_Black, Black British, Black Welsh, Caribbean or African: African',
 'ethnic_group_perc_Black, Black British, Black Welsh, Caribbean or African: Caribbean',
 'ethnic_group_perc_Black, Black British, Black Welsh, Caribbean or African: Other Black',] 
    asian = [ 'ethnic_group_perc_Asian, Asian British or Asian Welsh: Bangladeshi',
 'ethnic_group_perc_Asian, Asian British or Asian Welsh: Chinese',
 'ethnic_group_perc_Asian, Asian British or Asian Welsh: Indian',
 'ethnic_group_perc_Asian, Asian British or Asian Welsh: Pakistani',
 'ethnic_group_perc_Asian, Asian British or Asian Welsh: Other Asian',] 
    other = ['ethnic_group_perc_Does not apply',
 'ethnic_group_perc_Other ethnic group: Arab',
 'ethnic_group_perc_Other ethnic group: Any other ethnic group',] 
    mixed = [ 'ethnic_group_perc_Mixed or Multiple ethnic groups: White and Asian',
 'ethnic_group_perc_Mixed or Multiple ethnic groups: White and Black African',
 'ethnic_group_perc_Mixed or Multiple ethnic groups: White and Black Caribbean',
 'ethnic_group_perc_Mixed or Multiple ethnic groups: Other Mixed or Multiple ethnic groups',]
    list_cols = [white, black, asian, other, mixed]
    names = ['perc_white', 'perc_black', 'perc_asian', 'perc_asian', 'perc_ethnic_other', 'perc_mixed']
    return list_cols , names  

def hh_size_setting():
    small = ['household_siz_perc_perc_0 people in household','household_siz_perc_perc_1 person in household']
    medium = ['household_siz_perc_perc_2 people in household', 'household_siz_perc_perc_3 people in household',
     'household_siz_perc_perc_4 people in household'] 
    large = ['household_siz_perc_perc_5 people in household',
    'household_siz_perc_perc_6 people in household',
    'household_siz_perc_perc_7 people in household']
    list_cols = [small, medium, large]
    names = ['perc_hh_size_small', 'perc_hh_size_medium', 'perc_hh_size_large']
    return list_cols, names

def type_setting1():
    large = ['Very large detached', 'Large detached','Large semi detached' ,  'Tall terraces 3-4 storeys']
    standard = ['Standard size semi detached',  'Standard size detached']
    large_flats = ['Very tall point block flats', 'Tall flats 6-15 storeys']
    med_flats = ['Medium height flats 5-6 storeys', '3-4 storey and smaller flats']
    small_terraces = ['Small low terraces', '2 storeys terraces with t rear extension', 'Semi type house in multiples']
    flats = large_flats + med_flats
    estates = ['Linked and step linked premises', 'Planned balanced mixed estates']
 
    outbuilds = ['Domestic outbuilding']
    list_cols = [large, standard,  small_terraces, estates,  flats, outbuilds ]
    names = ['perc_large_houses', 'perc_standard_houses',  'perc_small_terraces', 'perc_estates', 'perc_all_flats', 'perc_outbuildings']
    return list_cols, names

def age_setting1():
    pre_1919 = ['Pre 1919']
    o1919_1999= ['1919-1944', '1945-1959', '1960-1979', '1980-1989', '1990-1999',]
    post_1999= ['Post 1999'] 


    age_cols = [pre_1919, o1919_1999, post_1999]
    age_names = ['perc_age_Pre-1919', 'perc_age_1919-1999', 'perc_age_Post-1999',  ]
    return  age_cols, age_names 
 

import pandas as pd
import numpy as np

def optimize_preprocessing(data):
    """
    Convert attributes into meta attributes by combinign certain subgroups 
    """
    # Pre-calculate all column mappings
    column_mappings = {
        'type': (type_setting1(), lambda x: x + '_pct'),
        'age': (age_setting1(), lambda x: x + '_pct'),
        'eth': (eth_setting(), lambda x: x),
        'econ': (econ_settings(), lambda x: x),
        'hh_size': (hh_size_setting(), lambda x: x)
    }
    
    # Vectorized operations instead of loops
    def process_columns(columns, names, transform_fn):
        # Create all column lists at once
        all_cols = [[transform_fn(x) for x in col_group] for col_group in columns]
        
        # Perform vectorized operations
        results = pd.DataFrame({
            name: data[cols].fillna(0).sum(axis=1) 
            for name, cols in zip(names, all_cols)
        })
        
        return results
    
    # Process all categories at once
    results = []
    for (columns, names), transform_fn in column_mappings.values():
        results.append(process_columns(columns, names, transform_fn))
    
    # Combine results efficiently
    return pd.concat([data] + results, axis=1)

# Modified function calls
def pre_process_pc(data):
    data =  optimize_preprocessing(data)
    return data 



# 6 is 5 without residuals 

clustering_col_dict = {1:   [
    'total_gas',
    'residuals',
'total_elec',
      'PCI',
            'gas_per_hh',
    'perc_large_houses',
    'perc_standard_houses',
    'perc_small_terraces',
    'perc_all_flats',
    'perc_age_Pre-1919',
    'perc_age_Post-1999',
    'perc_white',
    'perc_black',
    'Perc_econ_employed',
      'perc_econ_inactive',
      'perc_hh_size_small',
      'perc_hh_size_medium',
    ], 
    2:    [
    'total_gas',
    'residuals',
'total_elec',
            'gas_per_hh',
    'perc_large_houses',
    'perc_standard_houses',
    'perc_small_terraces',
    'perc_all_flats',
    'perc_age_Pre-1919',
    'perc_age_Post-1999',
    'perc_white',
    'perc_black',
    'Perc_econ_employed',
      'perc_econ_inactive',
      'perc_hh_size_small',
      'perc_hh_size_medium',
    ], 

    3:   [
      'PCI',
            'gas_per_hh',
    'perc_large_houses',
    'perc_standard_houses',
    'perc_small_terraces',
    'perc_all_flats',
    'perc_age_Pre-1919',
    'perc_age_Post-1999',
    'perc_white',
    'perc_black',
    'Perc_econ_employed',
      'perc_econ_inactive',
      'perc_hh_size_small',
      'perc_hh_size_medium',
    ], 
    4:  [  'PCI',
            'HDD',
            'gas_per_hh',
    'perc_large_houses',
    'perc_standard_houses',
    'perc_small_terraces',
    'perc_all_flats',
    'perc_age_Pre-1919',
    'perc_age_Post-1999',
    'perc_white',
    'Perc_econ_employed',
      'perc_econ_inactive',
      'perc_hh_size_medium',
    'perc_hh_size_small']
    ,
    5:   [
    'total_gas',
    'HDD',
    'residuals',
'total_elec',
      'PCI',
            'gas_per_hh',
    'perc_standard_houses',
    'perc_small_terraces',
    'perc_all_flats',
    'perc_age_Pre-1919',
    'perc_age_Post-1999',
    'perc_white',
    'perc_black',
    'Perc_econ_employed',
      'perc_econ_inactive',
      'perc_hh_size_small',
      'perc_hh_size_medium',
    ], 
        6:   [
    'total_gas',
    'HDD',
'total_elec',
      'PCI',
            'gas_per_hh',
    'perc_standard_houses',
    'perc_small_terraces',
    'perc_all_flats',
    'perc_age_Pre-1919',
    'perc_age_Post-1999',
    'perc_white',
    'perc_black',
    'Perc_econ_employed',
      'perc_econ_inactive',
      'perc_hh_size_small',
      'perc_hh_size_medium',
    ], 

        7:   [
    'total_gas',
    'HDD',
'total_elec',
      'PCI',
            'gas_per_hh',
    'perc_standard_houses',
    'perc_small_terraces',
    # 'perc_all_flats',
    'perc_age_Pre-1919',
    'perc_age_Post-1999',
    'perc_white',
    # 'perc_black',
    'Perc_econ_employed',
      'perc_econ_inactive',
      'perc_hh_size_small',
      'perc_hh_size_medium',
    ], 

        8:   [
    'total_gas',
    'HDD',
# 'total_elec',
      'PCI',
            # 'gas_per_hh',
    'perc_standard_houses',
    'perc_small_terraces',
    'perc_all_flats',
    'perc_age_Pre-1919',
    'perc_age_Post-1999',
    'perc_white',
    # 'perc_black',
    'Perc_econ_employed',
      'perc_econ_inactive',
    #   'perc_hh_size_small',
      'perc_hh_size_medium',
    ], 
        9:   [
    'total_gas',
    # 'HDD',
'total_elec',
      'PCI',
            'gas_per_hh',
    'perc_standard_houses',
    'perc_small_terraces',
    # 'perc_all_flats',
    'perc_age_Pre-1919',
    'perc_age_Post-1999',
    'perc_white',
    # 'perc_black',
    'Perc_econ_employed',
      'perc_econ_inactive',
      'perc_hh_size_small',
      'perc_hh_size_medium',
    ], 
            10:   [
    # 'total_gas',
    # 'HDD',
# 'total_elec',
      'PCI',
            'gas_per_hh',
    'perc_standard_houses',
    'perc_small_terraces',
    # 'perc_all_flats',
    # 'perc_age_Pre-1919',
    # 'perc_age_Post-1999',
    'perc_white',
    # 'perc_black',
    'Perc_econ_employed',
      'perc_econ_inactive',
      'perc_hh_size_small',
      'perc_hh_size_medium',
    ], 
        11:    [
    'total_gas',
    'HDD',
'total_elec',
      'PCI',
            'gas_per_hh',
    # 'perc_standard_houses',
    # 'perc_small_terraces',
    # 'perc_all_flats',
    'perc_age_Pre-1919',
    'perc_age_Post-1999',
    # 'perc_white',
    # 'perc_black',
    'Perc_econ_employed',
      'perc_econ_inactive',
      'perc_hh_size_small',
      'perc_hh_size_medium',
    ], 
    12:    [
    'total_gas',
    'HDD',
# 'total_elec',
      'PCI',
            'gas_per_hh',
    # 'perc_standard_houses',
    # 'perc_small_terraces',
    # 'perc_all_flats',
    'perc_age_Pre-1919',
    'perc_age_Post-1999',
    # 'perc_white',
    # 'perc_black',
    'Perc_econ_employed',
      'perc_econ_inactive',
      'perc_hh_size_small',
      'perc_hh_size_medium',
    ], 
    13:    [
    'total_gas',
    'HDD',
'total_elec',
      'PCI',
            'gas_per_hh',
    'perc_standard_houses',
    'perc_small_terraces',
    'perc_all_flats',
    # 'perc_age_Pre-1919',
    # 'perc_age_Post-1999',
    # 'perc_white',
    # 'perc_black',
    'Perc_econ_employed',
      'perc_econ_inactive',
      'perc_hh_size_small',
      'perc_hh_size_medium',
    ], 

    14:    [
    'total_gas',
    # 'HDD',
# 'total_elec',
      'PCI',
            'gas_per_hh',
    'perc_standard_houses',
    'perc_small_terraces',
    'perc_all_flats',
    # 'perc_age_Pre-1919',
    # 'perc_age_Post-1999',
    # 'perc_white',
    # 'perc_black',
    'Perc_econ_employed',
      'perc_econ_inactive',
      'perc_hh_size_small',
      'perc_hh_size_medium',
    ], 

     15:    [
    'total_gas',
    # 'HDD',
'total_elec',
      'PCI',
            'gas_per_hh',
    # 'perc_standard_houses',
    # 'perc_small_terraces',
    # 'perc_all_flats',
    'perc_age_Pre-1919',
    'perc_age_Post-1999',
    # 'perc_white',
    # 'perc_black',
    'Perc_econ_employed',
      'perc_econ_inactive',
      'perc_hh_size_small',
      'perc_hh_size_medium',
    ], 
        16:    [
    # 'total_gas',
    'HDD',
# 'total_elec',
      'PCI',
            'gas_per_hh',
    'perc_standard_houses',
    'perc_small_terraces',
    'perc_all_flats',
    # 'perc_age_Pre-1919',
    # 'perc_age_Post-1999',
    # 'perc_white',
    # 'perc_black',
    'Perc_econ_employed',
      'perc_econ_inactive',
      'perc_hh_size_small',
      'perc_hh_size_medium',
    ], 
            17:    [
    'total_gas',
    'HDD',
'total_elec',
      'residuals',
            'gas_per_hh',
    'perc_standard_houses',
    'perc_small_terraces',
    'perc_all_flats',
    # 'perc_age_Pre-1919',
    # 'perc_age_Post-1999',
    # 'perc_white',
    # 'perc_black',
    'Perc_econ_employed',
      'perc_econ_inactive',
      'perc_hh_size_small',
      'perc_hh_size_medium',
    ], 

        18:    [
    'total_gas',
    'HDD',
'total_elec',
      'PCI',
            'gas_per_hh',
    'perc_standard_houses',
    'perc_small_terraces',
    'perc_large_houses',
    # 'perc_age_Pre-1919',
    # 'perc_age_Post-1999',
    # 'perc_white',
    # 'perc_black',
    'Perc_econ_employed',
      'perc_econ_inactive',
      'perc_hh_size_small',
      'perc_hh_size_medium',
    ], 
     19:    [
    'total_gas',
    'HDD',
'total_elec',
      'PCI',
            'gas_per_hh',
    'perc_standard_houses',
    'perc_small_terraces',
    'perc_all_flats',
    # 'perc_age_Pre-1919',
    # 'perc_age_Post-1999',
    # 'perc_white',
    # 'perc_black',
    'Perc_econ_employed',
      'perc_econ_inactive',
      'perc_hh_size_small',
      'perc_hh_size_medium',
      'urb_binary'
    ], 

        20:    [
'gas_EUI_H',
 'elec_EUI_H',
#  'total_gas',
    'HDD',
# 'total_elec',
      'PCI',
            'gas_per_hh',
    'perc_standard_houses',
    'perc_small_terraces',
    'perc_all_flats',
    # 'perc_age_Pre-1919',
    # 'perc_age_Post-1999',
    # 'perc_white',
    # 'perc_black',
    'Perc_econ_employed',
      'perc_econ_inactive',
      'perc_hh_size_small',
      'perc_hh_size_medium',
    ], 

        21:    [
'gas_EUI_H',
#  'elec_EUI_H',
 'total_gas',
    'HDD',
# 'total_elec',
      'PCI',
            'gas_per_hh',
    'perc_standard_houses',
    'perc_small_terraces',
    'perc_all_flats',
    # 'perc_age_Pre-1919',
    # 'perc_age_Post-1999',
    # 'perc_white',
    # 'perc_black',
    'Perc_econ_employed',
      'perc_econ_inactive',
      'perc_hh_size_small',
      'perc_hh_size_medium',
    ], 

    }


def load_gm(data, subset=None  ):
    X = data.copy() 

    print('X shape: ', X.shape)
    X = X.dropna()
    print('X shape afetr drop na : ', X.shape) 
 
    if subset is None:
      return  X
    else:
      return  X.iloc[0:subset] 
    
def run_cluster_global(data, cols, output_path =None, save=False,  max_clusters = 20 ) :
    
    def gmm_bic_score(estimator, X,  ):
        """Callable to pass to GridSearchCV that will use the BIC score."""
        # Make it negative since GridSearchCV expects a score to maximize
        return -estimator.bic(X)

    param_grid = {
        "n_components": range(2, max_clusters),
        "covariance_type": [ "spherical" , "diagonal", "tied",  "full"],
    }

    grid_search = GridSearchCV(
        GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
    )
    print('loading data')
    data =data[cols].copy() 
    X =  load_gm(data)

    print('starting standaardising')
 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_normalized = normalize(X_scaled)
    
    if pca: 
      pca = PCA(n_components=2)
      X_principal = pca.fit_transform(X_normalized)
      xgmm = X_principal.copy() 
    
    else:
        xgmm = X_normalized.copy() 
    
 
    print('starting grid search')
    grid_search.fit(xgmm)

    df = pd.DataFrame(grid_search.cv_results_)[
        ["param_n_components", "param_covariance_type", "mean_test_score"]
    ]
    df["mean_test_score"] = -df["mean_test_score"]
    df = df.rename(
        columns={
            "param_n_components": "Number of components",
            "param_covariance_type": "Type of covariance",
            "mean_test_score": "BIC score",
        }
    )
    print(df.sort_values(by="BIC score").head())
    if save: 
        
        df.to_csv(os.path.join(output_path, 'nebfull_bic_score.csv'))


    sns.catplot(
        data=df,
        kind="bar",
        x="Number of components",
        y="BIC score",
        hue="Type of covariance",
    )
    # save fig
    if save:  
      plt.savefig(os.path.join(output_path, 'neb_full_bic_score.png'))

    f, ax = plt.subplots(figsize=(10, 6))
    # show all x ticks 
    ax.set_xticks(range(3, 20))
    df[df['Type of covariance']=='full'].plot(x='Number of components', y='BIC score', ax=ax)


def run_single_cluster(df, cols, n , type='full', max_iter=100, verbose=0, pca=False):
    df =df[cols].copy() 
    X =  load_gm(df)
    X=X[cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_normalized = normalize(X_scaled)
    if pca: 
        pca = PCA(n_components=2)
        X_principal = pca.fit_transform(X_normalized)
        xgmm = X_principal.copy() 
    else:
        xgmm = X_normalized.copy() 


    print('tainnig gmm')
    gmm = GaussianMixture(n_components=n, covariance_type=type,
                          random_state=1,max_iter=max_iter, 
                          n_init=1,
                            verbose=verbose
                          )
    gmm.fit(xgmm)
    df['cluster'] = gmm.predict(xgmm)    

    X['cluster'] = df['cluster']


    return df, X, xgmm




def fill_with_zeros(df, columns_to_fill):
    """
    Fill missing values (NaNs) with zeros in specified columns.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing columns to fill
        columns_to_fill (list): List of column names where NaNs should be filled with zeros
                    
    Returns:
        pandas.DataFrame: DataFrame with filled columns
    """
    df_filled = df.copy()
    
    for col in columns_to_fill:
        if col in df_filled.columns:
            nan_count = df_filled[col].isna().sum()
            df_filled[col] = df_filled[col].fillna(0)
        else:
            print(f"Warning: Column '{col}' not found in DataFrame")
    
    return df_filled