import pandas as pd  
import os 
import glob 
import matplotlib.pyplot as plt 
import sys 
from src.col_set import clustering_variables_dict
import numpy as np 
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize

from src.pre_process import create_meta_variables
from src.cluster import optimize_preprocessing

from src.mixmodel import MixCombi, MixCombiTestFramework, run_mixcombi_on_data
 

# --- Data Loading and Preprocessing ---
# This part of the code is kept as is to prepare the data for the script.
# It assumes the required files are present at the specified paths.

def fill_with_zeros(df, columns_to_fill):
    """Fill missing values (NaNs) with zeros in specified columns."""
    df_filled = df.copy()
    for col in columns_to_fill:
        if col in df_filled.columns:
            df_filled[col] = df_filled[col].fillna(0)
        else:
            print(f"Warning: Column '{col}' not found in DataFrame")
    return df_filled




def load_data():
    
    try:
        neb = pd.read_csv('/rds/user/gb669/hpc-work/energy_map/data/automl_models/input_data/new_final/NEBULA_englandwales_domestic_filtered.csv')
        preds = pd.read_csv('/rds/user/gb669/hpc-work/energy_map/data/automl_models/input_data/new_final/preds_gas_78_ag.csv')
        hh_df = pd.read_csv('/rds/user/gb669/hpc-work/energy_map/data/automl_models/input_data/new_final/pcd_p002 (1).csv')
    except FileNotFoundError as e:
        print(f"Error: A required data file was not found. Please check the file paths. Error: {e}")
        sys.exit(1)
    
    neb_res = neb.merge(preds, on=['postcode', 'total_gas'])
    neb_res['residuals'] = neb_res['total_gas'] - neb_res['preds']
    neb_res['percentage_residuals'] = (neb_res['residuals'] / neb_res['total_gas']) * 100
    neb_res['residuals_per_area'] = neb_res['residuals'] / neb_res['clean_res_total_fl_area_H_total']
    neb_res['PCI'] = neb_res['total_gas'] / neb_res['preds']
    
    hh_df['Postcode'] = hh_df['Postcode'].str.strip()
    hh_df.rename(columns={'Count': 'count_of_households'}, inplace=True)
    
    merg = neb_res.merge(hh_df, left_on='postcode', right_on='Postcode', how='inner')
    merg['gas_per_hh'] = merg['total_gas'] / merg['count_of_households']
    merg['total_energy_per_hh'] = (merg['total_gas'] + merg['total_elec']) / merg['count_of_households']


    pd_df = optimize_preprocessing(merg)
    
    variable_groups = {
        'energy_cols1': ['total_energy_per_hh', 'PCI', 'total_gas', 'total_elec'],
        'energy_cols2': ['gas_EUI_H', 'elec_EUI_H', 'PCI', 'total_gas', 'total_elec'],
        'energy_cols3': ['total_energy_per_hh', 'total_gas', 'total_elec'],
        'energy_cols4': ['gas_EUI_H', 'elec_EUI_H', 'total_gas', 'total_elec'],
        'typ_options': ['high_density_pct', 'low_density_pct', 'all_flats_pct', 'perc_outbuildings'],
        'typ2': ['standard_sized_pct', 'large_properties_pct', 'terraces_mixed_pct', 'all_flats_pct', 'perc_outbuildings'],
        'age1': ['Post 1999_pct', 'Pre 1919_pct'],
        'age2': ['1919-1944_pct', '1945-1959_pct', '1960-1979_pct', '1980-1989_pct', '1990-1999_pct'],
        'age3': ['Post 1999_pct', 'Pre 1919_pct', '1919-1944_pct', '1945-1959_pct', '1960-1979_pct', '1980-1989_pct', '1990-1999_pct'],
        'pc_des': ['postcode_area', 'postcode_density'],
        'socio_ethnicity': ['perc_white', 'perc_asian'],
        'socio_employment': ['Perc_econ_employed', 'perc_econ_unemployed', 'perc_econ_inactive'],
        'socio_household': ['perc_hh_size_small', 'perc_hh_size_medium'],
        'socio_all': ['perc_white', 'perc_asian', 'Perc_econ_employed', 'perc_econ_unemployed', 'perc_econ_inactive', 'perc_hh_size_small', 'perc_hh_size_medium']
    }
    
    df_with_meta = create_meta_variables(pd_df)
    df = df_with_meta.copy()
    
    fz_cols = variable_groups['age1'] + variable_groups['age2'] + variable_groups['age3'] + ['Domestic outbuilding_pct']
    dfz = fill_with_zeros(df, fz_cols)
    dfz['total_energy'] = dfz['total_gas']+dfz['total_elec']
    return dfz 
    

def load_data_for_gm(data, cols):
    data =data[cols].copy() 
    X =  data.values
    print('starting standaardising')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_normalized = normalize(X_scaled)
    return X_normalized 

def proc_data(dfz, model_name = 'top_vars3_eT_pci3', n_samples= 1000, stratify=True):
    if stratify:
        samples = dfz.groupby('region').apply(lambda x: x.sample(n=n_samples, random_state=42)).reset_index(drop=True)
    else:
        samples = dfz.copy() 

    X_normalized = load_data_for_gm(samples, clustering_variables_dict[model_name])
    
    # take samples stratify by region 
    print(X_normalized.shape) 
    return X_normalized


if __name__ == "__main__":
        
    np.random.seed(42)
    stratify=False 
    model_name = os.getenv('MODEL_NAME') 
    op= '/home/gb669/rds/hpc-work/energy_map/uk_postcode_clustering/test_results/mixmodel'
    dfz = load_data()
    if stratify:
        n_samples = int(os.getenv('N_SAMPLES'))
        my_custom_data =  proc_data(dfz, model_name = model_name, n_samples= n_samples, stratify=True)
        op_path = f'{op}/{n_samples}'
    else:
        my_custom_data =  proc_data(dfz, model_name = model_name, n_samples= None, stratify=False)
        op_path = f'{op}/all_samples'
    
    my_true_labels=None
    os.makedirs(op_path, exist_ok=True) 
    run_mixcombi_on_data(my_custom_data, output_path=op_path, true_labels=my_true_labels, k_min=1, k_max=60)
    
    