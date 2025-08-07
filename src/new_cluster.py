import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize
from datetime import datetime
import logging

def setup_logging(log_dir="gmm_logs"):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('gmm_search')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    log_file = os.path.join(log_dir, f'gmm_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def save_results(results, filepath):
    """Save results to JSON file"""
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        return obj
    
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, pd.Series):
            serializable_results[key] = {k: convert_numpy(v) for k, v in value.to_dict().items()}
        else:
            serializable_results[key] = convert_numpy(value)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)

def load_results(filepath):
    """Load results from JSON file"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}

def gmm_parameter_search(dfz, coldict, samples=None, stratify_by=None, 
                     op_path='', n=50,
                     resume=True, model_to_test=None):
    """
    Perform Gaussian Mixture Model parameter search with logging and resume capability
    
    Parameters:
    -----------
    dfz : pandas.DataFrame
        Input dataframe
    coldict : dict
        Dictionary mapping names to column lists
    samples : int or None
        Number of samples to draw. If None, uses entire dataset (default: None)
    stratify_by : str or None
        Column to stratify sampling by. Ignored if samples is None (default: None)
    op_path : str
        Base output path for results and logs (default: '')
    n : int
        Maximum number of components to test (default: 50)
    resume : bool
        Whether to resume from existing results (default: True)
    model_to_test : str or None
        Name of specific model from coldict to test. If None, tests all models (default: None)
    
    Returns:
    --------
    dict : Results dictionary with best parameters and all BIC evaluations for the specified model 
           (or all models if model_to_test is None)
    """
    os.makedirs(op_path, exist_ok=True) 
    
    # Set up file paths based on whether we're testing a specific model
    if model_to_test:
        model_dir = f'{op_path}/{model_to_test}'
        os.makedirs(model_dir, exist_ok=True)
        if samples:
            results_file = f'{model_dir}/{n}_gmm_results_{samples}.json'
            all_bic_file = f'{model_dir}/{n}_gmm_all_bic_{samples}.json'
            log_dir = f'{model_dir}/{n}_gmm_logs_{samples}'
        else:
            results_file = f'{model_dir}/{n}_gmm_results_allsamples.json'
            all_bic_file = f'{model_dir}/{n}_gmm_all_bic_allsamples.json'
            log_dir = f'{model_dir}/{n}_gmm_logs_allsamples'
    else:
        if samples:
            results_file = f'{op_path}/{n}_gmm_results_{samples}.json'
            all_bic_file = f'{op_path}/{n}_gmm_all_bic_{samples}.json'
            log_dir = f'{op_path}/{n}_gmm_logs_{samples}'
        else:
            results_file = f'{op_path}/{n}_gmm_results_allsamples.json'
            all_bic_file = f'{op_path}/{n}_gmm_all_bic_allsamples.json'
            log_dir = f'{op_path}/{n}_gmm_logs_allsamples'
    
    print(results_file)
    print(f"All BIC results will be saved to: {all_bic_file}")
    
    # Setup logging
    logger = setup_logging(log_dir)
    logger.info("Starting GMM parameter search")
    
    # Filter models to test
    if model_to_test is not None:
        # Validate that the specified model exists in coldict
        if model_to_test not in coldict:
            logger.error(f"Model not found in coldict: {model_to_test}")
            raise ValueError(f"Model not found in coldict: {model_to_test}")
        
        # Filter coldict to only include the specified model
        filtered_coldict = {model_to_test: coldict[model_to_test]}
        logger.info(f"Testing specific model: {model_to_test}")
        logger.info(f"Available models in coldict: {list(coldict.keys())}")
    else:
        filtered_coldict = coldict
        logger.info(f"Testing all models: {list(coldict.keys())}")
    
    # Load existing results if resuming
    results = {}
    all_bic_results = {}
    
    if resume and os.path.exists(results_file):
        results = load_results(results_file)
        logger.info(f"Loaded existing results for {len(results)} column sets")
        logger.info(f"Already completed: {list(results.keys())}")
        
        # If testing a specific model, show if it's already completed
        if model_to_test is not None and model_to_test in results:
            logger.info(f"Model {model_to_test} already completed")
    
    if resume and os.path.exists(all_bic_file):
        all_bic_results = load_results(all_bic_file)
        logger.info(f"Loaded existing BIC results for {len(all_bic_results)} column sets")
    
    # Prepare data (sampling or full dataset)
    if samples is not None and stratify_by is not None:
        logger.info(f"Preparing stratified sample: {samples} samples by {stratify_by}")
        try:
            data_to_use = dfz.groupby(stratify_by).sample(n=samples, random_state=42)
            logger.info(f"Created stratified sample with {len(data_to_use)} rows")
        except Exception as e:
            logger.error(f"Error creating stratified sample: {e}")
            raise
    else:
        logger.info("Using entire dataset (no sampling)")
        data_to_use = dfz.copy()
        logger.info(f"Using full dataset with {len(data_to_use)} rows")
    
    # Parameter grid
    param_grid = {
        "n_components": np.arange(1, n),
        "covariance_type": ["full"],
    }
    total_combinations = len(list(ParameterGrid(param_grid)))
    logger.info(f"Parameter grid has {total_combinations} combinations")
    
    # Process each column set
    total_sets = len(filtered_coldict)
    completed_sets = len([name for name in results.keys() if name in filtered_coldict])
    
    for i, (name, cols) in enumerate(filtered_coldict.items(), 1):
        # Skip if already completed and resuming
        if resume and name in results:
            logger.info(f"Skipping {name} (already completed) - {i}/{total_sets}")
            continue
            
        logger.info(f"Processing {name} ({i}/{total_sets}) - Columns: {cols}")
        
        try:
            # Prepare data
            X = data_to_use[cols].values
            logger.info(f"Data shape for {name}: {X.shape}")
            
            # Check for missing values
            if np.isnan(X).any():
                logger.warning(f"Found missing values in {name}, filling with column means")
                X = pd.DataFrame(X).fillna(pd.DataFrame(X).mean()).values
            
            # Scale and normalize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_normalized = normalize(X_scaled)
            logger.info(f"Data preprocessed for {name}")
            
            # Evaluate all parameter combinations
            bic_evaluations = []
            logger.info(f"Starting BIC evaluation for {name} ({total_combinations} combinations)")
            
            for j, params in enumerate(ParameterGrid(param_grid), 1):
                try:
                    gmm = GaussianMixture(**params, random_state=42)
                    gmm.fit(X_normalized)
                    bic_value = gmm.bic(X_normalized)
                    bic_evaluations.append({**params, "BIC": bic_value})
                    
                    if j % 50 == 0:  # Log progress every 50 combinations
                        logger.info(f"Completed {j}/{total_combinations} combinations for {name}")
                        
                except Exception as e:
                    logger.warning(f"Failed to fit GMM for {name} with params {params}: {e}")
                    continue
            
            if not bic_evaluations:
                logger.error(f"No successful fits for {name}")
                continue
                
            # Find best parameters
            bic_df = pd.DataFrame(bic_evaluations).sort_values("BIC", ascending=True)
            best_params = bic_df.iloc[0]
            
            # Store results (best parameters only)
            results[name] = best_params
            
            # Store all BIC evaluations for this model
            all_bic_results[name] = bic_evaluations
            
            logger.info(f"Best parameters for {name}: n_components={best_params['n_components']}, "
                       f"covariance_type={best_params['covariance_type']}, BIC={best_params['BIC']:.2f}")
            logger.info(f"Saved {len(bic_evaluations)} BIC evaluations for {name}")
            
            # Save results after each completion
            save_results(results, results_file)
            save_results(all_bic_results, all_bic_file)
            completed_sets += 1
            logger.info(f"Saved results for {name} ({completed_sets}/{total_sets} completed)")
            
        except Exception as e:
            logger.error(f"Error processing {name}: {e}")
            continue
    
    logger.info(f"GMM parameter search completed. Results saved to {results_file}")
    logger.info(f"All BIC evaluations saved to {all_bic_file}")
    logger.info(f"Successfully processed {len([name for name in results.keys() if name in filtered_coldict])}/{total_sets} requested column sets")
    
    # Return results for the specified model or all models
    if model_to_test is not None:
        return {
            'best_params': {model_to_test: results[model_to_test]} if model_to_test in results else {},
            'all_bic': {model_to_test: all_bic_results[model_to_test]} if model_to_test in all_bic_results else {}
        }
    
    return {
        'best_params': results,
        'all_bic': all_bic_results
    }
