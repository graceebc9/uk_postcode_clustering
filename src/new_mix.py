import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.stats import multivariate_normal
import seaborn as sns
import time
from typing import Dict, Tuple, List, Optional
 
from .MixCombi import MixCombi
from .MixCombiTestFramework import MixCombiTestFramework   

import os
import json
import pickle

def run_mixcombi_with_predetermined_gmm(
    data: np.ndarray,
    n_components: int,
    covariance_type: str,
    output_path: str,
    true_labels: Optional[np.ndarray] = None,
    save_plots: bool = True,
    save_results: bool = True,
    rs: int = 42,
) -> Tuple[Dict, Dict, Dict]:
    """
    Runs the MixCombi algorithm using a Gaussian Mixture Model (GMM) with a
    predetermined number of clusters and covariance type.

    This function first fits a GMM with the specified parameters, then uses
    its learned components as input for the MixCombi process. It also handles
    saving results and plots similar to the original 'run_mixcombi_on_data'.

    Parameters:
    -----------
    data : np.ndarray
        Input data for clustering.
    n_components : int
        The predetermined number of clusters (components) for the initial GMM fit.
    covariance_type : str
        The covariance type for the GMM (e.g., 'full', 'tied', 'diag', 'spherical').
        Refer to sklearn.mixture.GaussianMixture documentation for valid types.
    output_path : str
        Path to the directory where results and plots will be saved.
        The directory will be created if it does not exist.
    true_labels : Optional[np.ndarray], default=None
        Ground truth labels for the data, if available. Used for evaluation metrics
        like Adjusted Rand Index (ARI). If not provided, Silhouette score is used.
    save_plots : bool, default=True
        Whether to save generated plots (e.g., GMM solution, Entropy Elbow,
        MixCombi solutions) to disk. Plots are saved as PNG files.
    save_results : bool, default=True
        Whether to save numerical results (e.g., evaluation metrics, summary,
        cluster labels) to disk as JSON, pickle, and text files.
    rs: int, default = 42
        randoms state for training GMM
    Returns:
    --------
    tuple: (gmm_fit_params, mixcombi_results, evaluation_results)
        gmm_fit_params (Dict): A dictionary containing parameters and the model
                               of the GMM fitted with the predetermined settings.
                               Includes 'bic_params', 'icl_params', and 'best_bic_model'.
        mixcombi_results (Dict): A dictionary containing the results from the
                                 MixCombi combination process, including labels
                                 and entropies for various K values.
        evaluation_results (Dict): A dictionary containing evaluation metrics
                                   (ARI, Silhouette) for each MixCombi solution (K).
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    framework = MixCombiTestFramework()

    print("=" * 80)
    print("RUNNING MIXCOMBI WITH PREDETERMINED GMM SETTINGS")
    print(f"Predetermined K: {n_components}, Covariance Type: '{covariance_type}'")
    print(f"Output directory: {output_path}")
    print("=" * 80)

    # 1. Fit GMM with predetermined parameters
    print(f"\nFitting GMM with K={n_components}, covariance_type='{covariance_type}'...")
    # Initialize and fit the GaussianMixture model with the specified parameters
    gmm_model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=rs)
    gmm_model.fit(data)
    print("GMM fitting complete.")

    # Prepare parameters for MixCombi based on the fitted GMM
    # bic_params will reflect the chosen GMM settings for MixCombi's initialization
    bic_params = {
        'K': n_components,
        'covariance_type': covariance_type,
        'bic': gmm_model.bic(data), 
         'mu': gmm_model.means_,
          'S': gmm_model.covariances_,
            'p': gmm_model.weights_
    }
  
    # icl_params will contain the actual parameters (weights, means, covariances)
    # of the fitted GMM, which MixCombi uses to define initial components.
    icl_params = {
        'K': n_components,
 
        'precisions_cholesky_': gmm_model.precisions_cholesky_,
        'log_likelihood': gmm_model.score(data) * data.shape[0] # score returns log-likelihood per sample
    }

    # Consolidate GMM fit parameters for return and internal use
    gmm_fit_params = {
        'bic_params': bic_params,
        'icl_params': icl_params,
        'best_bic_model': gmm_model,
        'best_icl_model': gmm_model # In this specific case, BIC and ICL models are the same
    }

    # Plot the initial GMM solution if data is 2-dimensional
    gmm_labels = gmm_model.predict(data)
    if data.shape[1] == 2:
        fig = framework.plot_data(data, true_labels, gmm_labels, title=f"Initial GMM Solution (K={n_components}, {covariance_type})")
        if save_plots and fig:
            fig.savefig(os.path.join(output_path, "01_predetermined_gmm_solution.png"),
                        dpi=300, bbox_inches='tight')
            plt.close(fig)
    else:
        print("Data is not 2-dimensional. Skipping initial GMM solution plot.")

    # Evaluate the initial GMM solution
    gmm_eval = framework.evaluate_clustering(data, true_labels, gmm_labels, "Predetermined GMM Initial")

    print("\n" + "="*50)
    print("RUNNING MIXCOMBI ALGORITHM")
    print("="*50)

    # 2. Run MixCombi
    # Initialize MixCombi with the data and the parameters from the fitted GMM
    mixcombi = MixCombi(data, bic_params, icl_params)
    mixcombi_results = mixcombi.combine_components()
    mixcombi.display_results()

    # Plot and save the entropy elbow plot generated by MixCombi
    print(f"\n" + "="*50)
    print("MIXCOMBI ENTROPY ELBOW PLOT")
    print("="*50)

    fig = framework.plot_entropy_elbow(mixcombi_results)
    if save_plots and fig:
        fig.savefig(os.path.join(output_path, "02_entropy_elbow_plot.png"),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    print("\n" + "-"*50)
    print("Please inspect the Entropy Elbow Plot to visually determine the optimal number of clusters.")
    print("-" * 50)

    # 3. Evaluate MixCombi solutions
    print(f"\n" + "="*50)
    print("EVALUATING MIXCOMBI SOLUTIONS")
    print("="*50)

    evaluation_results = {}
    # Iterate through different K values found by MixCombi and evaluate them
    for k in sorted(mixcombi_results.keys(), reverse=True):
        labels_k = mixcombi_results[k]['labels']
        eval_k = framework.evaluate_clustering(data, true_labels, labels_k, f"MixCombi K={k}")
        evaluation_results[k] = eval_k
        print()

        # Save individual solution plots for each K if data is 2D
        if data.shape[1] == 2 and save_plots:
            fig = framework.plot_data(data, true_labels, labels_k,
                                       title=f"MixCombi Solution K={k}")
            if fig:
                fig.savefig(os.path.join(output_path, f"03_mixcombi_solution_k{k}.png"),
                            dpi=300, bbox_inches='tight')
                plt.close(fig)

    # Determine and save the best MixCombi solution based on ARI or Silhouette
    summary_results = {}
    best_k = None
    if evaluation_results:
        if true_labels is not None:
            # If true labels are available, use Adjusted Rand Index (ARI)
            best_k = max(evaluation_results.keys(), key=lambda k_val: evaluation_results[k_val]['ari'])
            comparison_metric = 'ARI'
        else:
            # Otherwise, use Silhouette score
            best_k = max(evaluation_results.keys(), key=lambda k_val: evaluation_results[k_val]['silhouette'])
            comparison_metric = 'Silhouette'

        print(f"BEST MIXCOMBI SOLUTION (Programmatically selected): K={best_k}")
        print(f"  {comparison_metric}: {evaluation_results[best_k][comparison_metric.lower()]:.3f}")

        # Plot the best MixCombi solution
        best_labels = mixcombi_results[best_k]['labels']
        if data.shape[1] == 2 and save_plots:
            fig = framework.plot_data(data, true_labels, best_labels,
                                       title=f"Best MixCombi Solution (K={best_k})")
            if fig:
                fig.savefig(os.path.join(output_path, "04_best_mixcombi_solution.png"),
                            dpi=300, bbox_inches='tight')
                plt.close(fig)

        # Final comparison summary for console output
        print(f"\n" + "="*50)
        print("FINAL COMPARISON")
        print("="*50)

        if true_labels is not None:
            print(f"True clusters: {len(np.unique(true_labels))}")
            print(f"Initial GMM (K={n_components}): (ARI: {gmm_eval['ari']:.3f})")
            print(f"MixCombi best (K={best_k}): (ARI: {evaluation_results[best_k]['ari']:.3f})")

            summary_results = {
                'true_clusters': int(len(np.unique(true_labels))),
                'gmm_initial_k': int(n_components),
                'gmm_initial_covariance_type': covariance_type,
                'gmm_initial_ari': float(gmm_eval['ari']),
                'mixcombi_best': int(best_k),
                'mixcombi_best_ari': float(evaluation_results[best_k]['ari']),
                'comparison_metric': comparison_metric
            }
        else:
            print(f"Initial GMM (K={n_components}): (Silhouette: {gmm_eval['silhouette']:.3f})")
            print(f"MixCombi best (by Silhouette, K={best_k}): (Silhouette: {evaluation_results[best_k]['silhouette']:.3f})")

            summary_results = {
                'true_clusters': None,
                'gmm_initial_k': int(n_components),
                'gmm_initial_covariance_type': covariance_type,
                'gmm_initial_silhouette': float(gmm_eval['silhouette']),
                'mixcombi_best': int(best_k),
                'mixcombi_best_silhouette': float(evaluation_results[best_k]['silhouette']),
                'comparison_metric': comparison_metric
            }

    # Save all results to files if save_results is True
    if save_results:
        # Save summary results as JSON
        with open(os.path.join(output_path, "summary_results_predetermined.json"), 'w') as f:
            json.dump(summary_results, f, indent=2)

        # Prepare and save detailed evaluation results as JSON
        eval_results_serializable = {}
        for k, eval_dict in evaluation_results.items():
            eval_results_serializable[str(k)] = {key: float(val) for key, val in eval_dict.items()}

        with open(os.path.join(output_path, "detailed_evaluation_results_predetermined.json"), 'w') as f:
            json.dump(eval_results_serializable, f, indent=2)

        # Save initial GMM evaluation results as JSON
        gmm_eval_serializable = {key: float(val) for key, val in gmm_eval.items()}
        with open(os.path.join(output_path, "gmm_initial_evaluation_results.json"), 'w') as f:
            json.dump(gmm_eval_serializable, f, indent=2)

        # Save raw results as a pickle file for complete reproducibility
        results_pickle = {
            'data': data,
            'true_labels': true_labels,
            'gmm_fit_params': gmm_fit_params,
            'mixcombi_results': mixcombi_results,
            'evaluation_results': evaluation_results,
            'summary_results': summary_results
        }

        with open(os.path.join(output_path, "complete_results_predetermined.pkl"), 'wb') as f:
            pickle.dump(results_pickle, f)

        # Save cluster labels as text files for easy access
        np.savetxt(os.path.join(output_path, "predetermined_gmm_labels.txt"), gmm_labels, fmt='%d')
        if best_k is not None:
            np.savetxt(os.path.join(output_path, "best_mixcombi_labels_predetermined.txt"), mixcombi_results[best_k]['labels'], fmt='%d')

        if true_labels is not None:
            np.savetxt(os.path.join(output_path, "true_labels.txt"), true_labels, fmt='%d')

        print(f"\nResults saved to: {output_path}")
        print("Saved files:")
        print("  - summary_results_predetermined.json (high-level summary)")
        print("  - detailed_evaluation_results_predetermined.json (all metrics for each K)")
        print("  - gmm_initial_evaluation_results.json (metrics for the initial GMM)")
        print("  - complete_results_predetermined.pkl (full results for reproducibility)")
        print("  - *_labels.txt (cluster labels as text files)")
        if save_plots:
            print("  - *.png (all plots)")

    return gmm_fit_params, mixcombi_results, evaluation_results
