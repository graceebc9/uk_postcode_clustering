import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.stats import multivariate_normal
import seaborn as sns
import time
from typing import Dict, Tuple, List, Optional
 
from .MixCombi import MixCombi
from .MixCombiTestFrameWork import MixCombiTestFrameWork 
  
import os
import json
import pickle
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

def run_gmm_mixcombi_on_data(data: np.ndarray, output_path: str, true_labels: Optional[np.ndarray] = None, 
                        k_min: int = 1, k_max: int = 12, save_plots: bool = True, save_results: bool = True):
    """
    Main function to run the complete MixCombi framework on a given dataset and save results.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data for clustering
    output_path : str
        Path to directory where results and plots will be saved
    true_labels : Optional[np.ndarray]
        Ground truth labels if available
    k_min : int
        Minimum number of clusters to consider
    k_max : int
        Maximum number of clusters to consider
    save_plots : bool
        Whether to save plots to disk
    save_results : bool
        Whether to save numerical results to disk
    
    Returns:
    --------
    tuple: (data, true_labels, gmm_results, mixcombi_results, evaluation_results)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    framework = MixCombiTestFramework()
    
    print("=" * 80)
    print("RUNNING MIXCOMBI FRAMEWORK ON PROVIDED DATA")
    print(f"Output directory: {output_path}")
    print("=" * 80)
    
    # Plot and save input data
    if data.shape[1] == 2:
        if true_labels is not None:
            fig = framework.plot_data(data, true_labels=true_labels, title="Input Data (Ground Truth)")
            if save_plots and fig:
                fig.savefig(os.path.join(output_path, "01_input_data_ground_truth.png"), 
                           dpi=300, bbox_inches='tight')
                plt.close(fig)
        else:
            print("Input data provided without ground truth labels.")
    else:
        print("Data is not 2-dimensional. Skipping data plots.")
    
    # Fit GMM with information criteria
    gmm_results = framework.fit_gmm_with_ics(data, k_min=k_min, k_max=k_max)
    
    # BIC solution
    bic_params = gmm_results['bic_params']
    bic_labels = gmm_results['best_bic_model'].predict(data)
    
    print(f"\nBIC selected {bic_params['K']} components.")
    if data.shape[1] == 2:
        fig = framework.plot_data(data, true_labels, bic_labels, title="BIC Solution")
        if save_plots and fig:
            fig.savefig(os.path.join(output_path, "02_bic_solution.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    bic_eval = framework.evaluate_clustering(data, true_labels, bic_labels, "BIC")
    
    print(f"\n" + "="*50)
    print("RUNNING MIXCOMBI")
    print("="*50)
    
    # Run MixCombi
    mixcombi = MixCombi(data, bic_params, gmm_results['icl_params'])
    mixcombi_results = mixcombi.combine_components()
    mixcombi.display_results()
    
    # Plot and save entropy elbow plot
    print(f"\n" + "="*50)
    print("MIXCOMBI ENTROPY ELBOW PLOT")
    print("="*50)
    
    fig = framework.plot_entropy_elbow(mixcombi_results)
    if save_plots and fig:
        fig.savefig(os.path.join(output_path, "03_entropy_elbow_plot.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)

    print("\n" + "-"*50)
    print("Please inspect the Entropy Elbow Plot to visually determine the optimal number of clusters.")
    print("-" * 50)
    
    # Evaluate MixCombi solutions
    print(f"\n" + "="*50)
    print("EVALUATING MIXCOMBI SOLUTIONS")
    print("="*50)
    
    evaluation_results = {}
    
    for k in sorted(mixcombi_results.keys(), reverse=True):
        labels_k = mixcombi_results[k]['labels']
        eval_k = framework.evaluate_clustering(data, true_labels, labels_k, f"MixCombi K={k}")
        evaluation_results[k] = eval_k
        print()
        
        # Save individual solution plots
        if data.shape[1] == 2 and save_plots:
            fig = framework.plot_data(data, true_labels, labels_k, 
                                    title=f"MixCombi Solution K={k}")
            if fig:
                fig.savefig(os.path.join(output_path, f"04_mixcombi_solution_k{k}.png"), 
                           dpi=300, bbox_inches='tight')
                plt.close(fig)
    
    # Determine and save best solution
    if evaluation_results:
        if true_labels is not None:
            best_k = max(evaluation_results.keys(), key=lambda k: evaluation_results[k]['ari'])
            comparison_metric = 'ARI'
        else:
            best_k = max(evaluation_results.keys(), key=lambda k: evaluation_results[k]['silhouette'])
            comparison_metric = 'Silhouette'

        print(f"BEST MIXCOMBI SOLUTION (Programmatically selected): K={best_k}")
        print(f"  {comparison_metric}: {evaluation_results[best_k][comparison_metric.lower()]:.3f}")
        
        best_labels = mixcombi_results[best_k]['labels']
        if data.shape[1] == 2 and save_plots:
            fig = framework.plot_data(data, true_labels, best_labels, 
                              title=f"Best MixCombi Solution (K={best_k})")
            if fig:
                fig.savefig(os.path.join(output_path, "05_best_mixcombi_solution.png"), 
                           dpi=300, bbox_inches='tight')
                plt.close(fig)
    
    # Final comparison
    print(f"\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    
    summary_results = {}
    if true_labels is not None:
        print(f"True clusters: {len(np.unique(true_labels))}")
        print(f"BIC selected: {bic_params['K']} (ARI: {bic_eval['ari']:.3f})")
        print(f"MixCombi best: {best_k} (ARI: {evaluation_results[best_k]['ari']:.3f})")
        
        summary_results = {
            'true_clusters': int(len(np.unique(true_labels))),
            'bic_selected': int(bic_params['K']),
            'bic_ari': float(bic_eval['ari']),
            'mixcombi_best': int(best_k),
            'mixcombi_best_ari': float(evaluation_results[best_k]['ari']),
            'comparison_metric': comparison_metric
        }
    else:
        print(f"BIC selected: {bic_params['K']} (Silhouette: {bic_eval['silhouette']:.3f})")
        print(f"MixCombi best (by Silhouette): {best_k} (Silhouette: {evaluation_results[best_k]['silhouette']:.3f})")
        
        summary_results = {
            'true_clusters': None,
            'bic_selected': int(bic_params['K']),
            'bic_silhouette': float(bic_eval['silhouette']),
            'mixcombi_best': int(best_k),
            'mixcombi_best_silhouette': float(evaluation_results[best_k]['silhouette']),
            'comparison_metric': comparison_metric
        }
    
    # Save results to files
    if save_results:
        # Save summary results as JSON
        with open(os.path.join(output_path, "summary_results.json"), 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        # Save detailed evaluation results as JSON
        eval_results_serializable = {}
        for k, eval_dict in evaluation_results.items():
            eval_results_serializable[str(k)] = {key: float(val) for key, val in eval_dict.items()}
        
        with open(os.path.join(output_path, "detailed_evaluation_results.json"), 'w') as f:
            json.dump(eval_results_serializable, f, indent=2)
        
        # Save BIC evaluation results
        bic_eval_serializable = {key: float(val) for key, val in bic_eval.items()}
        with open(os.path.join(output_path, "bic_evaluation_results.json"), 'w') as f:
            json.dump(bic_eval_serializable, f, indent=2)
        
        # Save raw results as pickle for complete reproducibility
        results_pickle = {
            'data': data,
            'true_labels': true_labels,
            'gmm_results': gmm_results,
            'mixcombi_results': mixcombi_results,
            'evaluation_results': evaluation_results,
            'summary_results': summary_results
        }
        
        with open(os.path.join(output_path, "complete_results.pkl"), 'wb') as f:
            pickle.dump(results_pickle, f)
        
        # Save labels as text files for easy access
        np.savetxt(os.path.join(output_path, "bic_labels.txt"), bic_labels, fmt='%d')
        np.savetxt(os.path.join(output_path, "best_mixcombi_labels.txt"), best_labels, fmt='%d')
        
        if true_labels is not None:
            np.savetxt(os.path.join(output_path, "true_labels.txt"), true_labels, fmt='%d')
        
        print(f"\nResults saved to: {output_path}")
        print("Saved files:")
        print("  - summary_results.json (high-level summary)")
        print("  - detailed_evaluation_results.json (all metrics for each K)")
        print("  - bic_evaluation_results.json (BIC solution metrics)")
        print("  - complete_results.pkl (full results for reproducibility)")
        print("  - *_labels.txt (cluster labels as text files)")
        if save_plots:
            print("  - *.png (all plots)")
    
    return data, true_labels, gmm_results, mixcombi_results, evaluation_results
 
