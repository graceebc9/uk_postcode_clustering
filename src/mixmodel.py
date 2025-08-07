import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.stats import multivariate_normal
import seaborn as sns
import time
from typing import Dict, Tuple, List, Optional

# =================================================================================
#  REAL MixCombi IMPLEMENTATION
# =================================================================================
# This is the real MixCombi implementation.

class MixCombi:
    """
    Python implementation of the Mixture Component Combining algorithm
    from "Combining Mixture Components for Clustering"
    """
    
    def __init__(self, data: np.ndarray, bic_params: Dict, icl_params: Optional[Dict] = None):
        """
        Initialize MixCombi with data and BIC solution parameters
        """
        self.data = data
        self.n, self.d = data.shape
        
        self.bic_K = bic_params['K']
        self.bic_mu = np.array(bic_params['mu'])
        self.bic_S = np.array(bic_params['S'])
        self.bic_p = np.array(bic_params['p'])
        
        self.icl_params = icl_params
        
        self.combined_solutions = {}
        
    def log_safe(self, x: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
        """Safe logarithm to avoid log(0)"""
        return np.log(np.maximum(x, epsilon))
    
    def posterior_probabilities(self, mu: np.ndarray, S: np.ndarray, p: np.ndarray, K: int) -> np.ndarray:
        """
        Compute posterior probabilities tau_ik = P(component k | observation i)
        """
        tau = np.zeros((self.n, K))
        
        for k in range(K):
            tau[:, k] = p[k] * multivariate_normal.pdf(self.data, mu[k], S[k])
        
        row_sums = tau.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-15
        tau = tau / row_sums
        
        return tau
    
    def entropy(self, tau: np.ndarray) -> float:
        """Compute entropy of posterior probabilities"""
        return -np.sum(tau * self.log_safe(tau))
    
    def map_labels(self, tau: np.ndarray) -> np.ndarray:
        """Convert posterior probabilities to hard cluster assignments"""
        return np.argmax(tau, axis=1)
    
    def log_likelihood(self, mu: np.ndarray, S: np.ndarray, p: np.ndarray, K: int) -> float:
        """Compute log-likelihood of the data"""
        ll = 0.0
        for i in range(self.n):
            prob_sum = 0.0
            for k in range(K):
                prob_sum += p[k] * multivariate_normal.pdf(self.data[i], mu[k], S[k])
            ll += np.log(max(prob_sum, 1e-15))
        return ll
    
    def combine_components(self) -> Dict:
        """
        Main combining algorithm - creates hierarchy by combining components
        """
        print("Starting component combining...")
        start_time = time.time()
        
        K_bic = self.bic_K
        
        tau_bic = self.posterior_probabilities(self.bic_mu, self.bic_S, self.bic_p, K_bic)
        self.combined_solutions[K_bic] = {
            'M': np.eye(K_bic),
            'labels': self.map_labels(tau_bic),
            'tau': tau_bic,
            'ent': self.entropy(tau_bic)
        }
        
        for K in range(K_bic - 1, 0, -1):
            print(f"Computing {K}-component solution...")
            
            best_entropy = float('inf')
            best_M = None
            best_tau = None
            
            old_tau = self.combined_solutions[K + 1]['tau']
            for j in range(K + 1):
                for k in range(j + 1, K + 1):
                    M = self._create_combining_matrix(j, k, K)
                    new_tau = np.dot(old_tau, M.T)
                    ent = self.entropy(new_tau)
                    
                    if ent < best_entropy:
                        best_entropy = ent
                        best_M = M
                        best_tau = new_tau
            
            self.combined_solutions[K] = {
                'M': best_M,
                'tau': best_tau,
                'ent': best_entropy,
                'labels': self.map_labels(best_tau)
            }
        
        elapsed_time = time.time() - start_time
        print(f"Combining completed in {elapsed_time:.2f} seconds")
        
        return self.combined_solutions
    
    def _create_combining_matrix(self, j: int, k: int, K: int) -> np.ndarray:
        """
        Create combining matrix M that merges components j and k
        """
        M = np.zeros((K, K + 1))
        new_idx = 0
        for old_idx in range(K + 1):
            if old_idx == j:
                M[new_idx, j] = 1
                M[new_idx, k] = 1
                new_idx += 1
            elif old_idx == k:
                continue
            else:
                M[new_idx, old_idx] = 1
                new_idx += 1
        return M
    
    def display_results(self):
        """Display results table similar to MATLAB output"""
        print("\nCombining Results:")
        print("-" * 60)
        print(f"{'Criterion':<15} {'K':<5} {'ENT':<10} {'LogLik':<10}")
        print("-" * 60)
        
        if self.icl_params:
            icl_ent = self.entropy(self.posterior_probabilities(
                self.icl_params['mu'], 
                self.icl_params['S'], 
                self.icl_params['p'], 
                self.icl_params['K']
            ))
            icl_ll = self.log_likelihood(
                self.icl_params['mu'],
                self.icl_params['S'], 
                self.icl_params['p'],
                self.icl_params['K']
            )
            print(f"{'ICL':<15} {self.icl_params['K']:<5} {icl_ent:<10.0f} {icl_ll:<10.0f}")
        else:
            print(f"{'ICL':<15} {'???':<5} {'???':<10} {'???':<10}")
        
        for K in range(2, self.bic_K + 1):
            if K in self.combined_solutions:
                sol = self.combined_solutions[K]
                ll_combined = self.log_likelihood(self.bic_mu, self.bic_S, self.bic_p, self.bic_K) - sol['ent']
                print(f"{'Combined K=' + str(K):<15} {K:<5} {sol['ent']:<10.0f} {ll_combined:<10.0f}")
        
        print("-" * 60)


# =================================================================================
#  MAIN TESTING FRAMEWORK
# =================================================================================

class MixCombiTestFramework:
    """
    A framework for testing MixCombi with real or synthetic data.
    """

    def plot_data(self, data: np.ndarray, true_labels: Optional[np.ndarray] = None, 
                  predicted_labels: Optional[np.ndarray] = None, title: str = "Data"):
        """Plot the data with true and/or predicted labels. Handles case with no true_labels."""
        
        num_subplots = int(true_labels is not None) + int(predicted_labels is not None)
        if num_subplots == 0:
            print("No labels to plot. Skipping plot.")
            return

        fig, axes = plt.subplots(1, num_subplots, figsize=(8 * num_subplots, 6))
        
        if num_subplots == 1:
            axes = [axes]
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'lime']
        
        plot_idx = 0
        if true_labels is not None:
            ax = axes[plot_idx]
            for i in range(len(np.unique(true_labels))):
                mask = true_labels == i
                ax.scatter(data[mask, 0], data[mask, 1], 
                           c=colors[i % len(colors)], label=f'True Cluster {i}', alpha=0.7, s=30)
            ax.set_title(f'{title} - True Labels ({len(np.unique(true_labels))} clusters)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            plot_idx += 1
        
        if predicted_labels is not None:
            ax = axes[plot_idx]
            for i in range(len(np.unique(predicted_labels))):
                mask = predicted_labels == i
                ax.scatter(data[mask, 0], data[mask, 1], 
                           c=colors[i % len(colors)], label=f'Pred Cluster {i}', alpha=0.7, s=30)
            ax.set_title(f'{title} - Predicted Labels ({len(np.unique(predicted_labels))} clusters)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
        
        plt.tight_layout()
        plt.show()

    
 
    def plot_entropy_elbow(self, results: Dict):
        """Plots the entropy and the difference in entropy to find the elbow point."""
        k_values = sorted(results.keys())
        entropy_scores = [results[k]['ent'] for k in k_values]
        
        # Calculate the difference in entropy (first derivative)
        if len(entropy_scores) >= 2:
            entropy_diff = np.diff(entropy_scores)
            diff_k_values = k_values[1:] # K values for the difference plot
        
        # Create subplot layout
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Primary entropy plot
        ax1.plot(k_values, entropy_scores, 'ro-', linewidth=2, markersize=6, label='Entropy')
        ax1.set_xlabel('Number of Clusters (K)')
        ax1.set_ylabel('Entropy')
        ax1.set_title('MixCombi Entropy Elbow Plot')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(k_values)
        
        # Entropy difference plot
        if len(entropy_scores) >= 2:
            ax2.plot(diff_k_values, entropy_diff, 'gs-', linewidth=2, markersize=6, label='Entropy Difference (Δ)')
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            ax2.set_xlabel('Number of Clusters (K)')
            ax2.set_ylabel('Entropy Difference (Δ)')
            ax2.set_title('Entropy Difference (Rate of Change)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(diff_k_values)
            
            # Highlight potential elbow points (where difference plot flattens out)
            # Find the point where the rate of change is highest (most negative)
            min_idx = np.argmin(entropy_diff)
            elbow_k = diff_k_values[min_idx]
            ax2.scatter(elbow_k, entropy_diff[min_idx], color='red', s=100, zorder=5, 
                        label=f'Potential Elbow (K={elbow_k})')
            ax2.legend()
            
            # Also highlight this point on the main entropy plot
            elbow_entropy = entropy_scores[k_values.index(elbow_k)]
            ax1.scatter(elbow_k, elbow_entropy, color='red', s=100, zorder=5,
                        label=f'Suggested Elbow (K={elbow_k})')
            ax1.legend()
        else:
            ax2.text(0.5, 0.5, 'Need at least 2 K values for difference analysis', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Entropy Difference Analysis')
            
        plt.tight_layout()
       
       
        
        if len(entropy_scores) >= 2:
            print(f"\nElbow Analysis:")
            print(f"Suggested optimal K: {elbow_k}")
            print(f"The entropy difference plot shows the marginal gain of adding each cluster.")
            print(f"The most negative point indicates where the largest decrease in entropy occurred.")

        return fig 

    def fit_gmm_with_ics(self, data: np.ndarray, k_min: int = 1, k_max: int = 12) -> Dict:
        """
        Fits GMM models and calculates BIC, AIC, and ICL for each.
        Returns the parameters for the best BIC and ICL solutions.
        """
        print(f"\nFitting GMM with BIC/ICL selection (K={k_min} to {k_max})...")
        
        models = []
        bic_scores = []
        icl_scores = []
        k_values = range(k_min, k_max + 1)
        
        for k in k_values:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type='full',
                random_state=42,
                max_iter=200
            )
            gmm.fit(data)
            
            bic = gmm.bic(data)
            bic_scores.append(bic)
            
            tau = gmm.predict_proba(data)
            tau_safe = np.maximum(tau, 1e-15)
            entropy = -np.sum(tau * np.log(tau_safe))
            icl = bic - entropy
            icl_scores.append(icl)
            
            models.append(gmm)
            
            print(f"  K={k:2d}: BIC={bic:8.2f}, ICL={icl:8.2f}")
        
        best_k_bic = k_values[np.argmin(bic_scores)]
        best_k_icl = k_values[np.argmin(icl_scores)]
        
        print(f"\nBIC selected K = {best_k_bic}")
        print(f"ICL selected K = {best_k_icl}")
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, bic_scores, 'bo-', label='BIC Score', linewidth=2)
        plt.plot(k_values, icl_scores, 'go-', label='ICL Score', linewidth=2)
        plt.axvline(x=best_k_bic, color='blue', linestyle=':', label=f'BIC choice K={best_k_bic}', alpha=0.7)
        plt.axvline(x=best_k_icl, color='green', linestyle=':', label=f'ICL choice K={best_k_icl}', alpha=0.7)
        plt.xlabel('Number of Components (K)')
        plt.ylabel('Information Criterion')
        plt.title('Model Selection: BIC vs ICL')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        best_gmm_bic = models[np.argmin(bic_scores)]
        best_gmm_icl = models[np.argmin(icl_scores)]
        
        bic_params = {
            'K': best_k_bic,
            'mu': best_gmm_bic.means_,
            'S': best_gmm_bic.covariances_,
            'p': best_gmm_bic.weights_
        }
        
        icl_params = {
            'K': best_k_icl,
            'mu': best_gmm_icl.means_,
            'S': best_gmm_icl.covariances_,
            'p': best_gmm_icl.weights_
        }

        return {
            'bic_params': bic_params,
            'icl_params': icl_params,
            'best_bic_model': best_gmm_bic,
            'best_icl_model': best_gmm_icl
        }
    
    def evaluate_clustering(self, data: np.ndarray, true_labels: Optional[np.ndarray], pred_labels: np.ndarray, 
                          method_name: str = "") -> Dict:
        """
        Evaluates clustering performance. Adjusts for missing ground truth labels.
        """
        ari = -1.0
        if true_labels is not None:
            ari = adjusted_rand_score(true_labels, pred_labels)
        
        sil = -1.0
        if len(np.unique(pred_labels)) > 1:
            sil = silhouette_score(data, pred_labels)
        
        print(f"{method_name} Clustering Evaluation:")
        if true_labels is not None:
            print(f"  Adjusted Rand Index: {ari:.3f} (1.0 = perfect)")
        print(f"  Silhouette Score: {sil:.3f} (higher is better)")
        print(f"  Number of clusters found: {len(np.unique(pred_labels))}")
        
        return {'ari': ari, 'silhouette': sil, 'n_clusters': len(np.unique(pred_labels))}

# def run_mixcombi_on_data(data: np.ndarray, true_labels: Optional[np.ndarray] = None, k_min: int = 1, k_max: int = 12):
#     """
#     Main function to run the complete MixCombi framework on a given dataset.
#     """
#     framework = MixCombiTestFramework()
    
#     print("=" * 80)
#     print("RUNNING MIXCOMBI FRAMEWORK ON PROVIDED DATA")
#     print("=" * 80)
    
#     if data.shape[1] == 2:
#         if true_labels is not None:
#             framework.plot_data(data, true_labels=true_labels, title="Input Data (Ground Truth)")
#         else:
#             print("Input data provided without ground truth labels.")
#     else:
#         print("Data is not 2-dimensional. Skipping data plots.")
    
#     gmm_results = framework.fit_gmm_with_ics(data, k_min=k_min, k_max=k_max)
    
#     bic_params = gmm_results['bic_params']
#     bic_labels = gmm_results['best_bic_model'].predict(data)
    
#     print(f"\nBIC selected {bic_params['K']} components.")
#     if data.shape[1] == 2:
#         framework.plot_data(data, true_labels, bic_labels, title="BIC Solution")
#     bic_eval = framework.evaluate_clustering(data, true_labels, bic_labels, "BIC")
    
#     print(f"\n" + "="*50)
#     print("RUNNING MIXCOMBI")
#     print("="*50)
    
#     mixcombi = MixCombi(data, bic_params, gmm_results['icl_params'])
#     mixcombi_results = mixcombi.combine_components()
#     mixcombi.display_results()
    
#     # NEW STEP: Plot the Entropy Elbow Plot
#     print(f"\n" + "="*50)
#     print("MIXCOMBI ENTROPY ELBOW PLOT")
#     print("="*50)
    
#     framework.plot_entropy_elbow(mixcombi_results)

#     print("\n" + "-"*50)
#     print("Please inspect the Entropy Elbow Plot to visually determine the optimal number of clusters.")
#     print("-" * 50)
    
#     # The following evaluation is still useful for comparison, so we'll keep it.
#     print(f"\n" + "="*50)
#     print("EVALUATING MIXCOMBI SOLUTIONS")
#     print("="*50)
    
#     evaluation_results = {}
    
#     for k in sorted(mixcombi_results.keys(), reverse=True):
#         labels_k = mixcombi_results[k]['labels']
#         eval_k = framework.evaluate_clustering(data, true_labels, labels_k, f"MixCombi K={k}")
#         evaluation_results[k] = eval_k
#         print()
    
#     if evaluation_results:
#         if true_labels is not None:
#             best_k = max(evaluation_results.keys(), key=lambda k: evaluation_results[k]['ari'])
#             comparison_metric = 'ARI'
#         else:
#             best_k = max(evaluation_results.keys(), key=lambda k: evaluation_results[k]['silhouette'])
#             comparison_metric = 'Silhouette'

#         print(f"BEST MIXCOMBI SOLUTION (Programmatically selected): K={best_k}")
#         print(f"  {comparison_metric}: {evaluation_results[best_k][comparison_metric.lower()]:.3f}")
        
#         best_labels = mixcombi_results[best_k]['labels']
#         if data.shape[1] == 2:
#             framework.plot_data(data, true_labels, best_labels, 
#                           title=f"Best MixCombi Solution (K={best_k})")
    
#     print(f"\n" + "="*50)
#     print("FINAL COMPARISON")
#     print("="*50)
#     if true_labels is not None:
#         print(f"True clusters: {len(np.unique(true_labels))}")
#         print(f"BIC selected: {bic_params['K']} (ARI: {bic_eval['ari']:.3f})")
#         print(f"MixCombi best: {best_k} (ARI: {evaluation_results[best_k]['ari']:.3f})")
#     else:
#         print(f"BIC selected: {bic_params['K']} (Silhouette: {bic_eval['silhouette']:.3f})")
#         print(f"MixCombi best (by Silhouette): {best_k} (Silhouette: {evaluation_results[best_k]['silhouette']:.3f})")
    
#     return data, true_labels, gmm_results, mixcombi_results, evaluation_results
 

import os
import json
import pickle
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

def run_mixcombi_on_data(data: np.ndarray, output_path: str, true_labels: Optional[np.ndarray] = None, 
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
 

# class MixCombi:
#     """
#     Python implementation of the Mixture Component Combining algorithm
#     from "Combining Mixture Components for Clustering"
#     """
    
#     def __init__(self, data: np.ndarray, bic_params: Dict, icl_params: Optional[Dict] = None):
#         """
#         Initialize MixCombi with data and BIC solution parameters
        
#         Args:
#             data: n x d matrix containing the sample data
#             bic_params: Dictionary containing BIC solution parameters:
#                 - 'K': number of components
#                 - 'mu': K x d matrix of component means
#                 - 'S': d x d x K array of covariance matrices
#                 - 'p': 1 x K vector of mixing proportions
#             icl_params: Optional dictionary with ICL solution parameters (same structure)
#         """
#         self.data = data
#         self.n, self.d = data.shape
        
#         # BIC parameters
#         self.bic_K = bic_params['K']
#         self.bic_mu = np.array(bic_params['mu'])
#         self.bic_S = np.array(bic_params['S'])
#         self.bic_p = np.array(bic_params['p'])
        
#         # ICL parameters (optional)
#         self.icl_params = icl_params
        
#         # Results storage
#         self.combined_solutions = {}
        
#     def log_safe(self, x: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
#         """Safe logarithm to avoid log(0)"""
#         return np.log(np.maximum(x, epsilon))
    
#     def posterior_probabilities(self, mu: np.ndarray, S: np.ndarray, p: np.ndarray, K: int) -> np.ndarray:
#         """
#         Compute posterior probabilities tau_ik = P(component k | observation i)
        
#         Args:
#             mu: K x d matrix of component means
#             S: d x d x K array of covariance matrices
#             p: K-length vector of mixing proportions
#             K: number of components
            
#         Returns:
#             n x K matrix of posterior probabilities
#         """
#         tau = np.zeros((self.n, K))
        
#         # Compute likelihood for each component
#         for k in range(K):
#             if S.ndim == 3:
#                 cov_k = S[:, :, k]
#             else:
#                 cov_k = S
#             tau[:, k] = p[k] * multivariate_normal.pdf(self.data, mu[k], cov_k)
        
#         # Normalize to get posterior probabilities
#         row_sums = tau.sum(axis=1, keepdims=True)
#         row_sums[row_sums == 0] = 1e-15  # Avoid division by zero
#         tau = tau / row_sums
        
#         return tau
    
#     def entropy(self, tau: np.ndarray) -> float:
#         """Compute entropy of posterior probabilities"""
#         return -np.sum(tau * self.log_safe(tau))
    
#     def map_labels(self, tau: np.ndarray) -> np.ndarray:
#         """Convert posterior probabilities to hard cluster assignments"""
#         return np.argmax(tau, axis=1)
    
#     def log_likelihood(self, mu: np.ndarray, S: np.ndarray, p: np.ndarray, K: int) -> float:
#         """Compute log-likelihood of the data"""
#         ll = 0.0
#         for i in range(self.n):
#             prob_sum = 0.0
#             for k in range(K):
#                 if S.ndim == 3:
#                     cov_k = S[:, :, k]
#                 else:
#                     cov_k = S
#                 prob_sum += p[k] * multivariate_normal.pdf(self.data[i], mu[k], cov_k)
#             ll += np.log(max(prob_sum, 1e-15))
#         return ll
    
#     def combine_components(self) -> Dict:
#         """
#         Main combining algorithm - creates hierarchy by combining components
        
#         Returns:
#             Dictionary containing all combined solutions
#         """
#         print("Starting component combining...")
#         start_time = time.time()
        
#         # Initialize with BIC solution
#         K_bic = self.bic_K
        
#         # Store BIC solution
#         tau_bic = self.posterior_probabilities(self.bic_mu, self.bic_S, self.bic_p, K_bic)
#         self.combined_solutions[K_bic] = {
#             'M': np.eye(K_bic),
#             'labels': self.map_labels(tau_bic),
#             'tau': tau_bic,
#             'ent': self.entropy(tau_bic)
#         }
        
#         # Iteratively combine components
#         for K in range(K_bic - 1, 0, -1):
#             print(f"Computing {K}-component solution...")
            
#             best_entropy = float('inf')
#             best_M = None
#             best_tau = None
            
#             # Try all possible pairs of components to combine
#             # We need to combine 2 components from K+1 to get K components
#             for j in range(K + 1):
#                 for k in range(j + 1, K + 1):
#                     # Create combining matrix M
#                     M = self._create_combining_matrix(j, k, K)
                    
#                     # Compute new posterior probabilities
#                     old_tau = self.combined_solutions[K + 1]['tau']
#                     new_tau = np.dot(old_tau, M.T)
                    
#                     # Compute entropy
#                     ent = self.entropy(new_tau)
                    
#                     # Keep best solution (minimum entropy)
#                     if ent < best_entropy:
#                         best_entropy = ent
#                         best_M = M
#                         best_tau = new_tau
            
#             # Store best solution for this K
#             self.combined_solutions[K] = {
#                 'M': best_M,
#                 'tau': best_tau,
#                 'ent': best_entropy,
#                 'labels': self.map_labels(best_tau)
#             }
        
#         elapsed_time = time.time() - start_time
#         print(f"Combining completed in {elapsed_time:.2f} seconds")
        
#         return self.combined_solutions
    
#     def _create_combining_matrix(self, j: int, k: int, K: int) -> np.ndarray:
#         """
#         Create combining matrix M that merges components j and k
        
#         Args:
#             j, k: indices of components to combine (j < k)
#             K: target number of components
            
#         Returns:
#             K x (K+1) combining matrix
#         """
#         M = np.zeros((K, K + 1))
        
#         # Map old components to new components
#         new_idx = 0
#         for old_idx in range(K + 1):
#             if old_idx == j:
#                 # Component j gets combined with component k
#                 M[new_idx, j] = 1
#                 M[new_idx, k] = 1
#                 new_idx += 1
#             elif old_idx == k:
#                 # Component k is merged into j, skip
#                 continue
#             else:
#                 # All other components map one-to-one
#                 M[new_idx, old_idx] = 1
#                 new_idx += 1
        
#         return M
    
#     def display_results(self):
#         """Display results table similar to MATLAB output"""
#         print("\nCombining Results:")
#         print("-" * 60)
#         print(f"{'Criterion':<15} {'K':<5} {'ENT':<8} {'LogLik':<10}")
#         print("-" * 60)
        
#         # ICL row (if available)
#         if self.icl_params:
#             icl_tau = self.posterior_probabilities(
#                 self.icl_params['mu'], 
#                 self.icl_params['S'], 
#                 self.icl_params['p'], 
#                 self.icl_params['K']
#             )
#             icl_ent = self.entropy(icl_tau)
#             icl_ll = self.log_likelihood(
#                 self.icl_params['mu'],
#                 self.icl_params['S'], 
#                 self.icl_params['p'],
#                 self.icl_params['K']
#             )
#             print(f"{'ICL':<15} {self.icl_params['K']:<5} {icl_ent:<8.0f} {icl_ll:<10.0f}")
#         else:
#             print(f"{'ICL':<15} {'???':<5} {'???':<8} {'???':<10}")
        
#         # Combined solutions
#         for K in range(2, self.bic_K):
#             if K in self.combined_solutions:
#                 sol = self.combined_solutions[K]
#                 ll_combined = self.log_likelihood(self.bic_mu, self.bic_S, self.bic_p, self.bic_K) - sol['ent']
#                 print(f"{'Combined K=' + str(K):<15} {K:<5} {sol['ent']:<8.0f} {ll_combined:<10.0f}")
        
#         # BIC row
#         bic_ll = self.log_likelihood(self.bic_mu, self.bic_S, self.bic_p, self.bic_K)
#         print(f"{'BIC':<15} {self.bic_K:<5} {self.combined_solutions[self.bic_K]['ent']:<8.0f} {bic_ll:<10.0f}")
#         print("-" * 60)
