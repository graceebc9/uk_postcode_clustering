import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.stats import multivariate_normal
import seaborn as sns
import time
from typing import Dict, Tuple, List, Optional
 
 
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
