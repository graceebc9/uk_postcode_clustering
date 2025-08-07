import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.stats import multivariate_normal
import seaborn as sns
import time
from typing import Dict, Tuple, List, Optional
 
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
        print(bic_params)
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
