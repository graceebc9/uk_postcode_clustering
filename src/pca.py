 

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os 
from sklearn.metrics import mean_absolute_percentage_error as mape
 
from src.column_settings import settings_dict, age_cols, type_cols
from sklearn.model_selection import train_test_split
 

def run_pca(neb, cols , op_path, col_setting):
    op_path_pca = os.path.join(op_path, 'pca', str(col_setting) ) 
    os.makedirs(op_path_pca, exist_ok=True) 

    X = neb[cols]

    # 2. Standardize the data
    # PCA is sensitive to the scale of the variables. It's crucial to standardize the data
    # (mean=0, variance=1) before applying PCA.
    print("\nStandardizing data for PCA...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=cols) # Convert back to DataFrame for easier inspection
    print("Data standardized.")

    # 3. Perform PCA
    # We'll start by fitting PCA to all components to analyze explained variance.
    print("Performing PCA...")
    pca = PCA() # No n_components specified, so it keeps all components
    pca.fit(X_scaled)
    print("PCA completed.")

    # 4. Analyze Explained Variance
    # The explained variance ratio tells you how much variance each principal component explains.
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = explained_variance_ratio.cumsum()

    # print("\nExplained Variance Ratio for each Principal Component:")
    # for i, ratio in enumerate(explained_variance_ratio):
    #     print(f"PC{i+1}: {ratio:.4f} (Cumulative: {cumulative_explained_variance[i]:.4f})")

    # 5. Scree Plot to visualize explained variance
    print("\nGenerating Scree Plot...")
    plt.figure(figsize=(18, 7))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--', color='blue', label='Individual Explained Variance')
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='x', linestyle='-', color='red', label='Cumulative Explained Variance')
    plt.title('Scree Plot: Explained Variance by Principal Component', fontsize=18)
    plt.xlabel('Principal Component Number', fontsize=14)
    plt.ylabel('Explained Variance Ratio', fontsize=14)
    plt.axhline(y=0.95, color='gray', linestyle=':', label='95% Cumulative Variance Threshold') # Example threshold
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    plt.tight_layout()
    # save fig 
    scree_plot_path = os.path.join(op_path_pca, 'scree_plot.png')
    plt.savefig(scree_plot_path)
    plt.show()
    print("Scree Plot generated.")

    # 6. Interpret Principal Components (Loadings)
    # Loadings indicate how much each original variable contributes to each principal component.
    # High absolute loading values indicate a strong relationship.
    print("\nAnalyzing Principal Component Loadings (first 3 components):")
    # You can adjust the number of components to display based on your scree plot
    num_components_to_show = min(3, len(cols)) # Show up to 3 components or fewer if less exist

    pca_components_df = pd.DataFrame(pca.components_[:num_components_to_show], columns=cols,
                                    index=[f'PC{i+1}' for i in range(num_components_to_show)])

    # print("\nPrincipal Component Loadings:")
    # print(pca_components_df.T.round(3)) # Transpose for easier reading (variables as rows)

    # Optional: Visualize loadings as a heatmap
    print("\nGenerating Loadings Heatmap...")
    plt.figure(figsize=(num_components_to_show * 3 + 2, len(cols) * 0.7))
    sns.heatmap(pca_components_df.T, annot=True, cmap='viridis', fmt='.2f', linewidths=0.5, linecolor='black')
    plt.title('Principal Component Loadings', fontsize=18)
    plt.xlabel('Principal Component', fontsize=14)
    plt.ylabel('Original Variable', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    
    # save heatmap
    heatmap_path = os.path.join(op_path_pca, 'loadings_heatmap.png')
    plt.savefig(heatmap_path)
                                
    print("Loadings Heatmap generated.")
    plt.show()

    # 7. Project data onto selected principal components (optional, for further use)
    # Based on the scree plot, you might decide to keep a certain number of components.
    # Let's say you decide to keep components that explain 95% of the variance.
    n_components_selected = len(cols) 
    # print(f"\nBased on 95% cumulative variance, selecting {n_components_selected} principal components.")
    print('selecting all pca components')
    pca_final = PCA(n_components=n_components_selected)
    X_pca = pca_final.fit_transform(X_scaled)
    X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components_selected)])

    print(f"\nShape of original data: {X.shape}")
    print(f"Shape of PCA-transformed data (with {n_components_selected} components): {X_pca.shape}")
    # print("\nFirst 5 rows of PCA-transformed data:")
    # print(X_pca_df.head())
    return X_pca_df , pca_final
