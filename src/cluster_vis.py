import matplotlib.pyplot as plt
import seaborn as sns

def visualize_cluster_distributions(X, cols, figsize=(15, 10)):
    """
    Visualize distributions of variables across clusters
    
    Parameters:
    X: DataFrame with cluster column
    cols: list of column names to visualize
    figsize: tuple for figure size
    """
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create subplots
    n_vars = len(cols)
    n_cols = 3  # 3 columns of subplots
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    # Plot each variable
    for i, col in enumerate(cols):
        ax = axes[i]
        
        # Box plot showing distribution by cluster
        sns.boxplot(data=X, x='cluster', y=col, ax=ax)
        ax.set_title(f'Distribution of {col} by Cluster')
        ax.set_xlabel('Cluster')
        ax.set_ylabel(col)
        
        # Add mean points
        cluster_means = X.groupby('cluster')[col].mean()
        for cluster, mean_val in cluster_means.items():
            ax.scatter(cluster, mean_val, color='red', s=100, marker='D', 
                      label='Mean' if cluster == cluster_means.index[0] else "")
        
        if i == 0:  # Add legend only to first subplot
            ax.legend()
    
    # Remove empty subplots
    for i in range(n_vars, len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    plt.show()

# Alternative: Violin plots for more detailed distributions
def visualize_cluster_violins(X, cols, figsize=(15, 10)):
    """
    Violin plots showing detailed distributions
    """
    n_vars = len(cols)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, col in enumerate(cols):
        ax = axes[i]
        sns.violinplot(data=X, x='cluster', y=col, ax=ax)
        ax.set_title(f'Distribution of {col} by Cluster')
        ax.set_xlabel('Cluster')
        ax.set_ylabel(col)
    
    for i in range(n_vars, len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    plt.show()

# Heatmap of cluster means with scaling
def visualize_cluster_heatmap(X, cols, scaled=True):
    """
    Heatmap showing mean values of each variable by cluster
    
    Parameters:
    X: DataFrame with cluster column
    cols: list of column names
    scaled: bool, whether to use StandardScaler to normalize values
    """
    from sklearn.preprocessing import StandardScaler
    
    if scaled:
        # Scale the data first
        scaler = StandardScaler()
        X_scaled = X.copy()
        X_scaled[cols] = scaler.fit_transform(X[cols])
        cluster_means = X_scaled.groupby('cluster')[cols].mean()
        title = 'Standardized Mean Values by Cluster'
        fmt = '.2f'
    else:
        cluster_means = X.groupby('cluster')[cols].mean()
        title = 'Mean Values by Cluster'
        fmt = '.2f'
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_means.T, annot=True, cmap='RdBu_r', fmt=fmt, 
                center=0 if scaled else None, cbar_kws={'label': 'Standard Deviations from Mean' if scaled else 'Value'})
    plt.title(title)
    plt.xlabel('Cluster')
    plt.ylabel('Variables')
    plt.show()

# Usage example:
# visualize_cluster_distributions(X, cols)
# visualize_cluster_violins(X, cols)
# visualize_cluster_heatmap(X, cols, scaled=True)  # With scaling
# visualize_cluster_heatmap(X, cols, scaled=False)  # Without scaling
