"""Data Loading and Preprocessing: Loads the fish school feature data from output/fish_schools_features.csv, handles missing values and standardizes the features.
Optimal Cluster Number Determination: Uses multiple methods (silhouette score, Calinski-Harabasz index, Davies-Bouldin index, and elbow method) to determine the optimal number of clusters.
Dimensionality Reduction: Applies three techniques (PCA, t-SNE, and UMAP) to visualize high-dimensional data in 2D space.
Clustering Algorithms: Implements three clustering methods:

K-means for partitioning with spherical clusters
DBSCAN for density-based clustering with noise detection
HDBSCAN for hierarchical density-based clustering


Cluster Evaluation: Assesses cluster quality using silhouette score, Calinski-Harabasz score, and Davies-Bouldin index.
Stability Analysis: Tests how stable each clustering method is when resampling the data.
Visualizations: Creates multiple visualizations including:

Dimensionality reduction plots
Cluster visualizations for each method
Feature importance heatmaps
Feature distribution plots


Detailed Analysis: Provides extensive analysis of cluster characteristics:

Identifying distinctive features for each cluster
Generating feature profiles
Comparing performance across clustering methods


Comprehensive Report: Generates a summary report with recommendations on the most appropriate clustering method.

To run this script, save it to a file (e.g., fish_school_clustering.py) and execute it using Python. Make sure the output/fish_schools_features.csv file exists. The script creates a clustering_results directory with all the analysis outputs."""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

import umap
import hdbscan
import os
from scipy.stats import mode
from kneed import KneeLocator
import warnings

from typing import Optional, Tuple, List

warnings.filterwarnings('ignore')


class Pipeline:
    
    def __init__ (self, data_path: str, output_dir: str,n_components:int=3 ,scaled: bool = True):
        
        self.reduced_data = None
        self.clustering_results = None
        self.data = None
        self.scaled_data = None
        self.data_path: str = data_path
        self.output_dir: str = output_dir
        self.scaled: bool = scaled
        self.n_components: int = n_components
        self.best_k = None
    
    def __load_data (self, **kwargs) -> pd.DataFrame:
        
        """Loads the data"""
        
        data = pd.read_csv(self.data_path, **kwargs)
        
        return data
    
    def load_and_preprocess_data (self, **kwargs) -> Tuple[
        pd.DataFrame, Optional[pd.DataFrame | None]]:
        
        """Load features from CSV and preprocess them for clustering."""
        
        print(f"Loading data from {self.data_path}...")
        
        df = self.__load_data(**kwargs)
        
        print(f"Loaded dataset with {df.shape[0]} samples and {df.shape[1]} features.")
        
        numeric_cols = df.select_dtypes(include=np.number).columns
        
        df = df[numeric_cols]
        
        if df.isnull().sum().sum() > 0:
            df = df.fillna(df.mean())
        
        if np.isinf(df.values).sum() > 0:
            df = df.replace([np.inf, -np.inf], [1e9, -1e9])
        
        scaled_data = None
        
        if self.scaled:
            scaler = StandardScaler()
            
            scaled_data = scaler.fit_transform(df)
        
        self.data = df
        self.scaled_data = scaled_data
        
        return df, scaled_data
    
    def __find_optimal_k (self,
                          k_range=range(2, 11),
                          plot: bool = True,
                          ) \
            -> Tuple[int, Optional[plt.Figure | None]]:
        
        """Find the optimal number of clusters using multiple metrics."""
        
        print("Finding optimal number of clusters...")
        
        data = self.data
        
        if self.scaled:
            data = self.scaled_data
        
        silhouette_scores: List[float] = []
        calinski_scores: List[float] = []
        davies_bouldin_scores: List[float] = []
        inertia_values: List[float] = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            
            labels = kmeans.fit_predict(data)
            
            silhouette_scores.append(silhouette_score(data, labels))
            calinski_scores.append(calinski_harabasz_score(data, labels))
            davies_bouldin_scores.append(davies_bouldin_score(data, labels))
            inertia_values.append(kmeans.inertia_)
        
        fig = None
        
        if plot:
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            
            axs[0, 0].plot(list(k_range), silhouette_scores, 'bo-')
            axs[0, 0].set_xlabel('Number of Clusters (k)')
            axs[0, 0].set_ylabel('Silhouette Score')
            axs[0, 0].set_title('Silhouette Score Method (Higher is better)')
            axs[0, 0].grid(True)
            
            # Calinski-Harabasz score (higher is better)
            axs[0, 1].plot(list(k_range), calinski_scores, 'ro-')
            axs[0, 1].set_xlabel('Number of Clusters (k)')
            axs[0, 1].set_ylabel('Calinski-Harabasz Score')
            axs[0, 1].set_title('Calinski-Harabasz Method (Higher is better)')
            axs[0, 1].grid(True)
            
            # Davies-Bouldin score (lower is better)
            axs[1, 0].plot(list(k_range), davies_bouldin_scores, 'go-')
            axs[1, 0].set_xlabel('Number of Clusters (k)')
            axs[1, 0].set_ylabel('Davies-Bouldin Score')
            axs[1, 0].set_title('Davies-Bouldin Method (Lower is better)')
            axs[1, 0].grid(True)
            
            # Elbow method plot
            axs[1, 1].plot(list(k_range), inertia_values, 'mo-')
            axs[1, 1].set_xlabel('Number of Clusters (k)')
            axs[1, 1].set_ylabel('Inertia (Within-Cluster Sum of Squares)')
            axs[1, 1].set_title('Elbow Method')
            axs[1, 1].grid(True)
            
            plt.tight_layout()
            plt.show()
            fig.savefig(os.path.join(self.output_dir, "optimal_k_analysis.png"), dpi=300, bbox_inches='tight')
        
        # Determine optimal k based on metrics
        best_k_silhouette = k_range[np.argmax(silhouette_scores)]
        best_k_calinski = k_range[np.argmax(calinski_scores)]
        best_k_davies = k_range[np.argmin(davies_bouldin_scores)]
        
        # Find elbow point
        kneedle = KneeLocator(list(k_range), inertia_values, S=1.0, curve="convex", direction="decreasing")
        
        best_k_elbow = kneedle.elbow if kneedle.elbow else k_range[1]  # Default to second value if no clear elbow
        
        print(f"Best k according to Silhouette score: {best_k_silhouette}")
        print(f"Best k according to Calinski-Harabasz score: {best_k_calinski}")
        print(f"Best k according to Davies-Bouldin score: {best_k_davies}")
        print(f"Best k according to Elbow method: {best_k_elbow}")
        
        # Take the mode or median of the recommendations
        best_k_values = [best_k_silhouette, best_k_calinski, best_k_davies, best_k_elbow]
        
        best_k = int(mode(best_k_values, keepdims=True)[0][0])
        
        print(f"Recommended optimal number of clusters: {best_k}")
        
        self.best_k = best_k
        
        return best_k, fig
    
    def apply_dimensionality_reduction (self , methods=None):
        """Apply selected dimensionality reduction techniques and return the results."""
        n_components = self.n_components
        
        if methods is None:
            methods = ['pca', 'tsne', 'umap']
        
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError("n_components must be a positive integer.")
        
        data = self.scaled_data if self.scaled else self.data
        
        if data is None:
            raise ValueError(
                "No data available. Make sure to load and optionally scale your data before applying dimensionality reduction.")
        
        if n_components > data.shape[1]:
            raise ValueError(
                f"n_components ({n_components}) cannot be greater than number of features ({data.shape[1]}).")
        
        print("Applying dimensionality reduction techniques...")
        results = {}
        
        if 'pca' in methods:
            pca = PCA(n_components=n_components, random_state=42)
            pca_result = pca.fit_transform(data)
            print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
            results['pca'] = pca_result
        
        if 'tsne' in methods:
            try:
                perplexity = min(30, len(data) - 1)
                tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
                tsne_result = tsne.fit_transform(data)
                results['tsne'] = tsne_result
            except Exception as e:
                print(f"t-SNE failed: {e}")
        
        if 'umap' in methods:
            try:
                _umap = umap.UMAP(n_components=n_components, n_neighbors=15, random_state=42)
                umap_result = _umap.fit_transform(data)
                results['umap'] = umap_result
            except Exception as e:
                print(f"UMAP failed: {e}")
                
        self.reduced_data = results
        
        
        return results
    
    def visualize_reduced_data (self, plots_dir: str):
        """Visualize the dimensionality reduction results using Plotly (2D or 3D)."""
        print("Visualizing dimensionality reduction results with Plotly...")
        
        reduced_data = self.apply_dimensionality_reduction()
        # if self.scaled:
        #     reduced_data = self.scaled_data
        
        os.makedirs(os.path.join(self.output_dir, plots_dir), exist_ok=True)
        
        for method_name, result in reduced_data.items():
            if result.shape[1] == 2:
                fig = px.scatter(
                    x=result[:, 0],
                    y=result[:, 1],
                    title=f"{method_name.upper()} Projection (2D)",
                    labels={'x': 'Dimension 1', 'y': 'Dimension 2'}
                )
            elif result.shape[1] == 3:
                fig = px.scatter_3d(
                    x=result[:, 0],
                    y=result[:, 1],
                    z=result[:, 2],
                    title=f"{method_name.upper()} Projection (3D)",
                    labels={'x': 'Dimension 1', 'y': 'Dimension 2', 'z': 'Dimension 3'}
                )
            else:
                print(f"Skipping {method_name}: Expected 2 or 3 dimensions, got {result.shape[1]}")
                continue
            
            # Save the plot as an HTML file
            output_path = os.path.join(os.path.join(self.output_dir, plots_dir), f"{method_name}_projection.html")
            fig.show()
            fig.write_html(output_path)
        
        print(f"Saved dimensionality reduction visualizations to {os.path.join(self.output_dir, plots_dir)}")
    
    def apply_clustering_methods (self):
        """Apply different clustering methods to the data."""
        
        data = self.data
        if self.scaled:
            data = self.scaled_data
        
        optimal_k,_ = self.__find_optimal_k()
        
        print("Applying various clustering methods...")
        clustering_results = {}
        
        # K-means
        print("Running K-means clustering...")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(data)
        
        clustering_results['kmeans'] = {
            'labels': kmeans_labels,
            'model': kmeans,
            'name': 'K-means'
        }
        
        # DBSCAN - find good eps parameter
        print("Finding optimal eps for DBSCAN...")
        neighbors = NearestNeighbors(n_neighbors=min(10, len(data) - 1))
        neighbors_fit = neighbors.fit(data)
        distances, indices = neighbors_fit.kneighbors(data)
        
        # Sort distances in ascending order for k-distance graph
        distances_sorted = np.sort(distances, axis=0)
        
        # Plot k-distance graph to help find eps
        plt.figure(figsize=(10, 6))
        plt.plot(distances_sorted[:, min(5, len(data) - 1)])
        plt.xlabel('Data Points (sorted)')
        plt.ylabel('Distance to 5th Nearest Neighbor')
        plt.title('K-Distance Graph for DBSCAN eps Parameter Selection')
        plt.grid(True)
        
        # Find the point of maximum curvature in the k-distance graph
        x = np.arange(len(distances_sorted))
        knee = KneeLocator(x, distances_sorted[:, min(5, len(data) - 1)],
                           curve='convex', direction='increasing', S=1.0)
        knee_point = 0
        
        if knee.knee is not None:
            knee_point = knee.knee
            eps = distances_sorted[knee_point, min(5, len(data) - 1)]
        else:
            # Take the mean of the 5th nearest neighbor distance as a fallback
            eps = np.mean(distances_sorted[:, min(5, len(data) - 1)])
        
        os.makedirs(os.path.join(self.output_dir,"clustering_results"),exist_ok=True)
        plt.axhline(y=eps, color='r', linestyle='--')
        plt.axvline(x=knee_point if knee.knee is not None else len(distances_sorted) // 2, color='r', linestyle='--')
        plt.savefig(os.path.join(self.output_dir,"clustering_results", "dbscan_eps_selection.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Selected eps for DBSCAN: {eps:.4f}")
        
        # Run DBSCAN with the selected eps
        print("Running DBSCAN clustering...")
        min_samples = min(5, int(len(data) / 10))
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(data)
        
        # Check if DBSCAN found a reasonable number of clusters
        n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        print(f"DBSCAN found {n_clusters_dbscan} clusters and {np.sum(dbscan_labels == -1)} outliers")
        
        clustering_results['dbscan'] = {
            'labels': dbscan_labels,
            'model': dbscan,
            'name': 'DBSCAN'
        }
        
        # HDBSCAN
        print("Running HDBSCAN clustering...")
        min_cluster_size = max(5, int(len(data) / 20))  # Adjust based on data size
        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                        min_samples=None,
                                        prediction_data=True)
        hdbscan_labels = hdbscan_model.fit_predict(data)
        
        n_clusters_hdbscan = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
        print(f"HDBSCAN found {n_clusters_hdbscan} clusters and {np.sum(hdbscan_labels == -1)} outliers")
        
        clustering_results['hdbscan'] = {
            'labels': hdbscan_labels,
            'model': hdbscan_model,
            'name': 'HDBSCAN'
        }
        self.clustering_results = clustering_results
        return clustering_results
    
    def evaluate_clustering (self):
        """Evaluate the quality of each clustering method."""
        data=self.data
        if self.scaled:
            data = self.scaled_data
            
        
        print("\nEvaluating clustering quality:")
        evaluation_results = {}
        
        clustering_results = self.clustering_results if self.clustering_results else self.apply_clustering_methods()
        
        for method_name, result in clustering_results.items():
            labels = result['labels']
            
            # Skip evaluation if all samples are outliers or only one cluster
            unique_labels = set(labels) - {-1}  # Remove outlier class if present
            if len(unique_labels) <= 1:
                print(f"{method_name}: Insufficient clusters for evaluation")
                evaluation_results[method_name] = {
                    'silhouette': float('nan'),
                    'calinski': float('nan'),
                    'davies': float('nan')
                }
                continue
            
            # Create a version of the data and labels without outliers for evaluation
            mask = labels != -1
            if np.sum(mask) < 2:  # Need at least 2 samples for evaluation
                print(f"{method_name}: Too many outliers, cannot evaluate")
                evaluation_results[method_name] = {
                    'silhouette': float('nan'),
                    'calinski': float('nan'),
                    'davies': float('nan')
                }
                continue
            
            data_no_outliers = data[mask]
            labels_no_outliers = labels[mask]
            
            # Calculate metrics
            try:
                silhouette = silhouette_score(data_no_outliers, labels_no_outliers)
                calinski = calinski_harabasz_score(data_no_outliers, labels_no_outliers)
                davies = davies_bouldin_score(data_no_outliers, labels_no_outliers)
                
                print(f"{method_name}:")
                print(f"  Silhouette Score: {silhouette:.4f} (higher is better)")
                print(f"  Calinski-Harabasz Score: {calinski:.4f} (higher is better)")
                print(f"  Davies-Bouldin Score: {davies:.4f} (lower is better)")
                
                evaluation_results[method_name] = {
                    'silhouette': silhouette,
                    'calinski': calinski,
                    'davies': davies
                }
            except Exception as e:
                print(f"{method_name}: Evaluation failed - {e}")
                evaluation_results[method_name] = {
                    'silhouette': float('nan'),
                    'calinski': float('nan'),
                    'davies': float('nan')
                }
        
        return evaluation_results


    
    def check_cluster_stability (self, n_iterations=10, sample_size=0.8, save_dir=None):
        """Check how stable the clusters are by resampling."""
        print("\nChecking cluster stability...")
        
        data = self.data
        
        clustering_results = self.clustering_results if self.clustering_results else self.apply_clustering_methods()
        
        if self.scaled:
            data = self.scaled_data
        
        # Create output directory if it doesn't exist
        if save_dir:
            os.makedirs(os.path.join(self.output_dir, save_dir), exist_ok=True)
        
        stability_results = {}
        
        for method_name, result in clustering_results.items():
            print(f"Checking stability for {method_name}...")
            model = result['model']
            original_labels = result['labels']
            
            # Skip methods with mostly outliers
            if np.sum(original_labels == -1) > len(data) * 0.5:
                print(f"  Skipping {method_name} - too many outliers")
                continue
            
            # Check if we can resample this model
            try:
                # For K-means, we can just refit
                if method_name == 'kmeans':
                    agreement_scores = []
                    
                    for i in range(n_iterations):
                        # Sample data
                        # indices = np.random.choice(len(data), size=int(len(data) * sample_size), replace=False)
                        
                        # Fit model
                        new_model = KMeans(n_clusters=model.n_clusters, random_state=i, n_init=10)
                        
                        # Predict on full dataset
                        full_labels = new_model.fit_predict(data)
                        
                        # Calculate agreement with original labels
                        # Need to account for label permutation
                        agreement = adjusted_rand_score(original_labels, full_labels)
                        agreement_scores.append(agreement)
                    
                    mean_agreement = np.mean(agreement_scores)
                    stability_results[method_name] = mean_agreement
                    print(f"  Stability score: {mean_agreement:.4f} (higher is better)")
                
                # For DBSCAN, it's trickier - we'll sample and refit
                elif method_name == 'dbscan':
                    agreement_scores = []
                    
                    for i in range(n_iterations):
                        # Sample data
                        indices = np.random.choice(len(data), size=int(len(data) * sample_size), replace=False)
                        sampled_data = data[indices]
                        
                        # Fit model
                        new_model = DBSCAN(eps=model.eps, min_samples=model.min_samples)
                        sampled_labels = new_model.fit_predict(sampled_data)
                        
                        # For DBSCAN, we need to compare only the samples that are in both clusterings
                        # Filter out outliers from both
                        sampled_mask = sampled_labels != -1
                        original_mask = original_labels[indices] != -1
                        common_mask = sampled_mask & original_mask
                        
                        # Skip if not enough common samples
                        if np.sum(common_mask) < 10:
                            continue
                        
                        # Calculate agreement score
                        agreement = adjusted_rand_score(
                            sampled_labels[common_mask],
                            original_labels[indices][common_mask]
                        )
                        agreement_scores.append(agreement)
                    
                    if agreement_scores:
                        mean_agreement = np.mean(agreement_scores)
                        stability_results[method_name] = mean_agreement
                        print(f"  Stability score: {mean_agreement:.4f} (higher is better)")
                    else:
                        print(f"  Could not compute stability score for {method_name}")
                
                # For HDBSCAN, we can use the prediction method with different samples
                elif method_name == 'hdbscan':
                    agreement_scores = []
                    
                    for i in range(n_iterations):
                        # Sample data
                        indices = np.random.choice(len(data), size=int(len(data) * sample_size), replace=False)
                        sampled_data = data[indices]
                        
                        # Fit model
                        new_model = hdbscan.HDBSCAN(
                            min_cluster_size=model.min_cluster_size,
                            min_samples=model.min_samples,
                            prediction_data=True
                        )
                        sampled_labels = new_model.fit_predict(sampled_data)
                        
                        # Same as DBSCAN, filter outliers
                        sampled_mask = sampled_labels != -1
                        original_mask = original_labels[indices] != -1
                        common_mask = sampled_mask & original_mask
                        
                        # Skip if not enough common samples
                        if np.sum(common_mask) < 10:
                            continue
                        
                        # Calculate agreement score
                        agreement = adjusted_rand_score(
                            sampled_labels[common_mask],
                            original_labels[indices][common_mask]
                        )
                        agreement_scores.append(agreement)
                    
                    if agreement_scores:
                        mean_agreement = np.mean(agreement_scores)
                        stability_results[method_name] = mean_agreement
                        print(f"  Stability score: {mean_agreement:.4f} (higher is better)")
                    else:
                        print(f"  Could not compute stability score for {method_name}")
            
            except Exception as e:
                print(f"  Error checking stability for {method_name}: {e}")
                
                # Create a bar chart of stability scores
        
        if stability_results and save_dir:
            methods = list(stability_results.keys())
            scores = list(stability_results.values())
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(methods, scores, color='skyblue')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{height:.3f}', ha='center', va='bottom')
            
            plt.title('Cluster Stability Scores (Higher is Better)')
            plt.ylabel('Adjusted Rand Score')
            plt.ylim(0, 1.1)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.join(self.output_dir, save_dir), "stability_comparison.png"), dpi=300,
                        bbox_inches='tight')
            plt.show()
        
        return stability_results
    
    def visualize_clusters (self, save_dir: str):
        """Visualize clusters using Plotly with different dimensionality reduction techniques."""
        print("\nCreating interactive cluster visualizations...")
        
        clustering_results = self.clustering_results if self.clustering_results else self.apply_clustering_methods()
        
        os.makedirs(os.path.join(self.output_dir, save_dir), exist_ok=True)
        
        for dim_method, dim_data in self.reduced_data.items():
            dim = dim_data.shape[1]
            
            for method_name, result in clustering_results.items():
                labels = result['labels']
                n_outliers = np.sum(labels == -1)
                unique_labels = np.unique(labels)
                n_clusters = len(unique_labels[unique_labels != -1])
                
                # Assign a distinct color to each cluster
                cluster_names = [str(label) if label != -1 else "Outlier" for label in labels]
                
                title = f"{method_name} Clustering on {dim_method.upper()} ({n_clusters} clusters, {n_outliers} outliers)"
                
                if dim == 2:
                    fig = px.scatter(
                        x=dim_data[:, 0],
                        y=dim_data[:, 1],
                        color=cluster_names,
                        title=title,
                        labels={'x': f"{dim_method.upper()} Dimension 1", 'y': f"{dim_method.upper()} Dimension 2"},
                        opacity=0.7
                    )
                elif dim == 3:
                    fig = px.scatter_3d(
                        x=dim_data[:, 0],
                        y=dim_data[:, 1],
                        z=dim_data[:, 2],
                        color=cluster_names,
                        title=title,
                        labels={
                            'x': f"{dim_method.upper()} Dimension 1",
                            'y': f"{dim_method.upper()} Dimension 2",
                            'z': f"{dim_method.upper()} Dimension 3"
                        },
                        opacity=0.7
                    )
                else:
                    print(f"Skipping {method_name} on {dim_method}: Expected 2 or 3 dimensions, got {dim}")
                    continue
                
                output_path = os.path.join(self.output_dir, save_dir, f"{dim_method}_{method_name}.html")
                fig.show()
                fig.write_html(output_path)
        
        print(f"Saved interactive cluster visualizations to {os.path.join(self.data_path, save_dir)}")
    
    def analyze_cluster_characteristics (self, save_dir: str):
        """Analyze what features define each cluster."""
        print("\nAnalyzing cluster characteristics...")
        
        original_df = self.data
        
        clustering_results = self.clustering_results if self.clustering_results else self.apply_clustering_methods()
        # Create output directory if it doesn't exist
        os.makedirs(os.path.join(self.output_dir, save_dir), exist_ok=True)
        
        # For each clustering method
        for method_name, result in clustering_results.items():
            labels = result['labels']
            
            # Skip if only outliers or no clusters
            if len(set(labels) - {-1}) == 0:
                print(f"{method_name}: No valid clusters to analyze")
                continue
            
            # Ignore outliers
            if -1 in set(labels):
                clean_df = original_df.copy()
                clean_df['cluster'] = labels
                clean_df = clean_df[clean_df['cluster'] != -1]
                labels_clean = clean_df['cluster'].values
                clean_df = clean_df.drop('cluster', axis=1)
            else:
                clean_df = original_df.copy()
                labels_clean = labels
            
            if len(set(labels_clean)) <= 1:
                print(f"{method_name}: Not enough clusters to analyze")
                continue
            
            # Create a DataFrame with cluster assignments
            df_with_clusters = clean_df.copy()
            df_with_clusters['cluster'] = labels_clean
            
            # Calculate mean values for each feature in each cluster
            cluster_means = df_with_clusters.groupby('cluster').mean()
            
            # Calculate overall means
            overall_means = clean_df.mean()
            
            # Calculate how much each cluster differs from the overall mean
            # Avoid division by zero by adding a small epsilon where mean is zero
            adjusted_means = overall_means.copy()
            adjusted_means[adjusted_means == 0] = 1e-10
            diff_from_mean = (cluster_means - overall_means) / adjusted_means
            
            # Create a heatmap
            plt.figure(figsize=(max(12, len(clean_df.columns) / 2), 8))
            sns.heatmap(diff_from_mean, cmap='coolwarm', center=0,
                        linewidths=.5, cbar_kws={"shrink": .8})
            plt.title(f'Feature Importance Heatmap - {result["name"]}', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.join(self.output_dir, save_dir), f"{method_name}_feature_heatmap.png"),
                        dpi=300, bbox_inches='tight')
            plt.show()
            
            # Feature distribution across clusters
            # Select a subset of features for clarity if there are many
            if len(clean_df.columns) > 10:
                # Find the features with highest variance across clusters
                feature_variance = diff_from_mean.var(axis=0).sort_values(ascending=False)
                top_features = feature_variance.index[:10].tolist()
            else:
                top_features = clean_df.columns.tolist()
            
            # Create box plots for top features
            for feature in top_features:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='cluster', y=feature, data=df_with_clusters)
                plt.title(f'Distribution of {feature} Across Clusters - {result["name"]}')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(os.path.join(self.output_dir, save_dir), f"{method_name}_{feature}_boxplot.png"),
                    dpi=300,
                    bbox_inches='tight')
                plt.show()
            
            # Find top 5 distinctive features for each cluster
            path = os.path.join(os.path.join(self.output_dir, save_dir), f"{method_name}_distinctive_features.txt")
            with open(path, 'w', encoding='utf-8') as f:
                # with open(os.path.join(save_dir, f"{method_name}_distinctive_features.txt"), 'w') as f:
                f.write(f"Distinctive Features for {result['name']} Clustering\n")
                f.write("-" * 60 + "\n\n")
                
                for cluster in sorted(set(labels_clean)):
                    # Get absolute differences
                    cluster_diff = diff_from_mean.loc[cluster].abs()
                    # Get top features
                    top_features = cluster_diff.nlargest(5)
                    
                    f.write(f"Cluster {cluster}:\n")
                    for feature, value in top_features.items():
                        # Get the original difference (with sign)
                        orig_value = diff_from_mean.loc[cluster, feature]
                        direction = "higher" if orig_value > 0 else "lower"
                        f.write(f"  {feature}: {abs(orig_value) * 100:.1f}% {direction} than average\n")
                    f.write("\n")
            
            print(f"Saved {method_name} cluster characteristics analysis")
    
    def compare_clustering_methods (self, save_dir):
        """Create a summary comparison of all clustering methods."""
        print("\nGenerating clustering method comparison summary...")
        
        evaluation_results = self.evaluate_clustering()
        
        stability_results = self.check_cluster_stability(save_dir=os.path.join(self.output_dir, save_dir),
                                                         n_iterations=10)
        
        clustering_results = self.clustering_results if self.clustering_results else self.apply_clustering_methods()
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.join(self.output_dir, save_dir), exist_ok=True)
        
        # Create a DataFrame for comparison
        methods = list(evaluation_results.keys())
        metrics = ['silhouette', 'calinski', 'davies']
        metrics_names = ['Silhouette Score (↑)', 'Calinski-Harabasz Score (↑)', 'Davies-Bouldin Score (↓)']
        
        comparison_data = []
        for method in methods:
            row = [evaluation_results[method].get(metric, float('nan')) for metric in metrics]
            # Add stability if available
            if stability_results and method in stability_results:
                row.append(stability_results[method])
            else:
                row.append(float('nan'))
            comparison_data.append(row)
        
        columns = metrics_names + ['Stability Score (↑)']
        df_comparison = pd.DataFrame(comparison_data, index=methods, columns=columns)
        
        # Save to CSV
        df_comparison.to_csv(os.path.join(self.output_dir,save_dir, "clustering_comparison.csv"))
        
        # Create a heatmap for visual comparison
        plt.figure(figsize=(12, 6))
        mask = np.isnan(df_comparison.values)
        
        # Normalize scores for better visualization
        normalized_df = df_comparison.copy()
        # Reverse Davies-Bouldin so higher is always better
        if 'Davies-Bouldin Score (↓)' in normalized_df.columns:
            normalized_df['Davies-Bouldin Score (↓)'] = -normalized_df['Davies-Bouldin Score (↓)']
        
        # Scale each column to [0, 1]
        for col in normalized_df.columns:
            col_data = normalized_df[col].values
            if not np.all(np.isnan(col_data)):
                col_min = np.nanmin(col_data)
                col_max = np.nanmax(col_data)
                if col_max > col_min:
                    normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
        
        # Create heatmap
        ax = sns.heatmap(normalized_df, annot=df_comparison, fmt=".3f", cmap="YlGnBu",
                         mask=mask, linewidths=.5, cbar_kws={"shrink": .8})
        plt.title('Clustering Method Comparison (Higher is Better for All Metrics)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.join(self.output_dir, save_dir), "method_comparison_heatmap.png"), dpi=300,
                    bbox_inches='tight')
        plt.show()
        
        # Write a summary text file
        summary_path = os.path.join(os.path.join(self.output_dir, save_dir), "clustering_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            # with open(os.path.join(save_dir, "clustering_summary.txt"), 'w') as f:
            f.write("Fish School Clustering Analysis Summary\n")
            f.write("=====================================\n\n")
            
            f.write("Overview of Methods:\n")
            for method in methods:
                f.write(f"- {method.upper()}\n")
            f.write("\n")
            
            f.write("Performance Metrics (↑ = higher is better, ↓ = lower is better):\n")
            f.write(df_comparison.to_string())
            f.write("\n\n")
            
            # Find best method for each metric
            f.write("Best Method Per Metric:\n")
            
            # Silhouette
            if not df_comparison['Silhouette Score (↑)'].isna().all():
                best_silhouette = df_comparison['Silhouette Score (↑)'].idxmax()
                f.write(
                    f"- Best Silhouette Score: {best_silhouette} ({df_comparison.loc[best_silhouette, 'Silhouette Score (↑)']:.4f})\n")
            
            # Calinski
            if not df_comparison['Calinski-Harabasz Score (↑)'].isna().all():
                best_calinski = df_comparison['Calinski-Harabasz Score (↑)'].idxmax()
                f.write(
                    f"- Best Calinski-Harabasz Score: {best_calinski} ({df_comparison.loc[best_calinski, 'Calinski-Harabasz Score (↑)']:.4f})\n")
            
            # Davies
            if not df_comparison['Davies-Bouldin Score (↓)'].isna().all():
                best_davies = df_comparison['Davies-Bouldin Score (↓)'].idxmin()
                f.write(
                    f"- Best Davies-Bouldin Score: {best_davies} ({df_comparison.loc[best_davies, 'Davies-Bouldin Score (↓)']:.4f})\n")
            
            # Stability
            if 'Stability Score (↑)' in df_comparison.columns and not df_comparison['Stability Score (↑)'].isna().all():
                best_stability = df_comparison['Stability Score (↑)'].idxmax()
                f.write(
                    f"- Best Stability Score: {best_stability} ({df_comparison.loc[best_stability, 'Stability Score (↑)']:.4f})\n")
            
            # Determine overall best method
            best_methods = []
            
            if not df_comparison['Silhouette Score (↑)'].isna().all():
                best_methods.append(df_comparison['Silhouette Score (↑)'].idxmax())
            
            if not df_comparison['Calinski-Harabasz Score (↑)'].isna().all():
                best_methods.append(df_comparison['Calinski-Harabasz Score (↑)'].idxmax())
            
            if not df_comparison['Davies-Bouldin Score (↓)'].isna().all():
                best_methods.append(df_comparison['Davies-Bouldin Score (↓)'].idxmin())
            
            if 'Stability Score (↑)' in df_comparison.columns and not df_comparison['Stability Score (↑)'].isna().all():
                best_methods.append(df_comparison['Stability Score (↑)'].idxmax())
            
            # Get most frequent method
            if best_methods:
                overall_best = max(set(best_methods), key=best_methods.count)
                f.write(f"\nOverall recommended clustering method: {overall_best.upper()}\n")
                
                # Add explanation for the recommendation
                f.write("\nRecommendation rationale:\n")
                strengths = []
                
                if not df_comparison['Silhouette Score (↑)'].isna().all():
                    if overall_best == df_comparison['Silhouette Score (↑)'].idxmax():
                        strengths.append("best separation between clusters")
                
                if not df_comparison['Calinski-Harabasz Score (↑)'].isna().all():
                    if overall_best == df_comparison['Calinski-Harabasz Score (↑)'].idxmax():
                        strengths.append("best ratio of between-cluster to within-cluster variance")
                
                if not df_comparison['Davies-Bouldin Score (↓)'].isna().all():
                    if overall_best == df_comparison['Davies-Bouldin Score (↓)'].idxmin():
                        strengths.append("lowest average similarity between clusters")
                
                if 'Stability Score (↑)' in df_comparison.columns and not df_comparison[
                    'Stability Score (↑)'].isna().all():
                    if overall_best == df_comparison['Stability Score (↑)'].idxmax():
                        strengths.append("most stable clusters across data resampling")
                
                f.write(f"The {overall_best.upper()} method provides " + ", ".join(strengths) + ".\n")
            
            # Add summary of cluster counts
            f.write("\nCluster Distribution Summary:\n")
            for method_name, result in clustering_results.items():
                labels = result['labels']
                unique_labels = set(labels)
                if -1 in unique_labels:
                    unique_labels.remove(-1)
                    outlier_count = np.sum(labels == -1)
                    outlier_percent = (outlier_count / len(labels)) * 100
                    f.write(
                        f"- {method_name.upper()}: {len(unique_labels)} clusters, {outlier_count} outliers ({outlier_percent:.1f}%)\n")
                else:
                    f.write(f"- {method_name.upper()}: {len(unique_labels)} clusters, no outliers\n")
    
    def create_feature_profile_for_clusters (self, save_dir: str):
        """Create detailed feature profiles for each cluster."""
        print("\nCreating feature profiles for clusters...")
        
        data = self.data
        clustering_results = self.clustering_results if self.clustering_results else self.apply_clustering_methods()
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.join(self.output_dir, save_dir), exist_ok=True)
        
        # For each clustering method
        for method_name, result in clustering_results.items():
            labels = result['labels']
            
            # Skip if only outliers or no clusters
            if len(set(labels) - {-1}) == 0:
                continue
            
            # Create a DataFrame with cluster assignments
            df_with_clusters = data.copy()
            df_with_clusters['cluster'] = labels
            
            # For each cluster (excluding outliers)
            for cluster in sorted(set(labels) - {-1}):
                cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster]
                
                # Calculate summary statistics
                stats = cluster_data.describe().T
                
                # Drop the 'cluster' column from stats
                if 'cluster' in stats.columns:
                    stats = stats.drop('cluster', axis=0)
                
                # Save statistics to CSV
                stats.to_csv(
                    os.path.join(os.path.join(self.output_dir, save_dir), f"{method_name}_cluster_{cluster}_stats.csv"))
                
                # Create distributions for key features
                if len(data.columns) > 5:
                    # Take top features with highest variance
                    var_features = data.var().sort_values(ascending=False).index[:5].tolist()
                else:
                    var_features = data.columns.tolist()
                
                # Create distribution plots
                plt.figure(figsize=(15, 10))
                for i, feature in enumerate(var_features):
                    plt.subplot(2, 3, i + 1)
                    
                    # Plot histogram of the feature for this cluster
                    sns.histplot(cluster_data[feature], kde=True, color='blue', alpha=0.6, label=f'Cluster {cluster}')
                    
                    # Plot histogram of the feature for all data
                    sns.histplot(df_with_clusters[feature], kde=True, color='gray', alpha=0.3, label='All data')
                    
                    plt.title(f'{feature} Distribution')
                    plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir,save_dir, f"{method_name}_cluster_{cluster}_distributions.png"), dpi=300,
                            bbox_inches='tight')
                plt.show()
                
                print(f"Created profile for {method_name} cluster {cluster}")


def main ():
    pipeline = Pipeline("data/fish_school_features.csv", "test",n_components=3)
    
    pipeline.load_and_preprocess_data()
    pipeline.apply_dimensionality_reduction()
    pipeline.visualize_reduced_data("vis_reduced_data")
    pipeline.apply_clustering_methods()
    pipeline.evaluate_clustering()
    pipeline.visualize_clusters("vis_clusters")
    pipeline.check_cluster_stability(save_dir="stability")
    pipeline.analyze_cluster_characteristics("characteristics")
    pipeline.compare_clustering_methods("comparison")
    pipeline.create_feature_profile_for_clusters("profiles")
    
    print(f"\nAnalysis complete! Results saved to test/")


if __name__ == "__main__":
    main()
