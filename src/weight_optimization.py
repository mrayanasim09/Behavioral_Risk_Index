"""
Advanced Weight Optimization for Behavioral Risk Index
Implements PCA, Grid Search, and Sensitivity Analysis for data-driven weights
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class BRIWeightOptimizer:
    """Advanced weight optimization for Behavioral Risk Index"""
    
    def __init__(self, features_df: pd.DataFrame, target_df: pd.DataFrame):
        """
        Initialize optimizer with features and target data
        
        Args:
            features_df: DataFrame with BRI component features
            target_df: DataFrame with target variables (VIX, volatility, etc.)
        """
        self.features_df = features_df
        self.target_df = target_df
        self.feature_columns = ['sent_vol', 'news_tone', 'herding', 'polarity_skew', 'event_density']
        self.optimal_weights = None
        self.pca_weights = None
        self.sensitivity_results = None
        
    def run_pca_analysis(self) -> dict:
        """Perform Principal Component Analysis on BRI features"""
        print("Running PCA Analysis...")
        
        # Prepare data
        feature_data = self.features_df[self.feature_columns].fillna(0)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        
        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_features)
        
        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Get component loadings (weights)
        loadings = pca.components_
        
        # Calculate PCA-based weights (first component loadings)
        pca_weights = np.abs(loadings[0])
        pca_weights = pca_weights / np.sum(pca_weights)  # Normalize to sum to 1
        
        self.pca_weights = dict(zip(self.feature_columns, pca_weights))
        
        results = {
            'explained_variance_ratio': explained_variance,
            'cumulative_variance': cumulative_variance,
            'loadings': loadings,
            'pca_weights': self.pca_weights,
            'n_components_95': np.argmax(cumulative_variance >= 0.95) + 1
        }
        
        print(f"PCA Results:")
        print(f"  - First component explains {explained_variance[0]:.1%} of variance")
        print(f"  - 95% variance explained by {results['n_components_95']} components")
        print(f"  - PCA Weights: {self.pca_weights}")
        
        return results
    
    def run_grid_search_optimization(self, target_column: str = 'vix') -> dict:
        """Run grid search to optimize weights for target variable"""
        print(f"Running Grid Search Optimization for {target_column}...")
        
        # Prepare data
        feature_data = self.features_df[self.feature_columns].fillna(0)
        
        # Merge with target
        merged_data = pd.merge(
            self.features_df[['date'] + self.feature_columns], 
            self.target_df[['date', target_column]], 
            on='date', 
            how='inner'
        ).dropna()
        
        if merged_data.empty:
            print(f"No overlapping data for {target_column}")
            return {}
        
        X = merged_data[self.feature_columns].values
        y = merged_data[target_column].values
        
        # Define weight search space
        weight_ranges = {
            'sent_vol': np.arange(0.1, 0.5, 0.05),
            'news_tone': np.arange(0.1, 0.4, 0.05),
            'herding': np.arange(0.1, 0.4, 0.05),
            'polarity_skew': np.arange(0.05, 0.2, 0.02),
            'event_density': np.arange(0.1, 0.4, 0.05)
        }
        
        best_score = -np.inf
        best_weights = None
        results = []
        
        # Grid search over weight combinations
        for sent_vol_w in weight_ranges['sent_vol']:
            for news_tone_w in weight_ranges['news_tone']:
                for herding_w in weight_ranges['herding']:
                    for polarity_skew_w in weight_ranges['polarity_skew']:
                        for event_density_w in weight_ranges['event_density']:
                            # Check if weights sum to 1 (within tolerance)
                            total_weight = sent_vol_w + news_tone_w + herding_w + polarity_skew_w + event_density_w
                            if abs(total_weight - 1.0) > 0.01:
                                continue
                            
                            # Calculate weighted BRI
                            weights = np.array([sent_vol_w, news_tone_w, herding_w, polarity_skew_w, event_density_w])
                            bri_scores = np.dot(X, weights) * 100
                            
                            # Calculate correlation with target
                            correlation = np.corrcoef(bri_scores, y)[0, 1]
                            
                            if not np.isnan(correlation) and correlation > best_score:
                                best_score = correlation
                                best_weights = {
                                    'sent_vol': sent_vol_w,
                                    'news_tone': news_tone_w,
                                    'herding': herding_w,
                                    'polarity_skew': polarity_skew_w,
                                    'event_density': event_density_w
                                }
                            
                            results.append({
                                'sent_vol': sent_vol_w,
                                'news_tone': news_tone_w,
                                'herding': herding_w,
                                'polarity_skew': polarity_skew_w,
                                'event_density': event_density_w,
                                'correlation': correlation
                            })
        
        self.optimal_weights = best_weights
        
        print(f"Grid Search Results:")
        print(f"  - Best correlation: {best_score:.4f}")
        print(f"  - Optimal weights: {best_weights}")
        
        return {
            'best_weights': best_weights,
            'best_correlation': best_score,
            'all_results': pd.DataFrame(results)
        }
    
    def run_sensitivity_analysis(self, base_weights: dict = None, perturbation: float = 0.05) -> dict:
        """Run sensitivity analysis on BRI weights"""
        print("Running Sensitivity Analysis...")
        
        if base_weights is None:
            base_weights = {
                'sent_vol': 0.3,
                'news_tone': 0.2,
                'herding': 0.2,
                'polarity_skew': 0.1,
                'event_density': 0.2
            }
        
        # Prepare data
        feature_data = self.features_df[self.feature_columns].fillna(0)
        merged_data = pd.merge(
            self.features_df[['date'] + self.feature_columns], 
            self.target_df[['date', 'vix']], 
            on='date', 
            how='inner'
        ).dropna()
        
        if merged_data.empty:
            print("No overlapping data for sensitivity analysis")
            return {}
        
        X = merged_data[self.feature_columns].values
        y = merged_data['vix'].values
        
        # Calculate base BRI
        base_weights_array = np.array([base_weights[col] for col in self.feature_columns])
        base_bri = np.dot(X, base_weights_array) * 100
        base_correlation = np.corrcoef(base_bri, y)[0, 1]
        
        sensitivity_results = {}
        
        # Test sensitivity for each weight
        for i, feature in enumerate(self.feature_columns):
            perturbations = np.arange(-perturbation, perturbation + 0.01, 0.01)
            correlations = []
            
            for pert in perturbations:
                # Create perturbed weights
                test_weights = base_weights_array.copy()
                test_weights[i] = max(0.01, min(0.8, test_weights[i] + pert))  # Keep within bounds
                
                # Renormalize weights
                test_weights = test_weights / np.sum(test_weights)
                
                # Calculate BRI with perturbed weights
                test_bri = np.dot(X, test_weights) * 100
                test_correlation = np.corrcoef(test_bri, y)[0, 1]
                correlations.append(test_correlation)
            
            sensitivity_results[feature] = {
                'perturbations': perturbations,
                'correlations': correlations,
                'sensitivity': np.std(correlations),
                'max_change': max(correlations) - min(correlations)
            }
        
        self.sensitivity_results = sensitivity_results
        
        print("Sensitivity Analysis Results:")
        for feature, results in sensitivity_results.items():
            print(f"  - {feature}: sensitivity = {results['sensitivity']:.4f}, max_change = {results['max_change']:.4f}")
        
        return sensitivity_results
    
    def optimize_weights_advanced(self, target_column: str = 'vix') -> dict:
        """Advanced optimization using scipy minimize"""
        print(f"Running Advanced Optimization for {target_column}...")
        
        # Prepare data
        merged_data = pd.merge(
            self.features_df[['date'] + self.feature_columns], 
            self.target_df[['date', target_column]], 
            on='date', 
            how='inner'
        ).dropna()
        
        if merged_data.empty:
            print(f"No overlapping data for {target_column}")
            return {}
        
        X = merged_data[self.feature_columns].values
        y = merged_data[target_column].values
        
        def objective(weights):
            """Objective function to maximize correlation"""
            # Ensure weights are positive and sum to 1
            weights = np.abs(weights)
            weights = weights / np.sum(weights)
            
            # Calculate BRI
            bri_scores = np.dot(X, weights) * 100
            
            # Calculate correlation (negative because we want to maximize)
            correlation = np.corrcoef(bri_scores, y)[0, 1]
            return -correlation if not np.isnan(correlation) else 1.0
        
        # Initial weights (theoretical)
        initial_weights = np.array([0.3, 0.2, 0.2, 0.1, 0.2])
        
        # Constraints: weights must sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: each weight between 0.01 and 0.8
        bounds = [(0.01, 0.8) for _ in range(5)]
        
        # Optimize
        result = minimize(
            objective, 
            initial_weights, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        optimal_weights = optimal_weights / np.sum(optimal_weights)  # Ensure normalization
        
        optimal_weights_dict = dict(zip(self.feature_columns, optimal_weights))
        
        # Calculate final correlation
        final_bri = np.dot(X, optimal_weights) * 100
        final_correlation = np.corrcoef(final_bri, y)[0, 1]
        
        print(f"Advanced Optimization Results:")
        print(f"  - Optimal weights: {optimal_weights_dict}")
        print(f"  - Final correlation: {final_correlation:.4f}")
        
        return {
            'optimal_weights': optimal_weights_dict,
            'correlation': final_correlation,
            'success': result.success,
            'message': result.message
        }
    
    def create_optimization_report(self, output_dir: str):
        """Create comprehensive optimization report with visualizations"""
        print("Creating Optimization Report...")
        
        # Create plots directory
        plots_dir = f"{output_dir}/optimization_plots"
        import os
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. PCA Analysis Plot
        if self.pca_weights:
            plt.figure(figsize=(12, 8))
            
            # Explained variance
            plt.subplot(2, 2, 1)
            pca_results = self.run_pca_analysis()
            plt.plot(range(1, len(pca_results['explained_variance_ratio']) + 1), 
                    pca_results['explained_variance_ratio'], 'bo-')
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title('PCA Explained Variance')
            plt.grid(True, alpha=0.3)
            
            # PCA Weights
            plt.subplot(2, 2, 2)
            features = list(self.pca_weights.keys())
            weights = list(self.pca_weights.values())
            plt.bar(features, weights, color='skyblue', edgecolor='black')
            plt.xlabel('Features')
            plt.ylabel('PCA Weight')
            plt.title('PCA-Based Weights')
            plt.xticks(rotation=45)
            
            # Theoretical vs PCA Weights
            plt.subplot(2, 2, 3)
            theoretical_weights = [0.3, 0.2, 0.2, 0.1, 0.2]
            x = np.arange(len(features))
            width = 0.35
            plt.bar(x - width/2, theoretical_weights, width, label='Theoretical', alpha=0.7)
            plt.bar(x + width/2, weights, width, label='PCA', alpha=0.7)
            plt.xlabel('Features')
            plt.ylabel('Weight')
            plt.title('Theoretical vs PCA Weights')
            plt.xticks(x, features, rotation=45)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/pca_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Sensitivity Analysis Plot
        if self.sensitivity_results:
            plt.figure(figsize=(15, 10))
            
            for i, (feature, results) in enumerate(self.sensitivity_results.items()):
                plt.subplot(2, 3, i+1)
                plt.plot(results['perturbations'], results['correlations'], 'b-', linewidth=2)
                plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Base Weight')
                plt.xlabel(f'{feature} Weight Change')
                plt.ylabel('Correlation with VIX')
                plt.title(f'Sensitivity: {feature}')
                plt.grid(True, alpha=0.3)
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/sensitivity_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Weight Comparison Summary
        plt.figure(figsize=(12, 8))
        
        # Collect all weight methods
        weight_methods = {}
        weight_methods['Theoretical'] = [0.3, 0.2, 0.2, 0.1, 0.2]
        
        if self.pca_weights:
            weight_methods['PCA'] = [self.pca_weights[col] for col in self.feature_columns]
        
        if self.optimal_weights:
            weight_methods['Grid Search'] = [self.optimal_weights[col] for col in self.feature_columns]
        
        # Plot comparison
        x = np.arange(len(self.feature_columns))
        width = 0.25
        
        for i, (method, weights) in enumerate(weight_methods.items()):
            plt.bar(x + i*width, weights, width, label=method, alpha=0.7)
        
        plt.xlabel('Features')
        plt.ylabel('Weight')
        plt.title('Weight Comparison Across Methods')
        plt.xticks(x + width, self.feature_columns, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/weight_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Optimization report saved to {plots_dir}/")
    
    def get_optimized_weights(self) -> dict:
        """Get the best optimized weights based on all methods"""
        weights_comparison = {}
        
        # Theoretical weights
        weights_comparison['theoretical'] = {
            'sent_vol': 0.3,
            'news_tone': 0.2,
            'herding': 0.2,
            'polarity_skew': 0.1,
            'event_density': 0.2
        }
        
        # Add PCA weights if available
        if self.pca_weights:
            weights_comparison['pca'] = self.pca_weights
        
        # Add optimal weights if available
        if self.optimal_weights:
            weights_comparison['optimized'] = self.optimal_weights
        
        return weights_comparison
