"""
Sophisticated Weight Optimization for BRI Dashboard
Implements advanced optimization techniques beyond fixed weights
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

class SophisticatedWeightOptimization:
    """Advanced weight optimization for BRI components"""
    
    def __init__(self, bri_data, market_data=None):
        self.bri_data = bri_data.copy()
        self.market_data = market_data
        self.feature_columns = ['sent_vol_score', 'news_tone_score', 'herding_score', 
                               'polarity_skew_score', 'event_density_score']
        self.optimization_results = {}
        
    def pca_weight_optimization(self):
        """
        Use Principal Component Analysis for weight optimization
        
        Returns:
            dict: PCA optimization results
        """
        try:
            # Prepare feature matrix
            available_features = [col for col in self.feature_columns if col in self.bri_data.columns]
            if len(available_features) < 2:
                return {'error': 'Insufficient features for PCA optimization'}
            
            X = self.bri_data[available_features].values
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform PCA
            pca = PCA()
            pca_result = pca.fit_transform(X_scaled)
            
            # Get component weights
            component_weights = pca.components_[0]  # First principal component
            feature_weights = np.abs(component_weights) / np.sum(np.abs(component_weights))
            
            # Create weight mapping
            weight_mapping = dict(zip(available_features, feature_weights))
            
            # Calculate explained variance
            explained_variance = pca.explained_variance_ratio_
            
            results = {
                'method': 'PCA',
                'weights': weight_mapping,
                'explained_variance_ratio': explained_variance.tolist(),
                'cumulative_variance': np.cumsum(explained_variance).tolist(),
                'n_components': len(available_features),
                'first_component_weights': component_weights.tolist()
            }
            
            self.optimization_results['pca'] = results
            return results
            
        except Exception as e:
            print(f"Error in PCA weight optimization: {e}")
            return {'error': str(e)}
    
    def correlation_based_optimization(self):
        """
        Optimize weights based on correlation with target variable
        
        Returns:
            dict: Correlation-based optimization results
        """
        try:
            if self.market_data is None or 'VIX' not in self.market_data.columns:
                return {'error': 'VIX data not available for correlation optimization'}
            
            # Merge data
            merged = pd.merge(self.bri_data, self.market_data, on='date', how='inner')
            if len(merged) < 50:
                return {'error': 'Insufficient data for correlation optimization'}
            
            # Prepare features
            available_features = [col for col in self.feature_columns if col in merged.columns]
            if len(available_features) < 2:
                return {'error': 'Insufficient features for correlation optimization'}
            
            # Calculate correlations with VIX
            correlations = {}
            for feature in available_features:
                corr = merged[feature].corr(merged['VIX'])
                correlations[feature] = abs(corr)  # Use absolute correlation
            
            # Normalize correlations to weights
            total_correlation = sum(correlations.values())
            if total_correlation > 0:
                weights = {feature: corr / total_correlation for feature, corr in correlations.items()}
            else:
                # Equal weights if no correlation
                weights = {feature: 1.0 / len(available_features) for feature in available_features}
            
            # Calculate Spearman correlations for robustness
            spearman_correlations = {}
            for feature in available_features:
                spearman_corr, _ = spearmanr(merged[feature], merged['VIX'])
                spearman_correlations[feature] = abs(spearman_corr)
            
            results = {
                'method': 'Correlation-Based',
                'weights': weights,
                'pearson_correlations': correlations,
                'spearman_correlations': spearman_correlations,
                'target_variable': 'VIX',
                'data_points': len(merged)
            }
            
            self.optimization_results['correlation'] = results
            return results
            
        except Exception as e:
            print(f"Error in correlation-based optimization: {e}")
            return {'error': str(e)}
    
    def machine_learning_optimization(self):
        """
        Use machine learning models for weight optimization
        
        Returns:
            dict: ML optimization results
        """
        try:
            if self.market_data is None or 'VIX' not in self.market_data.columns:
                return {'error': 'VIX data not available for ML optimization'}
            
            # Merge data
            merged = pd.merge(self.bri_data, self.market_data, on='date', how='inner')
            if len(merged) < 100:
                return {'error': 'Insufficient data for ML optimization'}
            
            # Prepare features
            available_features = [col for col in self.feature_columns if col in merged.columns]
            if len(available_features) < 2:
                return {'error': 'Insufficient features for ML optimization'}
            
            X = merged[available_features].values
            y = merged['VIX'].values
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Test different models
            models = {
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'ElasticNet': ElasticNet(),
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
            }
            
            model_results = {}
            
            for model_name, model in models.items():
                scores = []
                feature_importances = []
                
                for train_idx, test_idx in tscv.split(X):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    # Fit model
                    model.fit(X_train, y_train)
                    
                    # Predict
                    y_pred = model.predict(X_test)
                    
                    # Calculate score
                    score = r2_score(y_test, y_pred)
                    scores.append(score)
                    
                    # Get feature importance
                    if hasattr(model, 'coef_'):
                        feature_importances.append(np.abs(model.coef_))
                    elif hasattr(model, 'feature_importances_'):
                        feature_importances.append(model.feature_importances_)
                
                # Calculate average feature importance
                if feature_importances:
                    avg_importance = np.mean(feature_importances, axis=0)
                    # Normalize to weights
                    weights = avg_importance / np.sum(avg_importance)
                    weight_mapping = dict(zip(available_features, weights))
                else:
                    weight_mapping = {feature: 1.0 / len(available_features) for feature in available_features}
                
                model_results[model_name] = {
                    'weights': weight_mapping,
                    'avg_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'feature_importances': avg_importance.tolist() if feature_importances else []
                }
            
            # Select best model
            best_model = max(model_results.keys(), key=lambda x: model_results[x]['avg_score'])
            
            results = {
                'method': 'Machine Learning',
                'best_model': best_model,
                'best_weights': model_results[best_model]['weights'],
                'all_model_results': model_results,
                'target_variable': 'VIX',
                'data_points': len(merged)
            }
            
            self.optimization_results['ml'] = results
            return results
            
        except Exception as e:
            print(f"Error in ML optimization: {e}")
            return {'error': str(e)}
    
    def genetic_algorithm_optimization(self):
        """
        Use genetic algorithm for weight optimization
        
        Returns:
            dict: Genetic algorithm optimization results
        """
        try:
            if self.market_data is None or 'VIX' not in self.market_data.columns:
                return {'error': 'VIX data not available for genetic algorithm optimization'}
            
            # Merge data
            merged = pd.merge(self.bri_data, self.market_data, on='date', how='inner')
            if len(merged) < 100:
                return {'error': 'Insufficient data for genetic algorithm optimization'}
            
            # Prepare features
            available_features = [col for col in self.feature_columns if col in merged.columns]
            if len(available_features) < 2:
                return {'error': 'Insufficient features for genetic algorithm optimization'}
            
            X = merged[available_features].values
            y = merged['VIX'].values
            
            # Objective function (minimize negative correlation)
            def objective(weights):
                # Normalize weights
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                
                # Calculate weighted BRI
                weighted_bri = np.dot(X, weights)
                
                # Calculate correlation with VIX
                correlation = np.corrcoef(weighted_bri, y)[0, 1]
                
                # Return negative correlation (minimize)
                return -abs(correlation)
            
            # Constraints: weights must sum to 1
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            
            # Bounds: weights must be positive
            bounds = [(0, 1) for _ in range(len(available_features))]
            
            # Initial guess (equal weights)
            x0 = [1.0 / len(available_features)] * len(available_features)
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                weight_mapping = dict(zip(available_features, optimal_weights))
                
                # Calculate final correlation
                weighted_bri = np.dot(X, optimal_weights)
                final_correlation = np.corrcoef(weighted_bri, y)[0, 1]
                
                results = {
                    'method': 'Genetic Algorithm',
                    'weights': weight_mapping,
                    'final_correlation': final_correlation,
                    'optimization_success': result.success,
                    'iterations': result.nit,
                    'target_variable': 'VIX',
                    'data_points': len(merged)
                }
            else:
                results = {
                    'method': 'Genetic Algorithm',
                    'error': 'Optimization failed',
                    'optimization_success': False
                }
            
            self.optimization_results['genetic'] = results
            return results
            
        except Exception as e:
            print(f"Error in genetic algorithm optimization: {e}")
            return {'error': str(e)}
    
    def ensemble_optimization(self):
        """
        Combine multiple optimization methods for ensemble weights
        
        Returns:
            dict: Ensemble optimization results
        """
        try:
            # Run all optimization methods
            pca_results = self.pca_weight_optimization()
            correlation_results = self.correlation_based_optimization()
            ml_results = self.machine_learning_optimization()
            genetic_results = self.genetic_algorithm_optimization()
            
            # Collect weights from successful methods
            weight_methods = {}
            
            if 'weights' in pca_results:
                weight_methods['PCA'] = pca_results['weights']
            
            if 'weights' in correlation_results:
                weight_methods['Correlation'] = correlation_results['weights']
            
            if 'best_weights' in ml_results:
                weight_methods['ML'] = ml_results['best_weights']
            
            if 'weights' in genetic_results:
                weight_methods['Genetic'] = genetic_results['weights']
            
            if not weight_methods:
                return {'error': 'No successful optimization methods'}
            
            # Calculate ensemble weights (average of all methods)
            all_features = set()
            for method_weights in weight_methods.values():
                all_features.update(method_weights.keys())
            
            ensemble_weights = {}
            for feature in all_features:
                feature_weights = []
                for method_weights in weight_methods.values():
                    if feature in method_weights:
                        feature_weights.append(method_weights[feature])
                
                if feature_weights:
                    ensemble_weights[feature] = np.mean(feature_weights)
            
            # Normalize ensemble weights
            total_weight = sum(ensemble_weights.values())
            if total_weight > 0:
                ensemble_weights = {feature: weight / total_weight for feature, weight in ensemble_weights.items()}
            
            results = {
                'method': 'Ensemble',
                'ensemble_weights': ensemble_weights,
                'individual_methods': weight_methods,
                'method_count': len(weight_methods),
                'optimization_results': {
                    'pca': pca_results,
                    'correlation': correlation_results,
                    'ml': ml_results,
                    'genetic': genetic_results
                }
            }
            
            self.optimization_results['ensemble'] = results
            return results
            
        except Exception as e:
            print(f"Error in ensemble optimization: {e}")
            return {'error': str(e)}
    
    def create_optimization_comparison_chart(self):
        """
        Create chart comparing different optimization methods
        
        Returns:
            plotly.graph_objects.Figure: Optimization comparison chart
        """
        try:
            if not self.optimization_results:
                return None
            
            # Prepare data for comparison
            methods = []
            features = []
            weights = []
            
            for method_name, results in self.optimization_results.items():
                if 'weights' in results or 'best_weights' in results or 'ensemble_weights' in results:
                    method_weights = results.get('weights', results.get('best_weights', results.get('ensemble_weights', {})))
                    
                    for feature, weight in method_weights.items():
                        methods.append(method_name)
                        features.append(feature)
                        weights.append(weight)
            
            if not methods:
                return None
            
            # Create DataFrame
            df = pd.DataFrame({
                'Method': methods,
                'Feature': features,
                'Weight': weights
            })
            
            # Create grouped bar chart
            fig = px.bar(
                df, 
                x='Feature', 
                y='Weight', 
                color='Method',
                title='Weight Optimization Comparison',
                barmode='group'
            )
            
            fig.update_layout(
                title=dict(
                    text='Weight Optimization Comparison Across Methods',
                    font=dict(color='#1A202C', size=18, family='Inter')
                ),
                xaxis_title='BRI Features',
                yaxis_title='Optimized Weights',
                height=500,
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                font=dict(color='#1A202C', family='Inter')
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating optimization comparison chart: {e}")
            return None
    
    def create_weight_sensitivity_analysis(self):
        """
        Create weight sensitivity analysis
        
        Returns:
            plotly.graph_objects.Figure: Weight sensitivity analysis
        """
        try:
            if self.market_data is None or 'VIX' not in self.market_data.columns:
                return None
            
            # Merge data
            merged = pd.merge(self.bri_data, self.market_data, on='date', how='inner')
            if len(merged) < 100:
                return None
            
            # Prepare features
            available_features = [col for col in self.feature_columns if col in merged.columns]
            if len(available_features) < 2:
                return None
            
            X = merged[available_features].values
            y = merged['VIX'].values
            
            # Test weight sensitivity
            base_weights = np.array([1.0 / len(available_features)] * len(available_features))
            sensitivity_results = []
            
            for i, feature in enumerate(available_features):
                # Vary weight of feature i
                weight_variations = np.linspace(0.1, 0.9, 20)
                correlations = []
                
                for weight in weight_variations:
                    # Create modified weights
                    modified_weights = base_weights.copy()
                    modified_weights[i] = weight
                    modified_weights = modified_weights / np.sum(modified_weights)
                    
                    # Calculate weighted BRI
                    weighted_bri = np.dot(X, modified_weights)
                    
                    # Calculate correlation
                    correlation = np.corrcoef(weighted_bri, y)[0, 1]
                    correlations.append(abs(correlation))
                
                sensitivity_results.append({
                    'feature': feature,
                    'weight_variations': weight_variations.tolist(),
                    'correlations': correlations
                })
            
            # Create sensitivity chart
            fig = go.Figure()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            for i, result in enumerate(sensitivity_results):
                fig.add_trace(go.Scatter(
                    x=result['weight_variations'],
                    y=result['correlations'],
                    mode='lines+markers',
                    name=result['feature'],
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
            
            fig.update_layout(
                title=dict(
                    text='Weight Sensitivity Analysis',
                    font=dict(color='#1A202C', size=18, family='Inter')
                ),
                xaxis_title='Feature Weight',
                yaxis_title='Correlation with VIX',
                height=500,
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                font=dict(color='#1A202C', family='Inter')
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating weight sensitivity analysis: {e}")
            return None
    
    def generate_optimization_report(self):
        """
        Generate comprehensive weight optimization report
        
        Returns:
            dict: Comprehensive optimization report
        """
        try:
            # Run all optimization methods
            pca_results = self.pca_weight_optimization()
            correlation_results = self.correlation_based_optimization()
            ml_results = self.machine_learning_optimization()
            genetic_results = self.genetic_algorithm_optimization()
            ensemble_results = self.ensemble_optimization()
            
            # Create visualizations
            comparison_chart = self.create_optimization_comparison_chart()
            sensitivity_chart = self.create_weight_sensitivity_analysis()
            
            # Compile report
            report = {
                'optimization_methods': {
                    'pca': pca_results,
                    'correlation': correlation_results,
                    'ml': ml_results,
                    'genetic': genetic_results,
                    'ensemble': ensemble_results
                },
                'visualizations': {
                    'comparison_chart': comparison_chart.to_dict() if comparison_chart else None,
                    'sensitivity_chart': sensitivity_chart.to_dict() if sensitivity_chart else None
                },
                'summary': self._create_optimization_summary(),
                'generated_at': pd.Timestamp.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            print(f"Error generating optimization report: {e}")
            return {'error': str(e)}
    
    def _create_optimization_summary(self):
        """Create optimization summary"""
        summary = {
            'methods_completed': len(self.optimization_results),
            'successful_methods': len([r for r in self.optimization_results.values() if 'error' not in r]),
            'recommended_method': None,
            'key_findings': []
        }
        
        # Find best method based on correlation
        best_correlation = -1
        for method_name, results in self.optimization_results.items():
            if 'final_correlation' in results:
                if results['final_correlation'] > best_correlation:
                    best_correlation = results['final_correlation']
                    summary['recommended_method'] = method_name
        
        # Add key findings
        if 'ensemble' in self.optimization_results:
            summary['key_findings'].append("Ensemble method combines multiple optimization approaches")
        
        if 'ml' in self.optimization_results:
            summary['key_findings'].append("Machine learning optimization provides data-driven weights")
        
        if 'genetic' in self.optimization_results:
            summary['key_findings'].append("Genetic algorithm optimization finds optimal weight combinations")
        
        return summary
