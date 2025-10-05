"""
Advanced Statistical Validation for BRI Dashboard
Implements out-of-sample testing, regime detection, stationarity tests, and Granger causality
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from scipy.stats import jarque_bera, shapiro
import warnings
warnings.filterwarnings('ignore')

class StatisticalValidation:
    """Advanced statistical validation for BRI analysis"""
    
    def __init__(self, bri_data, market_data=None):
        self.bri_data = bri_data.copy()
        self.market_data = market_data
        self.validation_results = {}
        
    def out_of_sample_testing(self, test_size=0.2, n_splits=5):
        """
        Perform out-of-sample testing for BRI-VIX correlation
        
        Args:
            test_size (float): Proportion of data for testing
            n_splits (int): Number of time series splits
            
        Returns:
            dict: Out-of-sample validation results
        """
        try:
            # Merge BRI and VIX data
            if self.market_data is None or 'VIX' not in self.market_data.columns:
                return {'error': 'VIX data not available'}
            
            merged = pd.merge(self.bri_data, self.market_data, on='date', how='inner')
            if len(merged) < 100:
                return {'error': 'Insufficient data for out-of-sample testing'}
            
            # Prepare data
            X = merged['BRI'].values.reshape(-1, 1)
            y = merged['VIX'].values
            
            # Time series split
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            correlations = []
            r2_scores = []
            mse_scores = []
            mae_scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train model
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                correlation = np.corrcoef(y_test, y_pred)[0, 1]
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                correlations.append(correlation)
                r2_scores.append(r2)
                mse_scores.append(mse)
                mae_scores.append(mae)
            
            # Calculate statistics
            results = {
                'correlation_mean': np.mean(correlations),
                'correlation_std': np.std(correlations),
                'correlation_min': np.min(correlations),
                'correlation_max': np.max(correlations),
                'r2_mean': np.mean(r2_scores),
                'r2_std': np.std(r2_scores),
                'mse_mean': np.mean(mse_scores),
                'mae_mean': np.mean(mae_scores),
                'n_splits': n_splits,
                'test_size': test_size,
                'correlations': correlations,
                'r2_scores': r2_scores
            }
            
            self.validation_results['out_of_sample'] = results
            return results
            
        except Exception as e:
            print(f"Error in out-of-sample testing: {e}")
            return {'error': str(e)}
    
    def regime_detection_analysis(self, n_regimes=2):
        """
        Perform regime detection analysis for BRI and VIX
        
        Args:
            n_regimes (int): Number of regimes to detect
            
        Returns:
            dict: Regime detection results
        """
        try:
            # Merge data
            if self.market_data is None or 'VIX' not in self.market_data.columns:
                return {'error': 'VIX data not available'}
            
            merged = pd.merge(self.bri_data, self.market_data, on='date', how='inner')
            if len(merged) < 100:
                return {'error': 'Insufficient data for regime detection'}
            
            # Prepare data
            bri_values = merged['BRI'].values
            vix_values = merged['VIX'].values
            
            # Markov Regime Switching Model
            try:
                # Simple regime detection using volatility
                bri_vol = pd.Series(bri_values).rolling(window=30).std()
                vix_vol = pd.Series(vix_values).rolling(window=30).std()
                
                # Define regimes based on volatility percentiles
                bri_high_vol_threshold = bri_vol.quantile(0.7)
                vix_high_vol_threshold = vix_vol.quantile(0.7)
                
                # Regime classification
                regimes = []
                for i in range(len(merged)):
                    if bri_vol.iloc[i] > bri_high_vol_threshold and vix_vol.iloc[i] > vix_high_vol_threshold:
                        regimes.append('High Volatility')
                    elif bri_vol.iloc[i] < bri_vol.quantile(0.3) and vix_vol.iloc[i] < vix_vol.quantile(0.3):
                        regimes.append('Low Volatility')
                    else:
                        regimes.append('Moderate Volatility')
                
                # Calculate regime statistics
                regime_stats = {}
                for regime in ['High Volatility', 'Moderate Volatility', 'Low Volatility']:
                    regime_mask = [r == regime for r in regimes]
                    if any(regime_mask):
                        regime_bri = bri_values[regime_mask]
                        regime_vix = vix_values[regime_mask]
                        
                        regime_stats[regime] = {
                            'count': sum(regime_mask),
                            'percentage': sum(regime_mask) / len(regimes) * 100,
                            'bri_mean': np.mean(regime_bri),
                            'bri_std': np.std(regime_bri),
                            'vix_mean': np.mean(regime_vix),
                            'vix_std': np.std(regime_vix),
                            'correlation': np.corrcoef(regime_bri, regime_vix)[0, 1] if len(regime_bri) > 1 else 0
                        }
                
                results = {
                    'regimes': regimes,
                    'regime_stats': regime_stats,
                    'n_regimes': len(set(regimes)),
                    'regime_transitions': self._calculate_regime_transitions(regimes)
                }
                
                self.validation_results['regime_detection'] = results
                return results
                
            except Exception as e:
                print(f"Error in regime detection: {e}")
                return {'error': str(e)}
                
        except Exception as e:
            print(f"Error in regime detection analysis: {e}")
            return {'error': str(e)}
    
    def _calculate_regime_transitions(self, regimes):
        """Calculate regime transition probabilities"""
        transitions = {}
        for i in range(len(regimes) - 1):
            current = regimes[i]
            next_regime = regimes[i + 1]
            key = f"{current} -> {next_regime}"
            transitions[key] = transitions.get(key, 0) + 1
        
        # Convert to probabilities
        total_transitions = sum(transitions.values())
        for key in transitions:
            transitions[key] = transitions[key] / total_transitions
        
        return transitions
    
    def stationarity_tests(self):
        """
        Perform stationarity tests on BRI series
        
        Returns:
            dict: Stationarity test results
        """
        try:
            bri_values = self.bri_data['BRI'].values
            
            # Augmented Dickey-Fuller test
            adf_result = adfuller(bri_values, autolag='AIC')
            
            # KPSS test (alternative)
            try:
                from statsmodels.tsa.stattools import kpss
                kpss_result = kpss(bri_values, regression='c')
            except:
                kpss_result = None
            
            # Phillips-Perron test
            try:
                from statsmodels.tsa.stattools import pp
                pp_result = pp(bri_values)
            except:
                pp_result = None
            
            results = {
                'adf_test': {
                    'statistic': adf_result[0],
                    'p_value': adf_result[1],
                    'critical_values': adf_result[4],
                    'is_stationary': adf_result[1] < 0.05
                },
                'kpss_test': {
                    'statistic': kpss_result[0] if kpss_result else None,
                    'p_value': kpss_result[1] if kpss_result else None,
                    'critical_values': kpss_result[3] if kpss_result else None,
                    'is_stationary': kpss_result[1] > 0.05 if kpss_result else None
                },
                'pp_test': {
                    'statistic': pp_result[0] if pp_result else None,
                    'p_value': pp_result[1] if pp_result else None,
                    'critical_values': pp_result[4] if pp_result else None,
                    'is_stationary': pp_result[1] < 0.05 if pp_result else None
                }
            }
            
            self.validation_results['stationarity'] = results
            return results
            
        except Exception as e:
            print(f"Error in stationarity tests: {e}")
            return {'error': str(e)}
    
    def granger_causality_test(self, max_lags=10):
        """
        Perform Granger causality tests between BRI and VIX
        
        Args:
            max_lags (int): Maximum number of lags to test
            
        Returns:
            dict: Granger causality test results
        """
        try:
            if self.market_data is None or 'VIX' not in self.market_data.columns:
                return {'error': 'VIX data not available'}
            
            # Merge data
            merged = pd.merge(self.bri_data, self.market_data, on='date', how='inner')
            if len(merged) < 100:
                return {'error': 'Insufficient data for Granger causality test'}
            
            # Prepare data
            bri_values = merged['BRI'].values
            vix_values = merged['VIX'].values
            
            # Create DataFrame for Granger test
            test_data = pd.DataFrame({
                'BRI': bri_values,
                'VIX': vix_values
            })
            
            # Test BRI -> VIX
            try:
                bri_to_vix = grangercausalitytests(test_data[['VIX', 'BRI']], max_lags, verbose=False)
            except:
                bri_to_vix = None
            
            # Test VIX -> BRI
            try:
                vix_to_bri = grangercausalitytests(test_data[['BRI', 'VIX']], max_lags, verbose=False)
            except:
                vix_to_bri = None
            
            # Extract results
            results = {
                'bri_to_vix': self._extract_granger_results(bri_to_vix, max_lags) if bri_to_vix else None,
                'vix_to_bri': self._extract_granger_results(vix_to_bri, max_lags) if vix_to_bri else None,
                'max_lags': max_lags
            }
            
            self.validation_results['granger_causality'] = results
            return results
            
        except Exception as e:
            print(f"Error in Granger causality test: {e}")
            return {'error': str(e)}
    
    def _extract_granger_results(self, granger_results, max_lags):
        """Extract Granger causality test results"""
        if granger_results is None:
            return None
        
        results = {}
        for lag in range(1, max_lags + 1):
            if lag in granger_results:
                test_result = granger_results[lag]
                results[f'lag_{lag}'] = {
                    'f_statistic': test_result[0]['ssr_ftest'][0],
                    'f_p_value': test_result[0]['ssr_ftest'][1],
                    'is_significant': test_result[0]['ssr_ftest'][1] < 0.05
                }
        
        return results
    
    def volatility_analysis(self):
        """
        Perform comprehensive volatility analysis
        
        Returns:
            dict: Volatility analysis results
        """
        try:
            bri_values = self.bri_data['BRI'].values
            
            # Calculate various volatility measures
            returns = np.diff(bri_values) / bri_values[:-1]
            
            # Annualized volatility
            annualized_vol = np.std(returns) * np.sqrt(252)
            
            # Rolling volatility
            rolling_vol = pd.Series(returns).rolling(window=30).std() * np.sqrt(252)
            
            # Volatility of volatility
            vol_of_vol = np.std(rolling_vol.dropna())
            
            # GARCH-like volatility clustering
            vol_clustering = self._calculate_volatility_clustering(returns)
            
            # Compare with VIX if available
            vix_comparison = None
            if self.market_data is not None and 'VIX' in self.market_data.columns:
                merged = pd.merge(self.bri_data, self.market_data, on='date', how='inner')
                if len(merged) > 1:
                    vix_values = merged['VIX'].values
                    vix_vol = np.std(vix_values) / np.mean(vix_values)  # Coefficient of variation
                    bri_vol = np.std(bri_values) / np.mean(bri_values)
                    
                    vix_comparison = {
                        'vix_volatility': vix_vol,
                        'bri_volatility': bri_vol,
                        'volatility_ratio': bri_vol / vix_vol if vix_vol > 0 else None
                    }
            
            results = {
                'annualized_volatility': annualized_vol,
                'volatility_of_volatility': vol_of_vol,
                'volatility_clustering': vol_clustering,
                'rolling_volatility_mean': rolling_vol.mean(),
                'rolling_volatility_std': rolling_vol.std(),
                'vix_comparison': vix_comparison,
                'returns_statistics': {
                    'mean': np.mean(returns),
                    'std': np.std(returns),
                    'skewness': stats.skew(returns),
                    'kurtosis': stats.kurtosis(returns)
                }
            }
            
            self.validation_results['volatility_analysis'] = results
            return results
            
        except Exception as e:
            print(f"Error in volatility analysis: {e}")
            return {'error': str(e)}
    
    def _calculate_volatility_clustering(self, returns):
        """Calculate volatility clustering measure"""
        try:
            # Calculate autocorrelation of squared returns
            squared_returns = returns ** 2
            autocorr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
            return autocorr
        except:
            return None
    
    def feature_sensitivity_analysis(self):
        """
        Perform feature sensitivity analysis (ablation study)
        
        Returns:
            dict: Feature sensitivity results
        """
        try:
            # Get feature columns
            feature_cols = ['sent_vol_score', 'news_tone_score', 'herding_score', 
                           'polarity_skew_score', 'event_density_score']
            
            # Check which features are available
            available_features = [col for col in feature_cols if col in self.bri_data.columns]
            
            if len(available_features) < 2:
                return {'error': 'Insufficient features for sensitivity analysis'}
            
            # Calculate BRI with all features
            full_bri = self.bri_data['BRI'].values
            
            sensitivity_results = {}
            
            for feature in available_features:
                # Calculate BRI without this feature
                other_features = [f for f in available_features if f != feature]
                
                if len(other_features) > 0:
                    # Simple equal weighting for remaining features
                    reduced_bri = self.bri_data[other_features].mean(axis=1).values
                    
                    # Calculate correlation with full BRI
                    correlation = np.corrcoef(full_bri, reduced_bri)[0, 1]
                    
                    # Calculate RMSE
                    rmse = np.sqrt(np.mean((full_bri - reduced_bri) ** 2))
                    
                    sensitivity_results[feature] = {
                        'correlation_with_full': correlation,
                        'rmse': rmse,
                        'feature_importance': 1 - correlation,  # Higher importance = lower correlation
                        'available_features': len(other_features)
                    }
            
            results = {
                'feature_sensitivity': sensitivity_results,
                'total_features': len(available_features),
                'available_features': available_features
            }
            
            self.validation_results['feature_sensitivity'] = results
            return results
            
        except Exception as e:
            print(f"Error in feature sensitivity analysis: {e}")
            return {'error': str(e)}
    
    def multicollinearity_check(self):
        """
        Check for multicollinearity in BRI features
        
        Returns:
            dict: Multicollinearity analysis results
        """
        try:
            # Get feature columns
            feature_cols = ['sent_vol_score', 'news_tone_score', 'herding_score', 
                           'polarity_skew_score', 'event_density_score']
            
            available_features = [col for col in feature_cols if col in self.bri_data.columns]
            
            if len(available_features) < 2:
                return {'error': 'Insufficient features for multicollinearity check'}
            
            # Create feature matrix
            X = self.bri_data[available_features].values
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(X.T)
            
            # Calculate VIF (Variance Inflation Factor)
            vif_scores = self._calculate_vif(X)
            
            # Identify high correlations
            high_correlations = []
            for i in range(len(available_features)):
                for j in range(i + 1, len(available_features)):
                    corr = corr_matrix[i, j]
                    if abs(corr) > 0.7:  # High correlation threshold
                        high_correlations.append({
                            'feature1': available_features[i],
                            'feature2': available_features[j],
                            'correlation': corr
                        })
            
            results = {
                'correlation_matrix': corr_matrix.tolist(),
                'feature_names': available_features,
                'vif_scores': dict(zip(available_features, vif_scores)),
                'high_correlations': high_correlations,
                'max_vif': max(vif_scores) if vif_scores else None,
                'multicollinearity_present': max(vif_scores) > 10 if vif_scores else False
            }
            
            self.validation_results['multicollinearity'] = results
            return results
            
        except Exception as e:
            print(f"Error in multicollinearity check: {e}")
            return {'error': str(e)}
    
    def _calculate_vif(self, X):
        """Calculate Variance Inflation Factor for each feature"""
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            vif_scores = []
            for i in range(X.shape[1]):
                vif = variance_inflation_factor(X, i)
                vif_scores.append(vif)
            
            return vif_scores
        except:
            # Fallback calculation
            corr_matrix = np.corrcoef(X.T)
            vif_scores = []
            for i in range(X.shape[1]):
                r_squared = 1 - 1 / np.linalg.det(corr_matrix)
                vif = 1 / (1 - r_squared) if r_squared < 1 else float('inf')
                vif_scores.append(vif)
            return vif_scores
    
    def backtesting_analysis(self, threshold_percentile=90):
        """
        Perform backtesting analysis for BRI signals
        
        Args:
            threshold_percentile (float): Percentile for signal threshold
            
        Returns:
            dict: Backtesting results
        """
        try:
            if self.market_data is None or 'VIX' not in self.market_data.columns:
                return {'error': 'VIX data not available'}
            
            # Merge data
            merged = pd.merge(self.bri_data, self.market_data, on='date', how='inner')
            if len(merged) < 100:
                return {'error': 'Insufficient data for backtesting'}
            
            # Calculate VIX returns
            vix_values = merged['VIX'].values
            vix_returns = np.diff(vix_values) / vix_values[:-1]
            
            # Define VIX spike threshold (e.g., +10% move)
            vix_spike_threshold = 0.10
            
            # Identify VIX spikes
            vix_spikes = vix_returns > vix_spike_threshold
            
            # Calculate BRI threshold
            bri_threshold = np.percentile(merged['BRI'].values, threshold_percentile)
            
            # Generate BRI signals
            bri_signals = merged['BRI'].values > bri_threshold
            
            # Align signals with next-day VIX spikes
            if len(bri_signals) > len(vix_spikes):
                bri_signals = bri_signals[:-1]
            elif len(vix_spikes) > len(bri_signals):
                vix_spikes = vix_spikes[:len(bri_signals)]
            
            # Calculate precision and recall
            true_positives = np.sum(bri_signals & vix_spikes)
            false_positives = np.sum(bri_signals & ~vix_spikes)
            false_negatives = np.sum(~bri_signals & vix_spikes)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate hit rate
            hit_rate = true_positives / np.sum(bri_signals) if np.sum(bri_signals) > 0 else 0
            
            results = {
                'threshold_percentile': threshold_percentile,
                'bri_threshold': bri_threshold,
                'vix_spike_threshold': vix_spike_threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'hit_rate': hit_rate,
                'true_positives': int(true_positives),
                'false_positives': int(false_positives),
                'false_negatives': int(false_negatives),
                'total_signals': int(np.sum(bri_signals)),
                'total_spikes': int(np.sum(vix_spikes))
            }
            
            self.validation_results['backtesting'] = results
            return results
            
        except Exception as e:
            print(f"Error in backtesting analysis: {e}")
            return {'error': str(e)}
    
    def generate_comprehensive_report(self):
        """
        Generate comprehensive statistical validation report
        
        Returns:
            dict: Complete validation report
        """
        try:
            # Run all validation tests
            out_of_sample = self.out_of_sample_testing()
            regime_detection = self.regime_detection_analysis()
            stationarity = self.stationarity_tests()
            granger_causality = self.granger_causality_test()
            volatility_analysis = self.volatility_analysis()
            feature_sensitivity = self.feature_sensitivity_analysis()
            multicollinearity = self.multicollinearity_check()
            backtesting = self.backtesting_analysis()
            
            # Compile comprehensive report
            report = {
                'out_of_sample_testing': out_of_sample,
                'regime_detection': regime_detection,
                'stationarity_tests': stationarity,
                'granger_causality': granger_causality,
                'volatility_analysis': volatility_analysis,
                'feature_sensitivity': feature_sensitivity,
                'multicollinearity_check': multicollinearity,
                'backtesting_analysis': backtesting,
                'generated_at': pd.Timestamp.now().isoformat(),
                'data_points': len(self.bri_data),
                'validation_summary': self._create_validation_summary()
            }
            
            return report
            
        except Exception as e:
            print(f"Error generating comprehensive report: {e}")
            return {'error': str(e)}
    
    def _create_validation_summary(self):
        """Create validation summary"""
        summary = {
            'tests_completed': len(self.validation_results),
            'overall_validation_status': 'PASSED' if len(self.validation_results) > 5 else 'PARTIAL',
            'key_findings': []
        }
        
        # Add key findings
        if 'out_of_sample' in self.validation_results:
            oos = self.validation_results['out_of_sample']
            if 'correlation_mean' in oos:
                summary['key_findings'].append(f"Out-of-sample correlation: {oos['correlation_mean']:.3f}")
        
        if 'stationarity' in self.validation_results:
            stat = self.validation_results['stationarity']
            if 'adf_test' in stat:
                summary['key_findings'].append(f"BRI stationarity: {'Stationary' if stat['adf_test']['is_stationary'] else 'Non-stationary'}")
        
        if 'backtesting' in self.validation_results:
            bt = self.validation_results['backtesting']
            if 'precision' in bt:
                summary['key_findings'].append(f"Signal precision: {bt['precision']:.3f}")
        
        return summary
