"""
Validation module for BRI pipeline.
Implements econometric validation including GARCH, VAR, and forecasting tests.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from arch import arch_model
import logging

from utils import calculate_returns, calculate_realized_volatility

class ValidationEngine:
    """Main validation class for BRI pipeline."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Validation parameters
        self.test_size = self.config.get('test_size', 0.2)
        self.random_state = self.config.get('random_state', 42)
        self.forecast_horizon = self.config.get('forecast_horizon', 5)
        
        # Set random seed
        np.random.seed(self.random_state)
    
    def prepare_validation_data(self, bri_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for validation by merging and aligning time series."""
        self.logger.info("Preparing validation data")
        
        # Ensure both dataframes have 'date' column
        if 'Date' in market_df.columns:
            market_df = market_df.rename(columns={'Date': 'date'})
        
        # Normalize datetime columns to remove timezone info
        bri_df['date'] = pd.to_datetime(bri_df['date']).dt.tz_localize(None)
        market_df['date'] = pd.to_datetime(market_df['date']).dt.tz_localize(None)
        
        # Merge BRI and market data
        merged_df = pd.merge(bri_df, market_df, on='date', how='inner')
        
        # Sort by date
        merged_df = merged_df.sort_values('date').reset_index(drop=True)
        
        # Calculate additional market variables if not present
        if 'returns' not in merged_df.columns and '^GSPC_Close' in merged_df.columns:
            merged_df['returns'] = calculate_returns(merged_df['^GSPC_Close'])
        
        if 'realized_vol' not in merged_df.columns and 'returns' in merged_df.columns:
            merged_df['realized_vol'] = calculate_realized_volatility(merged_df['returns'])
        
        # Create lagged variables
        merged_df['BRI_lag1'] = merged_df['BRI_t'].shift(1)
        merged_df['VIX_lag1'] = merged_df['^VIX_Close'].shift(1) if '^VIX_Close' in merged_df.columns else None
        merged_df['returns_lag1'] = merged_df['returns'].shift(1)
        
        # Remove rows with missing values
        merged_df = merged_df.dropna()
        
        self.logger.info(f"Prepared validation data with {len(merged_df)} observations")
        
        return merged_df
    
    def test_stationarity(self, series: pd.Series, name: str = "Series") -> Dict[str, float]:
        """Test stationarity using Augmented Dickey-Fuller test."""
        self.logger.info(f"Testing stationarity of {name}")
        
        # Check if series is empty or has insufficient data
        series_clean = series.dropna()
        if len(series_clean) < 10:
            self.logger.warning(f"Insufficient data for stationarity test of {name}: {len(series_clean)} observations")
            return {
                'adf_statistic': np.nan,
                'p_value': np.nan,
                'critical_values': {},
                'is_stationary': False,
                'error': 'Insufficient data'
            }
        
        try:
            adf_result = adfuller(series_clean)
            
            return {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05
            }
        except Exception as e:
            self.logger.error(f"Error in stationarity test for {name}: {e}")
            return {
                'adf_statistic': np.nan,
                'p_value': np.nan,
                'critical_values': {},
                'is_stationary': False,
                'error': str(e)
            }
    
    def compute_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute correlation matrix for key variables."""
        self.logger.info("Computing correlation matrix")
        
        # Select key variables for correlation
        corr_vars = ['BRI_t', '^VIX_Close', 'returns', 'realized_vol']
        available_vars = [var for var in corr_vars if var in df.columns]
        
        if not available_vars:
            self.logger.warning("No variables available for correlation analysis")
            return pd.DataFrame()
        
        corr_matrix = df[available_vars].corr()
        
        return corr_matrix
    
    def granger_causality_test(self, df: pd.DataFrame, x_col: str, y_col: str, 
                              max_lags: int = 4) -> Dict[str, float]:
        """Perform Granger causality test between two variables."""
        self.logger.info(f"Testing Granger causality: {x_col} -> {y_col}")
        
        # Prepare data for Granger test
        data = df[[x_col, y_col]].dropna()
        
        if len(data) < max_lags + 1:
            self.logger.warning("Insufficient data for Granger causality test")
            return {}
        
        try:
            # Perform Granger causality test
            granger_result = grangercausalitytests(data, maxlag=max_lags, verbose=False)
            
            # Extract p-values for each lag
            p_values = {}
            for lag in range(1, max_lags + 1):
                if lag in granger_result:
                    p_values[f'lag_{lag}'] = granger_result[lag][0]['ssr_ftest'][1]
            
            # Find minimum p-value across lags
            min_p_value = min(p_values.values()) if p_values else 1.0
            
            return {
                'min_p_value': min_p_value,
                'is_significant': min_p_value < 0.05,
                'p_values_by_lag': p_values
            }
            
        except Exception as e:
            self.logger.error(f"Error in Granger causality test: {e}")
            return {}
    
    def run_ols_regression(self, df: pd.DataFrame, dependent_var: str, 
                          independent_vars: List[str]) -> Dict:
        """Run OLS regression analysis."""
        self.logger.info(f"Running OLS regression: {dependent_var} ~ {independent_vars}")
        
        # Prepare data
        y = df[dependent_var].dropna()
        X = df[independent_vars].dropna()
        
        # Align data
        common_index = y.index.intersection(X.index)
        y = y.loc[common_index]
        X = X.loc[common_index]
        
        if len(y) == 0 or len(X) == 0:
            self.logger.warning("No data available for OLS regression")
            return {}
        
        # Add constant term
        X = sm.add_constant(X)
        
        try:
            # Fit OLS model
            model = sm.OLS(y, X).fit()
            
            # Extract results
            results = {
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'f_p_value': model.f_pvalue,
                'coefficients': model.params.to_dict(),
                'p_values': model.pvalues.to_dict(),
                'std_errors': model.bse.to_dict(),
                'conf_int': model.conf_int().to_dict(),
                'n_observations': model.nobs,
                'aic': model.aic,
                'bic': model.bic
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in OLS regression: {e}")
            return {}
    
    def run_garch_model(self, returns: pd.Series, exog_vars: Optional[pd.DataFrame] = None) -> Dict:
        """Run GARCH(1,1) model on returns."""
        self.logger.info("Running GARCH(1,1) model")
        
        # Remove missing values
        returns = returns.dropna()
        
        if len(returns) < 50:
            self.logger.warning("Insufficient data for GARCH model")
            return {}
        
        try:
            # Fit GARCH model
            if exog_vars is not None:
                # Align exogenous variables
                exog_vars = exog_vars.loc[returns.index].dropna()
                common_index = returns.index.intersection(exog_vars.index)
                returns = returns.loc[common_index]
                exog_vars = exog_vars.loc[common_index]
                
                model = arch_model(returns, vol='Garch', p=1, q=1, x=exog_vars)
            else:
                model = arch_model(returns, vol='Garch', p=1, q=1)
            
            # Fit the model
            garch_fit = model.fit(disp='off')
            
            # Extract results
            results = {
                'log_likelihood': garch_fit.loglikelihood,
                'aic': garch_fit.aic,
                'bic': garch_fit.bic,
                'parameters': garch_fit.params.to_dict(),
                'p_values': garch_fit.pvalues.to_dict(),
                'std_errors': garch_fit.std_errors.to_dict(),
                'n_observations': garch_fit.nobs
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in GARCH model: {e}")
            return {}
    
    def run_var_model(self, df: pd.DataFrame, variables: List[str], 
                     max_lags: int = 4) -> Dict:
        """Run Vector Autoregression (VAR) model."""
        self.logger.info(f"Running VAR model with variables: {variables}")
        
        # Prepare data
        var_data = df[variables].dropna()
        
        if len(var_data) < max_lags + 10:
            self.logger.warning("Insufficient data for VAR model")
            return {}
        
        try:
            # Fit VAR model
            var_model = VAR(var_data)
            var_fit = var_model.fit(maxlags=max_lags, ic='aic')
            
            # Extract results
            results = {
                'selected_lags': var_fit.k_ar,
                'aic': var_fit.aic,
                'bic': var_fit.bic,
                'hqic': var_fit.hqic,
                'log_likelihood': var_fit.llf,
                'n_observations': var_fit.nobs
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in VAR model: {e}")
            return {}
    
    def rolling_forecast_evaluation(self, df: pd.DataFrame, target_var: str, 
                                  features: List[str], window_size: int = 252) -> Dict:
        """Perform rolling window forecast evaluation."""
        self.logger.info(f"Running rolling forecast evaluation for {target_var}")
        
        # Prepare data
        data = df[[target_var] + features].dropna()
        
        if len(data) < window_size + self.forecast_horizon:
            self.logger.warning("Insufficient data for rolling forecast evaluation")
            return {}
        
        # Initialize results
        forecasts = []
        actuals = []
        
        # Rolling window
        for i in range(window_size, len(data) - self.forecast_horizon + 1):
            # Training data
            train_data = data.iloc[i-window_size:i]
            
            # Test data
            test_data = data.iloc[i:i+self.forecast_horizon]
            
            try:
                # Fit simple linear model
                X_train = train_data[features]
                y_train = train_data[target_var]
                X_test = test_data[features]
                y_test = test_data[target_var]
                
                # Add constant
                X_train = sm.add_constant(X_train)
                X_test = sm.add_constant(X_test)
                
                # Fit model
                model = sm.OLS(y_train, X_train).fit()
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Store results
                forecasts.extend(y_pred.tolist())
                actuals.extend(y_test.tolist())
                
            except Exception as e:
                self.logger.warning(f"Error in rolling forecast at window {i}: {e}")
                continue
        
        if not forecasts:
            self.logger.warning("No successful forecasts generated")
            return {}
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, forecasts))
        mae = mean_absolute_error(actuals, forecasts)
        mape = np.mean(np.abs((np.array(actuals) - np.array(forecasts)) / np.array(actuals))) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'n_forecasts': len(forecasts),
            'forecasts': forecasts,
            'actuals': actuals
        }
    
    def compare_models_with_without_bri(self, df: pd.DataFrame, target_var: str, 
                                      base_features: List[str]) -> Dict:
        """Compare model performance with and without BRI."""
        self.logger.info("Comparing models with and without BRI")
        
        # Model without BRI
        features_without_bri = [f for f in base_features if 'BRI' not in f]
        results_without = self.rolling_forecast_evaluation(df, target_var, features_without_bri)
        
        # Model with BRI
        results_with = self.rolling_forecast_evaluation(df, target_var, base_features)
        
        if not results_without or not results_with:
            self.logger.warning("Could not generate forecasts for comparison")
            return {}
        
        # Calculate improvement
        rmse_improvement = (results_without['rmse'] - results_with['rmse']) / results_without['rmse'] * 100
        mae_improvement = (results_without['mae'] - results_with['mae']) / results_without['mae'] * 100
        
        return {
            'without_bri': results_without,
            'with_bri': results_with,
            'rmse_improvement_pct': rmse_improvement,
            'mae_improvement_pct': mae_improvement,
            'bri_adds_value': rmse_improvement > 0
        }
    
    def generate_validation_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive validation report."""
        self.logger.info("Generating validation report")
        
        report = {}
        
        # Stationarity tests
        report['stationarity'] = {}
        for var in ['BRI_t', '^VIX_Close', 'returns', 'realized_vol']:
            if var in df.columns:
                report['stationarity'][var] = self.test_stationarity(df[var], var)
        
        # Correlations
        report['correlations'] = self.compute_correlations(df)
        
        # Granger causality tests
        report['granger_causality'] = {}
        if 'BRI_t' in df.columns and '^VIX_Close' in df.columns:
            report['granger_causality']['BRI_to_VIX'] = self.granger_causality_test(
                df, 'BRI_t', '^VIX_Close'
            )
            report['granger_causality']['VIX_to_BRI'] = self.granger_causality_test(
                df, '^VIX_Close', 'BRI_t'
            )
        
        if 'BRI_t' in df.columns and 'realized_vol' in df.columns:
            report['granger_causality']['BRI_to_RV'] = self.granger_causality_test(
                df, 'BRI_t', 'realized_vol'
            )
            report['granger_causality']['RV_to_BRI'] = self.granger_causality_test(
                df, 'realized_vol', 'BRI_t'
            )
        
        # OLS regressions
        report['ols_regressions'] = {}
        
        # RV ~ VIX + BRI
        if all(var in df.columns for var in ['realized_vol', '^VIX_Close', 'BRI_lag1']):
            report['ols_regressions']['RV_vs_VIX_BRI'] = self.run_ols_regression(
                df, 'realized_vol', ['^VIX_Close', 'BRI_lag1']
            )
        
        # Returns ~ VIX + BRI
        if all(var in df.columns for var in ['returns', '^VIX_Close', 'BRI_lag1']):
            report['ols_regressions']['Returns_vs_VIX_BRI'] = self.run_ols_regression(
                df, 'returns', ['^VIX_Close', 'BRI_lag1']
            )
        
        # GARCH models
        report['garch_models'] = {}
        
        if 'returns' in df.columns:
            # GARCH without BRI
            report['garch_models']['without_bri'] = self.run_garch_model(df['returns'])
            
            # GARCH with BRI
            if 'BRI_lag1' in df.columns:
                report['garch_models']['with_bri'] = self.run_garch_model(
                    df['returns'], df[['BRI_lag1']]
                )
        
        # VAR model
        var_vars = ['BRI_t', '^VIX_Close', 'returns', 'realized_vol']
        available_var_vars = [var for var in var_vars if var in df.columns]
        if len(available_var_vars) >= 2:
            report['var_model'] = self.run_var_model(df, available_var_vars)
        
        # Forecast evaluation
        if 'realized_vol' in df.columns:
            base_features = ['^VIX_Close', 'BRI_lag1'] if 'BRI_lag1' in df.columns else ['^VIX_Close']
            report['forecast_evaluation'] = self.compare_models_with_without_bri(
                df, 'realized_vol', base_features
            )
        
        return report
    
    def save_validation_results(self, report: Dict, output_dir: str = "output"):
        """Save validation results to files."""
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save correlation matrix
        if 'correlations' in report and not report['correlations'].empty:
            report['correlations'].to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))
        
        # Save OLS regression results
        if 'ols_regressions' in report:
            for name, results in report['ols_regressions'].items():
                if results:
                    # Save coefficients
                    coef_df = pd.DataFrame({
                        'coefficient': results['coefficients'],
                        'std_error': results['std_errors'],
                        'p_value': results['p_values']
                    })
                    coef_df.to_csv(os.path.join(output_dir, f'ols_{name}_coefficients.csv'))
        
        # Save full report as JSON
        with open(os.path.join(output_dir, 'validation_report.json'), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Validation results saved to {output_dir}")
