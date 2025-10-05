"""
Research-level validation module for Behavioral Risk Index.
Includes crisis period analysis, advanced econometric tests, and regulatory insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical and econometric imports
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from utils import setup_logging, safe_divide

class ResearchValidationEngine:
    """Advanced validation engine for research-level BRI analysis."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = setup_logging('INFO')
        
        # Crisis periods for analysis
        self.crisis_periods = {
            '2008_financial_crisis': ('2007-12-01', '2009-06-30'),
            '2010_flash_crash': ('2010-05-01', '2010-05-31'),
            '2011_debt_ceiling': ('2011-07-01', '2011-08-31'),
            '2015_china_devaluation': ('2015-08-01', '2015-08-31'),
            '2016_brexit': ('2016-06-01', '2016-07-31'),
            '2018_trade_war': ('2018-03-01', '2018-12-31'),
            '2020_covid': ('2020-02-01', '2020-04-30'),
            '2022_inflation': ('2022-01-01', '2022-12-31'),
            '2023_banking_crisis': ('2023-03-01', '2023-05-31')
        }
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def comprehensive_analysis(self, bri_data: pd.DataFrame, market_data: pd.DataFrame) -> Dict:
        """Run comprehensive research-level analysis."""
        self.logger.info("Starting comprehensive research-level analysis")
        
        # Prepare data
        merged_data = self._prepare_research_data(bri_data, market_data)
        
        if merged_data.empty:
            self.logger.error("No data available for analysis")
            return {}
        
        results = {}
        
        # 1. Crisis Period Analysis
        self.logger.info("Running crisis period analysis")
        results['crisis_analysis'] = self._analyze_crisis_periods(merged_data)
        
        # 2. Advanced Econometric Tests
        self.logger.info("Running advanced econometric tests")
        results['econometric_tests'] = self._run_advanced_econometric_tests(merged_data)
        
        # 3. Forecasting Analysis
        self.logger.info("Running forecasting analysis")
        results['forecasting'] = self._run_forecasting_analysis(merged_data)
        
        # 4. Regime Switching Analysis
        self.logger.info("Running regime switching analysis")
        results['regime_analysis'] = self._run_regime_switching_analysis(merged_data)
        
        # 5. Cross-Asset Analysis
        self.logger.info("Running cross-asset analysis")
        results['cross_asset'] = self._run_cross_asset_analysis(merged_data)
        
        # 6. Regulatory Insights
        self.logger.info("Generating regulatory insights")
        results['regulatory_insights'] = self._generate_regulatory_insights(merged_data, results)
        
        # 7. Generate Research Plots
        self.logger.info("Generating research plots")
        self._generate_research_plots(merged_data, results)
        
        return results
    
    def _prepare_research_data(self, bri_data: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for research analysis."""
        # Ensure both dataframes have 'date' column
        if 'Date' in market_data.columns:
            market_data = market_data.rename(columns={'Date': 'date'})
        
        # Normalize datetime columns
        bri_data['date'] = pd.to_datetime(bri_data['date']).dt.tz_localize(None)
        market_data['date'] = pd.to_datetime(market_data['date']).dt.tz_localize(None)
        
        # Merge data
        merged_data = pd.merge(bri_data, market_data, on='date', how='inner')
        merged_data = merged_data.sort_values('date').reset_index(drop=True)
        
        # Create additional variables
        merged_data = self._create_research_variables(merged_data)
        
        return merged_data
    
    def _create_research_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional research variables."""
        # BRI changes and lags
        if 'BRI_t' in df.columns:
            df['BRI_change'] = df['BRI_t'].pct_change()
            df['BRI_lag1'] = df['BRI_t'].shift(1)
            df['BRI_lag2'] = df['BRI_t'].shift(2)
            df['BRI_lag5'] = df['BRI_t'].shift(5)
            df['BRI_ma5'] = df['BRI_t'].rolling(window=5).mean()
            df['BRI_ma20'] = df['BRI_t'].rolling(window=20).mean()
        
        # VIX changes and lags
        if '^VIX_Close' in df.columns:
            df['VIX_change'] = df['^VIX_Close'].pct_change()
            df['VIX_lag1'] = df['^VIX_Close'].shift(1)
            df['VIX_spike'] = (df['^VIX_Close'] > df['^VIX_Close'].rolling(window=20).mean() + 2 * df['^VIX_Close'].rolling(window=20).std()).astype(int)
        
        # Market returns and volatility
        if '^GSPC_returns' in df.columns:
            df['market_returns'] = df['^GSPC_returns']
            df['market_vol'] = df['^GSPC_returns'].rolling(window=20).std() * np.sqrt(252)
            df['market_vol_lag1'] = df['market_vol'].shift(1)
        
        # Realized volatility
        if '^GSPC_realized_vol' in df.columns:
            df['realized_vol'] = df['^GSPC_realized_vol']
            df['realized_vol_lag1'] = df['realized_vol'].shift(1)
        
        # Crisis indicators
        df['crisis_indicator'] = 0
        for crisis_name, (start_date, end_date) in self.crisis_periods.items():
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            df.loc[mask, 'crisis_indicator'] = 1
        
        return df
    
    def _analyze_crisis_periods(self, df: pd.DataFrame) -> Dict:
        """Analyze BRI behavior during crisis periods."""
        crisis_results = {}
        
        for crisis_name, (start_date, end_date) in self.crisis_periods.items():
            crisis_data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            if not crisis_data.empty and 'BRI_t' in crisis_data.columns:
                crisis_results[crisis_name] = {
                    'period': f"{start_date} to {end_date}",
                    'n_observations': len(crisis_data),
                    'bri_mean': crisis_data['BRI_t'].mean(),
                    'bri_std': crisis_data['BRI_t'].std(),
                    'bri_max': crisis_data['BRI_t'].max(),
                    'bri_min': crisis_data['BRI_t'].min(),
                    'bri_spikes': (crisis_data['BRI_t'] > crisis_data['BRI_t'].quantile(0.9)).sum(),
                    'vix_mean': crisis_data['^VIX_Close'].mean() if '^VIX_Close' in crisis_data.columns else np.nan,
                    'vix_max': crisis_data['^VIX_Close'].max() if '^VIX_Close' in crisis_data.columns else np.nan,
                    'market_return': crisis_data['market_returns'].sum() if 'market_returns' in crisis_data.columns else np.nan,
                    'volatility_mean': crisis_data['realized_vol'].mean() if 'realized_vol' in crisis_data.columns else np.nan
                }
        
        return crisis_results
    
    def _run_advanced_econometric_tests(self, df: pd.DataFrame) -> Dict:
        """Run advanced econometric tests."""
        results = {}
        
        # Stationarity tests
        results['stationarity'] = self._test_stationarity_advanced(df)
        
        # Cointegration tests
        results['cointegration'] = self._test_cointegration(df)
        
        # Granger causality tests
        results['granger_causality'] = self._test_granger_causality_advanced(df)
        
        # Structural break tests
        results['structural_breaks'] = self._test_structural_breaks(df)
        
        # Heteroscedasticity tests
        results['heteroscedasticity'] = self._test_heteroscedasticity(df)
        
        return results
    
    def _test_stationarity_advanced(self, df: pd.DataFrame) -> Dict:
        """Advanced stationarity tests."""
        results = {}
        
        variables = ['BRI_t', '^VIX_Close', 'market_returns', 'realized_vol']
        
        for var in variables:
            if var in df.columns:
                series = df[var].dropna()
                if len(series) > 10:
                    # ADF test
                    adf_result = adfuller(series)
                    
                    # KPSS test (if available)
                    try:
                        from statsmodels.tsa.stattools import kpss
                        kpss_result = kpss(series, regression='c')
                        kpss_stat = kpss_result[0]
                        kpss_pvalue = kpss_result[1]
                    except:
                        kpss_stat = np.nan
                        kpss_pvalue = np.nan
                    
                    results[var] = {
                        'adf_statistic': adf_result[0],
                        'adf_pvalue': adf_result[1],
                        'adf_critical_values': adf_result[4],
                        'adf_stationary': adf_result[1] < 0.05,
                        'kpss_statistic': kpss_stat,
                        'kpss_pvalue': kpss_pvalue,
                        'kpss_stationary': kpss_pvalue > 0.05 if not np.isnan(kpss_pvalue) else np.nan
                    }
        
        return results
    
    def _test_cointegration(self, df: pd.DataFrame) -> Dict:
        """Test for cointegration between BRI and market variables."""
        results = {}
        
        if 'BRI_t' in df.columns and '^VIX_Close' in df.columns:
            try:
                from statsmodels.tsa.stattools import coint
                
                bri_series = df['BRI_t'].dropna()
                vix_series = df['^VIX_Close'].dropna()
                
                # Align series
                common_index = bri_series.index.intersection(vix_series.index)
                bri_aligned = bri_series.loc[common_index]
                vix_aligned = vix_series.loc[common_index]
                
                if len(bri_aligned) > 10:
                    coint_result = coint(bri_aligned, vix_aligned)
                    
                    results['BRI_VIX'] = {
                        'coint_statistic': coint_result[0],
                        'pvalue': coint_result[1],
                        'critical_values': coint_result[2],
                        'cointegrated': coint_result[1] < 0.05
                    }
            except Exception as e:
                self.logger.error(f"Error in cointegration test: {e}")
                results['BRI_VIX'] = {'error': str(e)}
        
        return results
    
    def _test_granger_causality_advanced(self, df: pd.DataFrame) -> Dict:
        """Advanced Granger causality tests."""
        results = {}
        
        # Test BRI -> VIX
        if 'BRI_t' in df.columns and '^VIX_Close' in df.columns:
            try:
                test_data = df[['BRI_t', '^VIX_Close']].dropna()
                if len(test_data) > 20:
                    gc_result = grangercausalitytests(test_data, maxlag=5, verbose=False)
                    
                    results['BRI_to_VIX'] = {}
                    for lag in range(1, 6):
                        if lag in gc_result:
                            f_stat = gc_result[lag][0]['ssr_ftest'][0]
                            f_pvalue = gc_result[lag][0]['ssr_ftest'][1]
                            results['BRI_to_VIX'][f'lag_{lag}'] = {
                                'f_statistic': f_stat,
                                'pvalue': f_pvalue,
                                'significant': f_pvalue < 0.05
                            }
            except Exception as e:
                self.logger.error(f"Error in Granger causality test: {e}")
                results['BRI_to_VIX'] = {'error': str(e)}
        
        return results
    
    def _test_structural_breaks(self, df: pd.DataFrame) -> Dict:
        """Test for structural breaks in the relationship."""
        results = {}
        
        if 'BRI_t' in df.columns and '^VIX_Close' in df.columns:
            try:
                from statsmodels.tsa.stattools import cusum_squares
                
                # Test for structural breaks in BRI
                bri_series = df['BRI_t'].dropna()
                if len(bri_series) > 20:
                    cusum_result = cusum_squares(bri_series)
                    results['BRI_structural_breaks'] = {
                        'cusum_statistic': cusum_result[0],
                        'pvalue': cusum_result[1],
                        'has_breaks': cusum_result[1] < 0.05
                    }
            except Exception as e:
                self.logger.error(f"Error in structural break test: {e}")
                results['BRI_structural_breaks'] = {'error': str(e)}
        
        return results
    
    def _test_heteroscedasticity(self, df: pd.DataFrame) -> Dict:
        """Test for heteroscedasticity."""
        results = {}
        
        if 'BRI_t' in df.columns and '^VIX_Close' in df.columns:
            try:
                # Simple regression to test residuals
                test_data = df[['BRI_t', '^VIX_Close']].dropna()
                if len(test_data) > 20:
                    X = test_data[['^VIX_Close']].values
                    y = test_data['BRI_t'].values
                    
                    # Fit linear regression
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    residuals = y - y_pred
                    
                    # Breusch-Pagan test
                    try:
                        from statsmodels.stats.diagnostic import het_breuschpagan
                        bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X)
                        
                        results['heteroscedasticity'] = {
                            'breusch_pagan_stat': bp_stat,
                            'breusch_pagan_pvalue': bp_pvalue,
                            'has_heteroscedasticity': bp_pvalue < 0.05
                        }
                    except:
                        results['heteroscedasticity'] = {'error': 'Breusch-Pagan test failed'}
            except Exception as e:
                self.logger.error(f"Error in heteroscedasticity test: {e}")
                results['heteroscedasticity'] = {'error': str(e)}
        
        return results
    
    def _run_forecasting_analysis(self, df: pd.DataFrame) -> Dict:
        """Run comprehensive forecasting analysis."""
        results = {}
        
        # GARCH models
        results['garch_models'] = self._run_garch_analysis(df)
        
        # VAR models
        results['var_models'] = self._run_var_analysis(df)
        
        # Machine learning models
        results['ml_models'] = self._run_ml_forecasting(df)
        
        # Model comparison
        results['model_comparison'] = self._compare_forecasting_models(df)
        
        return results
    
    def _run_garch_analysis(self, df: pd.DataFrame) -> Dict:
        """Run GARCH analysis."""
        results = {}
        
        if 'market_returns' in df.columns:
            try:
                returns = df['market_returns'].dropna() * 100  # Convert to percentage
                
                if len(returns) > 50:
                    # GARCH(1,1) without BRI
                    garch_basic = arch_model(returns, vol='Garch', p=1, q=1)
                    garch_basic_fit = garch_basic.fit(disp='off')
                    
                    results['garch_basic'] = {
                        'aic': garch_basic_fit.aic,
                        'bic': garch_basic_fit.bic,
                        'loglikelihood': garch_basic_fit.loglikelihood,
                        'params': garch_basic_fit.params.to_dict()
                    }
                    
                    # GARCH(1,1) with BRI as exogenous variable
                    if 'BRI_t' in df.columns:
                        bri_aligned = df['BRI_t'].reindex(returns.index).fillna(method='ffill')
                        garch_with_bri = arch_model(returns, vol='Garch', p=1, q=1, x=bri_aligned)
                        garch_with_bri_fit = garch_with_bri.fit(disp='off')
                        
                        results['garch_with_bri'] = {
                            'aic': garch_with_bri_fit.aic,
                            'bic': garch_with_bri_fit.bic,
                            'loglikelihood': garch_with_bri_fit.loglikelihood,
                            'params': garch_with_bri_fit.params.to_dict(),
                            'bri_improvement': garch_basic_fit.aic - garch_with_bri_fit.aic
                        }
            except Exception as e:
                self.logger.error(f"Error in GARCH analysis: {e}")
                results['error'] = str(e)
        
        return results
    
    def _run_var_analysis(self, df: pd.DataFrame) -> Dict:
        """Run VAR analysis."""
        results = {}
        
        var_columns = ['BRI_t', '^VIX_Close', 'market_returns', 'realized_vol']
        available_columns = [col for col in var_columns if col in df.columns]
        
        if len(available_columns) >= 2:
            try:
                var_data = df[available_columns].dropna()
                
                if len(var_data) > 30:
                    var_model = VAR(var_data)
                    var_fit = var_model.fit(maxlags=5, ic='aic')
                    
                    results['var_model'] = {
                        'aic': var_fit.aic,
                        'bic': var_fit.bic,
                        'hqic': var_fit.hqic,
                        'selected_lags': var_fit.k_ar,
                        'variables': available_columns
                    }
                    
                    # Impulse response analysis
                    if 'BRI_t' in available_columns:
                        try:
                            irf = var_fit.irf(10)
                            results['impulse_response'] = {
                                'bri_shock_to_vix': irf.irfs[:, available_columns.index('BRI_t'), available_columns.index('^VIX_Close')].tolist() if '^VIX_Close' in available_columns else None,
                                'vix_shock_to_bri': irf.irfs[:, available_columns.index('^VIX_Close'), available_columns.index('BRI_t')].tolist() if '^VIX_Close' in available_columns else None
                            }
                        except:
                            results['impulse_response'] = {'error': 'Could not compute impulse response'}
            except Exception as e:
                self.logger.error(f"Error in VAR analysis: {e}")
                results['error'] = str(e)
        
        return results
    
    def _run_ml_forecasting(self, df: pd.DataFrame) -> Dict:
        """Run machine learning forecasting models."""
        results = {}
        
        if 'BRI_t' in df.columns and '^VIX_Close' in df.columns:
            try:
                # Prepare features
                features = ['BRI_t', '^VIX_Close', 'market_returns', 'realized_vol']
                available_features = [col for col in features if col in df.columns]
                
                # Create lagged features
                ml_data = df[available_features].copy()
                for col in available_features:
                    for lag in range(1, 6):
                        ml_data[f'{col}_lag{lag}'] = ml_data[col].shift(lag)
                
                ml_data = ml_data.dropna()
                
                if len(ml_data) > 50:
                    # Prepare training data
                    X = ml_data.drop(['BRI_t'], axis=1)
                    y = ml_data['BRI_t']
                    
                    # Split data
                    split_idx = int(len(ml_data) * 0.8)
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    # Random Forest
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf_model.fit(X_train, y_train)
                    rf_pred = rf_model.predict(X_test)
                    
                    results['random_forest'] = {
                        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
                        'mae': mean_absolute_error(y_test, rf_pred),
                        'feature_importance': dict(zip(X.columns, rf_model.feature_importances_))
                    }
            except Exception as e:
                self.logger.error(f"Error in ML forecasting: {e}")
                results['error'] = str(e)
        
        return results
    
    def _compare_forecasting_models(self, df: pd.DataFrame) -> Dict:
        """Compare different forecasting models."""
        results = {}
        
        # This would implement model comparison logic
        # For now, return a placeholder
        results['comparison'] = {
            'note': 'Model comparison implementation needed',
            'models_tested': ['GARCH', 'VAR', 'Random Forest']
        }
        
        return results
    
    def _run_regime_switching_analysis(self, df: pd.DataFrame) -> Dict:
        """Run regime switching analysis."""
        results = {}
        
        if 'BRI_t' in df.columns:
            try:
                bri_series = df['BRI_t'].dropna()
                
                if len(bri_series) > 50:
                    # Markov Regime Switching Model
                    model = MarkovRegression(bri_series, k_regimes=2, trend='c')
                    fitted_model = model.fit()
                    
                    results['markov_regime'] = {
                        'aic': fitted_model.aic,
                        'bic': fitted_model.bic,
                        'loglikelihood': fitted_model.llf,
                        'regime_probabilities': fitted_model.smoothed_marginal_probabilities[1].tolist(),
                        'regime_means': fitted_model.params[['const']].values.tolist()
                    }
            except Exception as e:
                self.logger.error(f"Error in regime switching analysis: {e}")
                results['error'] = str(e)
        
        return results
    
    def _run_cross_asset_analysis(self, df: pd.DataFrame) -> Dict:
        """Run cross-asset analysis."""
        results = {}
        
        # Analyze correlations across different assets
        return_columns = [col for col in df.columns if col.endswith('_returns')]
        
        if len(return_columns) > 1:
            corr_matrix = df[return_columns].corr()
            results['correlation_matrix'] = corr_matrix.to_dict()
            
            # Rolling correlations
            if '^GSPC_returns' in return_columns:
                sp500_returns = df['^GSPC_returns']
                rolling_corrs = {}
                
                for col in return_columns:
                    if col != '^GSPC_returns':
                        rolling_corr = sp500_returns.rolling(window=20).corr(df[col])
                        rolling_corrs[col] = {
                            'mean': rolling_corr.mean(),
                            'std': rolling_corr.std(),
                            'min': rolling_corr.min(),
                            'max': rolling_corr.max()
                        }
                
                results['rolling_correlations'] = rolling_corrs
        
        return results
    
    def _generate_regulatory_insights(self, df: pd.DataFrame, analysis_results: Dict) -> Dict:
        """Generate regulatory insights and policy recommendations."""
        insights = {
            'summary': {},
            'policy_recommendations': [],
            'risk_indicators': {},
            'monitoring_suggestions': []
        }
        
        # BRI behavior during crises
        if 'crisis_analysis' in analysis_results:
            crisis_data = analysis_results['crisis_analysis']
            high_bri_crises = [k for k, v in crisis_data.items() if v.get('bri_mean', 0) > 10]
            
            insights['summary']['high_bri_crises'] = high_bri_crises
            insights['summary']['crisis_count'] = len(crisis_data)
        
        # Policy recommendations
        insights['policy_recommendations'] = [
            "Monitor BRI spikes as early warning indicators for market stress",
            "Use BRI in conjunction with VIX for comprehensive risk assessment",
            "Consider BRI in regulatory stress testing frameworks",
            "Develop BRI-based circuit breakers for extreme market conditions",
            "Include BRI in central bank communication strategies"
        ]
        
        # Risk indicators
        if 'BRI_t' in df.columns:
            bri_series = df['BRI_t'].dropna()
            insights['risk_indicators'] = {
                'current_bri_level': bri_series.iloc[-1] if len(bri_series) > 0 else np.nan,
                'bri_percentile': bri_series.rank(pct=True).iloc[-1] if len(bri_series) > 0 else np.nan,
                'recent_spikes': (bri_series > bri_series.quantile(0.9)).sum(),
                'trend': 'increasing' if bri_series.iloc[-5:].mean() > bri_series.iloc[-20:-5].mean() else 'decreasing'
            }
        
        # Monitoring suggestions
        insights['monitoring_suggestions'] = [
            "Daily BRI monitoring with 90th percentile threshold alerts",
            "Weekly BRI trend analysis and correlation with VIX",
            "Monthly BRI regime analysis for structural changes",
            "Quarterly BRI validation against market events",
            "Annual BRI model recalibration and validation"
        ]
        
        return insights
    
    def _generate_research_plots(self, df: pd.DataFrame, results: Dict):
        """Generate comprehensive research plots."""
        # This would implement comprehensive plotting
        # For now, create basic plots
        self.logger.info("Generating research plots...")
        
        # Create output directory
        import os
        os.makedirs('output/research_plots', exist_ok=True)
        
        # BRI vs VIX over time
        if 'BRI_t' in df.columns and '^VIX_Close' in df.columns:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # BRI time series
            ax1.plot(df['date'], df['BRI_t'], label='BRI', color='blue')
            ax1.set_title('Behavioral Risk Index Over Time')
            ax1.set_ylabel('BRI')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # VIX time series
            ax2.plot(df['date'], df['^VIX_Close'], label='VIX', color='red')
            ax2.set_title('VIX Over Time')
            ax2.set_ylabel('VIX')
            ax2.set_xlabel('Date')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('output/research_plots/bri_vix_timeseries.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info("Research plots generated successfully")
    
    def save_results(self, results: Dict, output_dir: str = 'output'):
        """Save research results to files."""
        import json
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main results
        with open(f'{output_dir}/research_analysis.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Research results saved to {output_dir}/research_analysis.json")
