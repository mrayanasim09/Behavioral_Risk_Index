#!/usr/bin/env python3
"""
Comprehensive Backtesting Report Generator
- 2 years of backtesting data
- Multiple scenarios analysis
- Options data integration
- Advanced risk metrics
- Professional report generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import json
import os
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveBacktestingReport:
    """Generate comprehensive backtesting reports with multiple scenarios"""
    
    def __init__(self, data_dir='output/enhanced_5year', output_dir='output/backtesting_reports'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/charts", exist_ok=True)
        os.makedirs(f"{self.output_dir}/tables", exist_ok=True)
        
        # Load data
        self.bri_data = self._load_bri_data()
        self.market_data = self._load_market_data()
        self.options_data = self._load_options_data()
        
        # Backtesting scenarios
        self.scenarios = {
            'baseline': {
                'name': 'Baseline Scenario',
                'description': 'Normal market conditions with standard BRI thresholds',
                'bri_threshold': 0.7,
                'vix_threshold': 0.7,
                'lookback_days': 30
            },
            'crisis': {
                'name': 'Crisis Scenario',
                'description': 'Market crisis with elevated BRI and VIX levels',
                'bri_threshold': 0.8,
                'vix_threshold': 0.8,
                'lookback_days': 14
            },
            'recovery': {
                'name': 'Recovery Scenario',
                'description': 'Post-crisis recovery with lower thresholds',
                'bri_threshold': 0.6,
                'vix_threshold': 0.6,
                'lookback_days': 45
            },
            'high_volatility': {
                'name': 'High Volatility Scenario',
                'description': 'High volatility period with dynamic thresholds',
                'bri_threshold': 0.75,
                'vix_threshold': 0.75,
                'lookback_days': 21
            },
            'bull_market': {
                'name': 'Bull Market Scenario',
                'description': 'Strong bull market with optimistic sentiment',
                'bri_threshold': 0.65,
                'vix_threshold': 0.65,
                'lookback_days': 60
            },
            'bear_market': {
                'name': 'Bear Market Scenario',
                'description': 'Extended bear market with pessimistic sentiment',
                'bri_threshold': 0.85,
                'vix_threshold': 0.85,
                'lookback_days': 14
            }
        }
    
    def _load_bri_data(self):
        """Load BRI data"""
        try:
            bri_file = f"{self.data_dir}/enhanced_bri_data.csv"
            if os.path.exists(bri_file):
                df = pd.read_csv(bri_file)
                df['date'] = pd.to_datetime(df['date'])
                logger.info(f"Loaded BRI data: {len(df)} records")
                return df
            else:
                logger.warning("No BRI data found, generating sample data")
                return self._generate_sample_bri_data()
        except Exception as e:
            logger.error(f"Error loading BRI data: {e}")
            return self._generate_sample_bri_data()
    
    def _load_market_data(self):
        """Load market data"""
        try:
            market_file = f"{self.data_dir}/market_data.csv"
            if os.path.exists(market_file):
                df = pd.read_csv(market_file)
                df['date'] = pd.to_datetime(df['date'])
                logger.info(f"Loaded market data: {len(df)} records")
                return df
            else:
                logger.warning("No market data found, generating sample data")
                return self._generate_sample_market_data()
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            return self._generate_sample_market_data()
    
    def _load_options_data(self):
        """Load options data"""
        try:
            options_file = f"{self.data_dir}/options_data.csv"
            if os.path.exists(options_file):
                df = pd.read_csv(options_file)
                df['date'] = pd.to_datetime(df['date'])
                logger.info(f"Loaded options data: {len(df)} records")
                return df
            else:
                logger.warning("No options data found, generating sample data")
                return self._generate_sample_options_data()
        except Exception as e:
            logger.error(f"Error loading options data: {e}")
            return self._generate_sample_options_data()
    
    def _generate_sample_bri_data(self):
        """Generate sample BRI data for 2 years"""
        logger.info("Generating sample BRI data for 2 years...")
        
        # Create date range for 2 years
        dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
        
        # Generate realistic BRI data
        np.random.seed(42)
        n_days = len(dates)
        
        # Base BRI with trend and seasonality
        trend = np.linspace(25, 35, n_days)
        seasonal = 5 * np.sin(2 * np.pi * np.arange(n_days) / 252)
        noise = np.random.randn(n_days) * 8
        
        # Add crisis periods
        crisis_periods = [
            (pd.Timestamp('2022-03-01'), pd.Timestamp('2022-04-15')),
            (pd.Timestamp('2022-09-01'), pd.Timestamp('2022-10-31')),
            (pd.Timestamp('2023-03-01'), pd.Timestamp('2023-04-15')),
            (pd.Timestamp('2023-10-01'), pd.Timestamp('2023-11-30')),
        ]
        
        crisis_boost = np.zeros(n_days)
        for start, end in crisis_periods:
            mask = (dates >= start) & (dates <= end)
            crisis_boost[mask] = np.random.uniform(15, 25, np.sum(mask))
        
        # Combine components
        bri_values = trend + seasonal + noise + crisis_boost
        bri_values = np.clip(bri_values, 0, 100)
        
        # Create DataFrame
        bri_data = pd.DataFrame({
            'date': dates,
            'BRI': bri_values,
            'BRI_MA_7': pd.Series(bri_values).rolling(7).mean(),
            'BRI_MA_30': pd.Series(bri_values).rolling(30).mean(),
            'BRI_volatility': pd.Series(bri_values).rolling(30).std(),
            'sentiment_volatility': np.random.uniform(0, 50, n_days),
            'media_herding': np.random.uniform(0, 100, n_days),
            'news_tone': np.random.uniform(0, 1, n_days),
            'event_density': np.random.uniform(0, 50, n_days),
            'polarity_skew': np.random.uniform(-2, 2, n_days)
        })
        
        return bri_data
    
    def _generate_sample_market_data(self):
        """Generate sample market data"""
        logger.info("Generating sample market data...")
        
        dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
        symbols = ['^VIX', '^GSPC', '^IXIC', '^DJI', 'SPY', 'QQQ', 'IWM']
        
        market_data = []
        for symbol in symbols:
            for date in dates:
                if symbol == '^VIX':
                    base_price = np.random.uniform(15, 35)
                elif symbol == '^GSPC':
                    base_price = np.random.uniform(3000, 5000)
                elif symbol == '^IXIC':
                    base_price = np.random.uniform(10000, 15000)
                elif symbol == '^DJI':
                    base_price = np.random.uniform(25000, 35000)
                else:
                    base_price = np.random.uniform(100, 500)
                
                volatility = np.random.uniform(0.01, 0.05)
                price_change = np.random.normal(0, volatility)
                close_price = base_price * (1 + price_change)
                
                market_data.append({
                    'date': date,
                    'symbol': symbol,
                    'Close': close_price,
                    'Volume': np.random.randint(1000000, 10000000)
                })
        
        return pd.DataFrame(market_data)
    
    def _generate_sample_options_data(self):
        """Generate sample options data"""
        logger.info("Generating sample options data...")
        
        dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
        symbols = ['SPY', 'QQQ', 'IWM', 'VIX']
        option_types = ['calls', 'puts']
        
        options_data = []
        for symbol in symbols:
            for date in dates:
                for option_type in option_types:
                    for i in range(5):  # 5 options per symbol/type/date
                        strike = np.random.uniform(100, 500)
                        expiration = date + timedelta(days=np.random.randint(30, 90))
                        
                        options_data.append({
                            'date': date,
                            'symbol': symbol,
                            'option_type': option_type,
                            'strike': strike,
                            'expiration': expiration,
                            'last_price': np.random.uniform(0.1, 50),
                            'volume': np.random.randint(0, 1000),
                            'open_interest': np.random.randint(0, 10000),
                            'implied_volatility': np.random.uniform(0.1, 0.8)
                        })
        
        return pd.DataFrame(options_data)
    
    def run_scenario_backtesting(self, scenario_name, scenario_config):
        """Run backtesting for a specific scenario"""
        logger.info(f"Running backtesting for {scenario_name} scenario...")
        
        # Filter data for scenario
        if scenario_name == 'crisis':
            # Focus on high BRI periods
            scenario_data = self.bri_data[self.bri_data['BRI'] > self.bri_data['BRI'].quantile(0.8)]
        elif scenario_name == 'recovery':
            # Focus on recovery periods
            scenario_data = self.bri_data[self.bri_data['BRI'] < self.bri_data['BRI'].quantile(0.3)]
        elif scenario_name == 'high_volatility':
            # Focus on high volatility periods
            scenario_data = self.bri_data[self.bri_data['BRI_volatility'] > self.bri_data['BRI_volatility'].quantile(0.8)]
        else:
            scenario_data = self.bri_data.copy()
        
        if len(scenario_data) < 30:
            return {'error': f'Insufficient data for {scenario_name} scenario'}
        
        # Calculate returns
        scenario_data = scenario_data.sort_values('date')
        scenario_data['BRI_return'] = scenario_data['BRI'].pct_change()
        
        # Get market returns
        market_pivot = self.market_data.pivot_table(
            index='date', 
            columns='symbol', 
            values='Close', 
            aggfunc='first'
        )
        
        # Calculate market returns
        market_returns = market_pivot.pct_change()
        
        # Merge with scenario data
        scenario_data = scenario_data.merge(
            market_returns[['^GSPC', '^VIX']], 
            left_on='date', 
            right_index=True, 
            how='left'
        )
        
        # Generate signals
        bri_threshold = scenario_data['BRI'].rolling(scenario_config['lookback_days']).quantile(scenario_config['bri_threshold'])
        vix_threshold = scenario_data['^VIX'].rolling(scenario_config['lookback_days']).quantile(scenario_config['vix_threshold'])
        
        scenario_data['BRI_signal'] = (scenario_data['BRI'] > bri_threshold).astype(int)
        scenario_data['VIX_signal'] = (scenario_data['^VIX'] > vix_threshold).astype(int)
        scenario_data['combined_signal'] = (scenario_data['BRI_signal'] | scenario_data['VIX_signal']).astype(int)
        
        # Calculate strategy returns
        scenario_data['strategy_return'] = scenario_data['combined_signal'] * scenario_data['^GSPC']
        scenario_data['strategy_return'] = scenario_data['strategy_return'].fillna(0)
        scenario_data['cumulative_return'] = (1 + scenario_data['strategy_return']).cumprod()
        
        # Calculate performance metrics
        total_return = scenario_data['cumulative_return'].iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(scenario_data)) - 1
        volatility = scenario_data['strategy_return'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        peak = scenario_data['cumulative_return'].expanding().max()
        drawdown = (scenario_data['cumulative_return'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calculate VaR and CVaR
        returns = scenario_data['strategy_return'].dropna()
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Calculate additional metrics
        win_rate = (scenario_data['strategy_return'] > 0).mean()
        avg_win = scenario_data[scenario_data['strategy_return'] > 0]['strategy_return'].mean()
        avg_loss = scenario_data[scenario_data['strategy_return'] < 0]['strategy_return'].mean()
        
        # Calculate signal quality
        vix_spikes = scenario_data['^VIX'] > scenario_data['^VIX'].rolling(30).quantile(0.8)
        signal_precision = (scenario_data['combined_signal'] & vix_spikes).sum() / scenario_data['combined_signal'].sum() if scenario_data['combined_signal'].sum() > 0 else 0
        signal_recall = (scenario_data['combined_signal'] & vix_spikes).sum() / vix_spikes.sum() if vix_spikes.sum() > 0 else 0
        signal_f1 = 2 * signal_precision * signal_recall / (signal_precision + signal_recall) if (signal_precision + signal_recall) > 0 else 0
        
        results = {
            'scenario': scenario_name,
            'period_days': len(scenario_data),
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'signal_precision': signal_precision,
            'signal_recall': signal_recall,
            'signal_f1': signal_f1,
            'bri_correlation': scenario_data['BRI'].corr(scenario_data['^VIX']),
            'bri_market_correlation': scenario_data['BRI'].corr(scenario_data['^GSPC'])
        }
        
        return results
    
    def run_all_scenarios(self):
        """Run backtesting for all scenarios"""
        logger.info("Running backtesting for all scenarios...")
        
        all_results = {}
        
        for scenario_name, scenario_config in self.scenarios.items():
            results = self.run_scenario_backtesting(scenario_name, scenario_config)
            all_results[scenario_name] = results
            
            # Save individual scenario results
            scenario_file = f"{self.output_dir}/tables/{scenario_name}_results.csv"
            if 'error' not in results:
                pd.DataFrame([results]).to_csv(scenario_file, index=False)
        
        return all_results
    
    def generate_comprehensive_report(self, backtesting_results):
        """Generate comprehensive backtesting report"""
        logger.info("Generating comprehensive backtesting report...")
        
        # Create summary table
        summary_data = []
        for scenario_name, results in backtesting_results.items():
            if 'error' not in results:
                summary_data.append({
                    'Scenario': scenario_name,
                    'Period (Days)': results['period_days'],
                    'Total Return': f"{results['total_return']:.2%}",
                    'Annual Return': f"{results['annual_return']:.2%}",
                    'Volatility': f"{results['volatility']:.2%}",
                    'Sharpe Ratio': f"{results['sharpe_ratio']:.2f}",
                    'Max Drawdown': f"{results['max_drawdown']:.2%}",
                    'VaR 95%': f"{results['var_95']:.2%}",
                    'VaR 99%': f"{results['var_99']:.2%}",
                    'Win Rate': f"{results['win_rate']:.2%}",
                    'Signal Precision': f"{results['signal_precision']:.2%}",
                    'Signal Recall': f"{results['signal_recall']:.2%}",
                    'Signal F1': f"{results['signal_f1']:.2f}"
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{self.output_dir}/tables/backtesting_summary.csv", index=False)
        
        # Create detailed report
        report = {
            'executive_summary': {
                'total_scenarios': len(backtesting_results),
                'successful_scenarios': len([r for r in backtesting_results.values() if 'error' not in r]),
                'best_performing_scenario': max(backtesting_results.items(), key=lambda x: x[1].get('annual_return', -999))[0] if any('error' not in r for r in backtesting_results.values()) else 'None',
                'average_annual_return': np.mean([r.get('annual_return', 0) for r in backtesting_results.values() if 'error' not in r]),
                'average_sharpe_ratio': np.mean([r.get('sharpe_ratio', 0) for r in backtesting_results.values() if 'error' not in r])
            },
            'scenario_analysis': backtesting_results,
            'risk_metrics': {
                'average_var_95': np.mean([r.get('var_95', 0) for r in backtesting_results.values() if 'error' not in r]),
                'average_var_99': np.mean([r.get('var_99', 0) for r in backtesting_results.values() if 'error' not in r]),
                'average_max_drawdown': np.mean([r.get('max_drawdown', 0) for r in backtesting_results.values() if 'error' not in r])
            },
            'signal_quality': {
                'average_precision': np.mean([r.get('signal_precision', 0) for r in backtesting_results.values() if 'error' not in r]),
                'average_recall': np.mean([r.get('signal_recall', 0) for r in backtesting_results.values() if 'error' not in r]),
                'average_f1': np.mean([r.get('signal_f1', 0) for r in backtesting_results.values() if 'error' not in r])
            }
        }
        
        # Save report
        with open(f"{self.output_dir}/comprehensive_backtesting_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create visualizations
        self.create_backtesting_charts(backtesting_results)
        
        logger.info(f"Comprehensive report saved to {self.output_dir}")
        return report
    
    def create_backtesting_charts(self, backtesting_results):
        """Create backtesting visualization charts"""
        logger.info("Creating backtesting charts...")
        
        # Performance comparison chart
        scenarios = [name for name, results in backtesting_results.items() if 'error' not in results]
        annual_returns = [results['annual_return'] for results in backtesting_results.values() if 'error' not in results]
        sharpe_ratios = [results['sharpe_ratio'] for results in backtesting_results.values() if 'error' not in results]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Annual Returns by Scenario', 'Sharpe Ratios by Scenario', 
                          'Risk-Return Scatter', 'Signal Quality Metrics'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Annual returns
        fig.add_trace(
            go.Bar(x=scenarios, y=annual_returns, name='Annual Return'),
            row=1, col=1
        )
        
        # Sharpe ratios
        fig.add_trace(
            go.Bar(x=scenarios, y=sharpe_ratios, name='Sharpe Ratio'),
            row=1, col=2
        )
        
        # Risk-return scatter
        volatilities = [results['volatility'] for results in backtesting_results.values() if 'error' not in results]
        fig.add_trace(
            go.Scatter(x=volatilities, y=annual_returns, mode='markers+text',
                      text=scenarios, textposition='top center', name='Risk-Return'),
            row=2, col=1
        )
        
        # Signal quality
        precisions = [results['signal_precision'] for results in backtesting_results.values() if 'error' not in results]
        recalls = [results['signal_recall'] for results in backtesting_results.values() if 'error' not in results]
        
        fig.add_trace(
            go.Bar(x=scenarios, y=precisions, name='Precision'),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Comprehensive Backtesting Analysis',
            height=800,
            showlegend=True
        )
        
        fig.write_html(f"{self.output_dir}/charts/backtesting_analysis.html")
        # fig.write_image(f"{self.output_dir}/charts/backtesting_analysis.png")  # Requires kaleido
        
        logger.info("Backtesting charts created successfully")
    
    def run_complete_analysis(self):
        """Run complete backtesting analysis"""
        logger.info("Starting comprehensive backtesting analysis...")
        
        # Run all scenarios
        backtesting_results = self.run_all_scenarios()
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report(backtesting_results)
        
        logger.info("Comprehensive backtesting analysis completed successfully!")
        return backtesting_results, report

if __name__ == "__main__":
    # Initialize backtesting report generator
    report_generator = ComprehensiveBacktestingReport()
    
    # Run complete analysis
    backtesting_results, report = report_generator.run_complete_analysis()
    
    print("\n" + "="*60)
    print("COMPREHENSIVE BACKTESTING ANALYSIS RESULTS")
    print("="*60)
    print(f"Total scenarios analyzed: {len(backtesting_results)}")
    print(f"Successful scenarios: {len([r for r in backtesting_results.values() if 'error' not in r])}")
    
    if any('error' not in r for r in backtesting_results.values()):
        best_scenario = max(backtesting_results.items(), key=lambda x: x[1].get('annual_return', -999))[0]
        best_return = backtesting_results[best_scenario]['annual_return']
        print(f"Best performing scenario: {best_scenario} ({best_return:.2%} annual return)")
        
        avg_return = np.mean([r.get('annual_return', 0) for r in backtesting_results.values() if 'error' not in r])
        avg_sharpe = np.mean([r.get('sharpe_ratio', 0) for r in backtesting_results.values() if 'error' not in r])
        print(f"Average annual return: {avg_return:.2%}")
        print(f"Average Sharpe ratio: {avg_sharpe:.2f}")
    
    print("="*60)
    print(f"Detailed results saved to: {report_generator.output_dir}")
    print("="*60)
