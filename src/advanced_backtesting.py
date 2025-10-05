"""
Advanced Backtesting and Trading Simulation for BRI Dashboard
Implements signal quality analysis, trading simulation, and comparison with other indicators
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class AdvancedBacktesting:
    """Advanced backtesting and trading simulation for BRI analysis"""
    
    def __init__(self, bri_data, market_data=None):
        self.bri_data = bri_data.copy()
        self.market_data = market_data
        self.backtesting_results = {}
        
    def signal_quality_analysis(self, target_threshold=0.10, lookforward_days=1):
        """
        Analyze signal quality for predicting VIX spikes
        
        Args:
            target_threshold (float): VIX spike threshold (e.g., 0.10 for 10%)
            lookforward_days (int): Days to look forward for target
            
        Returns:
            dict: Signal quality analysis results
        """
        try:
            if self.market_data is None or 'VIX' not in self.market_data.columns:
                return {'error': 'VIX data not available'}
            
            # Merge data
            merged = pd.merge(self.bri_data, self.market_data, on='date', how='inner')
            if len(merged) < 100:
                return {'error': 'Insufficient data for signal quality analysis'}
            
            # Calculate VIX returns
            vix_values = merged['VIX'].values
            vix_returns = np.diff(vix_values) / vix_values[:-1]
            
            # Create target variable (VIX spikes)
            targets = vix_returns > target_threshold
            
            # Align BRI signals with targets
            bri_signals = merged['BRI'].values[:-1]  # Remove last value to align with returns
            
            # Test different BRI thresholds
            thresholds = [50, 60, 70, 80, 90, 95]
            threshold_results = {}
            
            for threshold in thresholds:
                signals = bri_signals > threshold
                
                if len(signals) == len(targets):
                    # Calculate metrics
                    precision = precision_score(targets, signals, zero_division=0)
                    recall = recall_score(targets, signals, zero_division=0)
                    f1 = f1_score(targets, signals, zero_division=0)
                    
                    # Calculate hit rate
                    hit_rate = np.sum(signals & targets) / np.sum(signals) if np.sum(signals) > 0 else 0
                    
                    # Calculate false positive rate
                    false_positive_rate = np.sum(signals & ~targets) / np.sum(~targets) if np.sum(~targets) > 0 else 0
                    
                    threshold_results[threshold] = {
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'hit_rate': hit_rate,
                        'false_positive_rate': false_positive_rate,
                        'signal_count': int(np.sum(signals)),
                        'target_count': int(np.sum(targets))
                    }
            
            # Find optimal threshold
            best_threshold = max(threshold_results.keys(), 
                               key=lambda x: threshold_results[x]['f1_score'])
            
            results = {
                'threshold_results': threshold_results,
                'best_threshold': best_threshold,
                'best_metrics': threshold_results[best_threshold],
                'target_threshold': target_threshold,
                'lookforward_days': lookforward_days,
                'total_observations': len(targets),
                'total_spikes': int(np.sum(targets))
            }
            
            self.backtesting_results['signal_quality'] = results
            return results
            
        except Exception as e:
            print(f"Error in signal quality analysis: {e}")
            return {'error': str(e)}
    
    def trading_simulation(self, initial_capital=100000, transaction_cost=0.001):
        """
        Simulate trading strategy using BRI signals
        
        Args:
            initial_capital (float): Initial capital for simulation
            transaction_cost (float): Transaction cost as percentage
            
        Returns:
            dict: Trading simulation results
        """
        try:
            if self.market_data is None or 'VIX' not in self.market_data.columns:
                return {'error': 'VIX data not available'}
            
            # Merge data
            merged = pd.merge(self.bri_data, self.market_data, on='date', how='inner')
            if len(merged) < 100:
                return {'error': 'Insufficient data for trading simulation'}
            
            # Use SP500 as proxy for market returns
            if 'SP500' in merged.columns:
                market_values = merged['SP500'].values
            else:
                # Generate synthetic market data
                market_values = 4000 + np.cumsum(np.random.normal(0, 50, len(merged)))
            
            # Calculate market returns
            market_returns = np.diff(market_values) / market_values[:-1]
            
            # BRI signals (using 70th percentile threshold)
            bri_threshold = np.percentile(merged['BRI'].values, 70)
            bri_signals = merged['BRI'].values[:-1] > bri_threshold  # Remove last value to align
            
            # Align signals with returns
            if len(bri_signals) > len(market_returns):
                bri_signals = bri_signals[:len(market_returns)]
            elif len(market_returns) > len(bri_signals):
                market_returns = market_returns[:len(bri_signals)]
            
            # Simulate trading
            capital = initial_capital
            positions = []
            returns = []
            transaction_costs = []
            
            for i, (signal, market_return) in enumerate(zip(bri_signals, market_returns)):
                if signal:
                    # Go short (expect market to fall when BRI is high)
                    position_return = -market_return
                    transaction_cost = capital * transaction_cost
                else:
                    # Go long (normal market exposure)
                    position_return = market_return
                    transaction_cost = capital * transaction_cost * 0.5  # Lower cost for long positions
                
                # Update capital
                capital = capital * (1 + position_return) - transaction_cost
                
                positions.append(position_return)
                returns.append(position_return)
                transaction_costs.append(transaction_cost)
            
            # Calculate performance metrics
            total_return = (capital - initial_capital) / initial_capital
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Calculate maximum drawdown
            cumulative_returns = np.cumprod(1 + np.array(returns))
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # Calculate win rate
            win_rate = np.sum(np.array(returns) > 0) / len(returns)
            
            results = {
                'initial_capital': initial_capital,
                'final_capital': capital,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_transactions': len(returns),
                'total_transaction_costs': sum(transaction_costs),
                'bri_threshold': bri_threshold,
                'transaction_cost': transaction_cost
            }
            
            self.backtesting_results['trading_simulation'] = results
            return results
            
        except Exception as e:
            print(f"Error in trading simulation: {e}")
            return {'error': str(e)}
    
    def indicator_comparison(self):
        """
        Compare BRI with other market indicators
        
        Returns:
            dict: Indicator comparison results
        """
        try:
            if self.market_data is None:
                return {'error': 'Market data not available'}
            
            # Merge data
            merged = pd.merge(self.bri_data, self.market_data, on='date', how='inner')
            if len(merged) < 100:
                return {'error': 'Insufficient data for indicator comparison'}
            
            # Calculate VIX returns for target
            vix_values = merged['VIX'].values
            vix_returns = np.diff(vix_values) / vix_values[:-1]
            targets = vix_returns > 0.10  # 10% VIX spike threshold
            
            # Prepare indicators
            indicators = {}
            
            # BRI
            bri_values = merged['BRI'].values[:-1]
            indicators['BRI'] = bri_values
            
            # VIX (lagged)
            vix_lagged = merged['VIX'].values[:-1]
            indicators['VIX'] = vix_lagged
            
            # SP500 returns (if available)
            if 'SP500' in merged.columns:
                sp500_values = merged['SP500'].values
                sp500_returns = np.diff(sp500_values) / sp500_values[:-1]
                indicators['SP500_Returns'] = sp500_returns
            
            # BRI volatility
            bri_vol = pd.Series(bri_values).rolling(window=30).std().values
            indicators['BRI_Volatility'] = bri_vol
            
            # VIX volatility
            vix_vol = pd.Series(vix_lagged).rolling(window=30).std().values
            indicators['VIX_Volatility'] = vix_vol
            
            # Compare indicators
            comparison_results = {}
            
            for indicator_name, indicator_values in indicators.items():
                if len(indicator_values) == len(targets):
                    # Calculate correlation with targets
                    correlation = np.corrcoef(indicator_values, targets.astype(int))[0, 1]
                    
                    # Calculate AUC (Area Under Curve)
                    try:
                        auc = roc_auc_score(targets, indicator_values)
                    except:
                        auc = 0.5
                    
                    # Calculate precision/recall at 70th percentile
                    threshold = np.percentile(indicator_values, 70)
                    signals = indicator_values > threshold
                    
                    precision = precision_score(targets, signals, zero_division=0)
                    recall = recall_score(targets, signals, zero_division=0)
                    f1 = f1_score(targets, signals, zero_division=0)
                    
                    comparison_results[indicator_name] = {
                        'correlation': correlation,
                        'auc': auc,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'threshold': threshold
                    }
            
            # Rank indicators
            ranked_indicators = sorted(comparison_results.items(), 
                                     key=lambda x: x[1]['f1_score'], 
                                     reverse=True)
            
            results = {
                'indicator_comparison': comparison_results,
                'ranked_indicators': ranked_indicators,
                'best_indicator': ranked_indicators[0][0] if ranked_indicators else None,
                'target_threshold': 0.10,
                'total_observations': len(targets)
            }
            
            self.backtesting_results['indicator_comparison'] = results
            return results
            
        except Exception as e:
            print(f"Error in indicator comparison: {e}")
            return {'error': str(e)}
    
    def regime_switching_analysis(self, n_regimes=2):
        """
        Perform regime switching analysis for BRI and market data
        
        Args:
            n_regimes (int): Number of regimes to detect
            
        Returns:
            dict: Regime switching analysis results
        """
        try:
            if self.market_data is None:
                return {'error': 'Market data not available'}
            
            # Merge data
            merged = pd.merge(self.bri_data, self.market_data, on='date', how='inner')
            if len(merged) < 100:
                return {'error': 'Insufficient data for regime switching analysis'}
            
            # Prepare data
            bri_values = merged['BRI'].values
            vix_values = merged['VIX'].values
            
            # Simple regime detection based on volatility
            bri_vol = pd.Series(bri_values).rolling(window=30).std()
            vix_vol = pd.Series(vix_values).rolling(window=30).std()
            
            # Define regimes
            high_vol_threshold = bri_vol.quantile(0.7)
            low_vol_threshold = bri_vol.quantile(0.3)
            
            regimes = []
            for i in range(len(merged)):
                if bri_vol.iloc[i] > high_vol_threshold:
                    regimes.append('High Volatility')
                elif bri_vol.iloc[i] < low_vol_threshold:
                    regimes.append('Low Volatility')
                else:
                    regimes.append('Moderate Volatility')
            
            # Calculate regime-specific statistics
            regime_stats = {}
            for regime in ['High Volatility', 'Moderate Volatility', 'Low Volatility']:
                regime_mask = [r == regime for r in regimes]
                if any(regime_mask):
                    regime_bri = bri_values[regime_mask]
                    regime_vix = vix_values[regime_mask]
                    
                    # Calculate BRI-VIX correlation in this regime
                    correlation = np.corrcoef(regime_bri, regime_vix)[0, 1] if len(regime_bri) > 1 else 0
                    
                    regime_stats[regime] = {
                        'count': sum(regime_mask),
                        'percentage': sum(regime_mask) / len(regimes) * 100,
                        'bri_mean': np.mean(regime_bri),
                        'bri_std': np.std(regime_bri),
                        'vix_mean': np.mean(regime_vix),
                        'vix_std': np.std(regime_vix),
                        'correlation': correlation
                    }
            
            # Calculate regime transition probabilities
            transitions = self._calculate_regime_transitions(regimes)
            
            results = {
                'regimes': regimes,
                'regime_stats': regime_stats,
                'regime_transitions': transitions,
                'n_regimes': len(set(regimes)),
                'regime_duration': self._calculate_regime_duration(regimes)
            }
            
            self.backtesting_results['regime_switching'] = results
            return results
            
        except Exception as e:
            print(f"Error in regime switching analysis: {e}")
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
    
    def _calculate_regime_duration(self, regimes):
        """Calculate average duration of each regime"""
        durations = {}
        current_regime = regimes[0]
        duration = 1
        
        for i in range(1, len(regimes)):
            if regimes[i] == current_regime:
                duration += 1
            else:
                if current_regime not in durations:
                    durations[current_regime] = []
                durations[current_regime].append(duration)
                current_regime = regimes[i]
                duration = 1
        
        # Add last regime
        if current_regime not in durations:
            durations[current_regime] = []
        durations[current_regime].append(duration)
        
        # Calculate averages
        avg_durations = {}
        for regime, duration_list in durations.items():
            avg_durations[regime] = np.mean(duration_list)
        
        return avg_durations
    
    def create_backtesting_visualizations(self):
        """
        Create visualizations for backtesting results
        
        Returns:
            dict: Visualization data
        """
        try:
            visualizations = {}
            
            # Signal quality visualization
            if 'signal_quality' in self.backtesting_results:
                sq_results = self.backtesting_results['signal_quality']
                if 'threshold_results' in sq_results:
                    thresholds = list(sq_results['threshold_results'].keys())
                    precisions = [sq_results['threshold_results'][t]['precision'] for t in thresholds]
                    recalls = [sq_results['threshold_results'][t]['recall'] for t in thresholds]
                    f1_scores = [sq_results['threshold_results'][t]['f1_score'] for t in thresholds]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=thresholds, y=precisions, mode='lines+markers', name='Precision'))
                    fig.add_trace(go.Scatter(x=thresholds, y=recalls, mode='lines+markers', name='Recall'))
                    fig.add_trace(go.Scatter(x=thresholds, y=f1_scores, mode='lines+markers', name='F1 Score'))
                    
                    fig.update_layout(
                        title='Signal Quality by BRI Threshold',
                        xaxis_title='BRI Threshold',
                        yaxis_title='Score',
                        height=400
                    )
                    
                    visualizations['signal_quality'] = fig.to_dict()
            
            # Trading simulation visualization
            if 'trading_simulation' in self.backtesting_results:
                ts_results = self.backtesting_results['trading_simulation']
                
                # Create performance summary
                performance_data = {
                    'Metric': ['Total Return', 'Annualized Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown'],
                    'Value': [
                        ts_results.get('total_return', 0),
                        ts_results.get('annualized_return', 0),
                        ts_results.get('volatility', 0),
                        ts_results.get('sharpe_ratio', 0),
                        ts_results.get('max_drawdown', 0)
                    ]
                }
                
                fig = go.Figure(data=go.Bar(
                    x=performance_data['Metric'],
                    y=performance_data['Value'],
                    marker_color=['green' if v > 0 else 'red' for v in performance_data['Value']]
                ))
                
                fig.update_layout(
                    title='Trading Strategy Performance',
                    xaxis_title='Metric',
                    yaxis_title='Value',
                    height=400
                )
                
                visualizations['trading_performance'] = fig.to_dict()
            
            # Indicator comparison visualization
            if 'indicator_comparison' in self.backtesting_results:
                ic_results = self.backtesting_results['indicator_comparison']
                if 'indicator_comparison' in ic_results:
                    indicators = list(ic_results['indicator_comparison'].keys())
                    f1_scores = [ic_results['indicator_comparison'][i]['f1_score'] for i in indicators]
                    
                    fig = go.Figure(data=go.Bar(
                        x=indicators,
                        y=f1_scores,
                        marker_color='blue'
                    ))
                    
                    fig.update_layout(
                        title='Indicator Comparison - F1 Scores',
                        xaxis_title='Indicator',
                        yaxis_title='F1 Score',
                        height=400
                    )
                    
                    visualizations['indicator_comparison'] = fig.to_dict()
            
            return visualizations
            
        except Exception as e:
            print(f"Error creating backtesting visualizations: {e}")
            return {'error': str(e)}
    
    def generate_comprehensive_backtesting_report(self):
        """
        Generate comprehensive backtesting report
        
        Returns:
            dict: Complete backtesting report
        """
        try:
            # Run all backtesting analyses
            signal_quality = self.signal_quality_analysis()
            trading_simulation = self.trading_simulation()
            indicator_comparison = self.indicator_comparison()
            regime_switching = self.regime_switching_analysis()
            
            # Create visualizations
            visualizations = self.create_backtesting_visualizations()
            
            # Compile comprehensive report
            report = {
                'signal_quality_analysis': signal_quality,
                'trading_simulation': trading_simulation,
                'indicator_comparison': indicator_comparison,
                'regime_switching_analysis': regime_switching,
                'visualizations': visualizations,
                'generated_at': pd.Timestamp.now().isoformat(),
                'data_points': len(self.bri_data),
                'backtesting_summary': self._create_backtesting_summary()
            }
            
            return report
            
        except Exception as e:
            print(f"Error generating comprehensive backtesting report: {e}")
            return {'error': str(e)}
    
    def _create_backtesting_summary(self):
        """Create backtesting summary"""
        summary = {
            'analyses_completed': len(self.backtesting_results),
            'overall_performance': 'GOOD' if len(self.backtesting_results) > 3 else 'PARTIAL',
            'key_metrics': []
        }
        
        # Add key metrics
        if 'signal_quality' in self.backtesting_results:
            sq = self.backtesting_results['signal_quality']
            if 'best_metrics' in sq:
                summary['key_metrics'].append(f"Best F1 Score: {sq['best_metrics']['f1_score']:.3f}")
        
        if 'trading_simulation' in self.backtesting_results:
            ts = self.backtesting_results['trading_simulation']
            if 'sharpe_ratio' in ts:
                summary['key_metrics'].append(f"Sharpe Ratio: {ts['sharpe_ratio']:.3f}")
        
        if 'indicator_comparison' in self.backtesting_results:
            ic = self.backtesting_results['indicator_comparison']
            if 'best_indicator' in ic:
                summary['key_metrics'].append(f"Best Indicator: {ic['best_indicator']}")
        
        return summary
