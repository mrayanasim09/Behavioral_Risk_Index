"""
Monte Carlo Simulations for BRI Dashboard
Implements risk scenario modeling with 1,000+ simulations
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import norm, t, skewnorm
import warnings
warnings.filterwarnings('ignore')

class MonteCarloSimulations:
    """Monte Carlo simulations for risk scenario modeling"""
    
    def __init__(self, bri_data, market_data=None):
        self.bri_data = bri_data.copy()
        self.market_data = market_data
        self.simulation_results = {}
        
    def fit_distribution(self, data, distribution_type='normal'):
        """
        Fit distribution to data
        
        Args:
            data (array): Data to fit
            distribution_type (str): Type of distribution to fit
            
        Returns:
            dict: Distribution parameters
        """
        try:
            if distribution_type == 'normal':
                params = norm.fit(data)
                return {'type': 'normal', 'params': params}
            elif distribution_type == 't':
                params = t.fit(data)
                return {'type': 't', 'params': params}
            elif distribution_type == 'skewed_normal':
                params = skewnorm.fit(data)
                return {'type': 'skewed_normal', 'params': params}
            else:
                return {'type': 'normal', 'params': norm.fit(data)}
        except Exception as e:
            print(f"Error fitting distribution: {e}")
            return {'type': 'normal', 'params': norm.fit(data)}
    
    def simulate_bri_paths(self, n_simulations=1000, time_horizon=30, method='bootstrap'):
        """
        Simulate BRI paths using Monte Carlo
        
        Args:
            n_simulations (int): Number of simulations
            time_horizon (int): Time horizon in days
            method (str): Simulation method ('bootstrap', 'parametric', 'garch')
            
        Returns:
            dict: Simulation results
        """
        try:
            df = self.bri_data.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Get recent BRI data
            recent_bri = df['BRI'].tail(252).values  # Last year of data
            current_bri = df['BRI'].iloc[-1]
            
            if method == 'bootstrap':
                # Bootstrap method - resample from historical data
                simulation_paths = []
                for _ in range(n_simulations):
                    path = [current_bri]
                    for _ in range(time_horizon):
                        # Randomly sample from historical returns
                        random_return = np.random.choice(recent_bri[1:] - recent_bri[:-1])
                        new_value = path[-1] + random_return
                        path.append(max(0, min(100, new_value)))  # Constrain to 0-100
                    simulation_paths.append(path[1:])  # Remove initial value
                    
            elif method == 'parametric':
                # Parametric method - fit distribution and simulate
                returns = recent_bri[1:] - recent_bri[:-1]
                returns = returns[~np.isnan(returns)]
                
                # Fit normal distribution to returns
                mu, sigma = norm.fit(returns)
                
                simulation_paths = []
                for _ in range(n_simulations):
                    path = [current_bri]
                    for _ in range(time_horizon):
                        random_return = np.random.normal(mu, sigma)
                        new_value = path[-1] + random_return
                        path.append(max(0, min(100, new_value)))  # Constrain to 0-100
                    simulation_paths.append(path[1:])  # Remove initial value
                    
            elif method == 'garch':
                # GARCH-like method with volatility clustering
                returns = recent_bri[1:] - recent_bri[:-1]
                returns = returns[~np.isnan(returns)]
                
                # Calculate rolling volatility
                volatility = pd.Series(returns).rolling(window=10).std().fillna(method='bfill')
                
                simulation_paths = []
                for _ in range(n_simulations):
                    path = [current_bri]
                    current_vol = volatility.iloc[-1]
                    
                    for _ in range(time_horizon):
                        # Update volatility (GARCH-like)
                        current_vol = 0.9 * current_vol + 0.1 * abs(returns[-1]) if len(returns) > 0 else current_vol
                        
                        # Generate return with current volatility
                        random_return = np.random.normal(0, current_vol)
                        new_value = path[-1] + random_return
                        path.append(max(0, min(100, new_value)))  # Constrain to 0-100
                    simulation_paths.append(path[1:])  # Remove initial value
            
            # Convert to numpy array
            simulation_paths = np.array(simulation_paths)
            
            # Calculate statistics
            final_values = simulation_paths[:, -1]
            mean_final = np.mean(final_values)
            std_final = np.std(final_values)
            
            # Calculate percentiles
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            percentile_values = {p: np.percentile(final_values, p) for p in percentiles}
            
            # Calculate VaR and CVaR
            var_95 = np.percentile(final_values, 5)  # 5th percentile (95% VaR)
            var_99 = np.percentile(final_values, 1)  # 1st percentile (99% VaR)
            
            cvar_95 = np.mean(final_values[final_values <= var_95])
            cvar_99 = np.mean(final_values[final_values <= var_99])
            
            # Store results
            self.simulation_results[method] = {
                'simulation_paths': simulation_paths,
                'final_values': final_values,
                'mean_final': mean_final,
                'std_final': std_final,
                'percentiles': percentile_values,
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'n_simulations': n_simulations,
                'time_horizon': time_horizon
            }
            
            return self.simulation_results[method]
            
        except Exception as e:
            print(f"Error in Monte Carlo simulation: {e}")
            return None
    
    def create_simulation_visualization(self, method='bootstrap', n_paths_to_show=100):
        """
        Create visualization of Monte Carlo simulations
        
        Args:
            method (str): Simulation method to visualize
            n_paths_to_show (int): Number of paths to show in plot
            
        Returns:
            plotly.graph_objects.Figure: Simulation visualization
        """
        try:
            if method not in self.simulation_results:
                self.simulate_bri_paths(method=method)
            
            results = self.simulation_results[method]
            simulation_paths = results['simulation_paths']
            
            # Create figure
            fig = go.Figure()
            
            # Add simulation paths
            for i in range(min(n_paths_to_show, len(simulation_paths))):
                fig.add_trace(go.Scatter(
                    x=list(range(1, results['time_horizon'] + 1)),
                    y=simulation_paths[i],
                    mode='lines',
                    line=dict(color='rgba(52, 152, 219, 0.1)', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Add mean path
            mean_path = np.mean(simulation_paths, axis=0)
            fig.add_trace(go.Scatter(
                x=list(range(1, results['time_horizon'] + 1)),
                y=mean_path,
                mode='lines',
                name='Mean Path',
                line=dict(color='#E53E3E', width=3)
            ))
            
            # Add confidence intervals
            percentiles = results['percentiles']
            
            # 95% confidence interval
            fig.add_trace(go.Scatter(
                x=list(range(1, results['time_horizon'] + 1)) + list(range(results['time_horizon'], 0, -1)),
                y=[np.percentile(simulation_paths[:, i], 2.5) for i in range(results['time_horizon'])] + 
                  [np.percentile(simulation_paths[:, i], 97.5) for i in range(results['time_horizon'] - 1, -1, -1)],
                fill='tonexty',
                fillcolor='rgba(229, 62, 62, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval',
                hoverinfo='skip'
            ))
            
            # 90% confidence interval
            fig.add_trace(go.Scatter(
                x=list(range(1, results['time_horizon'] + 1)) + list(range(results['time_horizon'], 0, -1)),
                y=[np.percentile(simulation_paths[:, i], 5) for i in range(results['time_horizon'])] + 
                  [np.percentile(simulation_paths[:, i], 95) for i in range(results['time_horizon'] - 1, -1, -1)],
                fill='tonexty',
                fillcolor='rgba(214, 158, 46, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='90% Confidence Interval',
                hoverinfo='skip'
            ))
            
            fig.update_layout(
                title=dict(
                    text=f'Monte Carlo BRI Simulation - {method.title()} Method',
                    font=dict(color='#1A365D', size=18, family='Inter')
                ),
                xaxis_title='Days Ahead',
                yaxis_title='BRI Level',
                height=600,
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                font=dict(color='#1A365D', family='Inter'),
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating simulation visualization: {e}")
            return None
    
    def create_risk_metrics_chart(self, method='bootstrap'):
        """
        Create risk metrics visualization
        
        Args:
            method (str): Simulation method
            
        Returns:
            plotly.graph_objects.Figure: Risk metrics chart
        """
        try:
            if method not in self.simulation_results:
                self.simulate_bri_paths(method=method)
            
            results = self.simulation_results[method]
            final_values = results['final_values']
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Final Value Distribution',
                    'Risk Metrics',
                    'Percentile Analysis',
                    'VaR and CVaR'
                ),
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # 1. Final Value Distribution
            fig.add_trace(
                go.Histogram(
                    x=final_values,
                    nbinsx=50,
                    name='Final Values',
                    marker_color='#3182CE',
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # Add mean line
            fig.add_vline(
                x=results['mean_final'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {results['mean_final']:.1f}",
                row=1, col=1
            )
            
            # 2. Risk Metrics
            risk_metrics = {
                'Mean': results['mean_final'],
                'Std Dev': results['std_final'],
                'VaR 95%': results['var_95'],
                'VaR 99%': results['var_99'],
                'CVaR 95%': results['cvar_95'],
                'CVaR 99%': results['cvar_99']
            }
            
            fig.add_trace(
                go.Bar(
                    x=list(risk_metrics.keys()),
                    y=list(risk_metrics.values()),
                    name='Risk Metrics',
                    marker_color=['#38A169', '#D69E2E', '#E53E3E', '#C53030', '#FC8181', '#F56565']
                ),
                row=1, col=2
            )
            
            # 3. Percentile Analysis
            percentiles = results['percentiles']
            fig.add_trace(
                go.Scatter(
                    x=list(percentiles.keys()),
                    y=list(percentiles.values()),
                    mode='lines+markers',
                    name='Percentiles',
                    line=dict(color='#3182CE', width=3),
                    marker=dict(size=8)
                ),
                row=2, col=1
            )
            
            # 4. VaR and CVaR Comparison
            var_cvar_data = {
                'VaR 95%': results['var_95'],
                'CVaR 95%': results['cvar_95'],
                'VaR 99%': results['var_99'],
                'CVaR 99%': results['cvar_99']
            }
            
            fig.add_trace(
                go.Bar(
                    x=list(var_cvar_data.keys()),
                    y=list(var_cvar_data.values()),
                    name='VaR/CVaR',
                    marker_color=['#E53E3E', '#FC8181', '#C53030', '#F56565']
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title=dict(
                    text=f'Monte Carlo Risk Analysis - {method.title()} Method',
                    font=dict(color='#1A365D', size=18, family='Inter')
                ),
                height=800,
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                font=dict(color='#1A365D', family='Inter'),
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating risk metrics chart: {e}")
            return None
    
    def stress_test_scenarios(self, n_simulations=1000, time_horizon=30):
        """
        Perform stress test scenarios
        
        Args:
            n_simulations (int): Number of simulations
            time_horizon (int): Time horizon in days
            
        Returns:
            dict: Stress test results
        """
        try:
            df = self.bri_data.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            current_bri = df['BRI'].iloc[-1]
            
            # Define stress scenarios
            stress_scenarios = {
                'baseline': {'volatility_multiplier': 1.0, 'drift': 0.0},
                'mild_stress': {'volatility_multiplier': 1.5, 'drift': -0.5},
                'moderate_stress': {'volatility_multiplier': 2.0, 'drift': -1.0},
                'severe_stress': {'volatility_multiplier': 3.0, 'drift': -2.0},
                'extreme_stress': {'volatility_multiplier': 4.0, 'drift': -3.0}
            }
            
            stress_results = {}
            
            for scenario_name, scenario_params in stress_scenarios.items():
                # Get historical volatility
                recent_bri = df['BRI'].tail(252).values
                returns = recent_bri[1:] - recent_bri[:-1]
                returns = returns[~np.isnan(returns)]
                
                base_volatility = np.std(returns)
                adjusted_volatility = base_volatility * scenario_params['volatility_multiplier']
                drift = scenario_params['drift']
                
                # Simulate paths
                simulation_paths = []
                for _ in range(n_simulations):
                    path = [current_bri]
                    for _ in range(time_horizon):
                        random_return = np.random.normal(drift, adjusted_volatility)
                        new_value = path[-1] + random_return
                        path.append(max(0, min(100, new_value)))  # Constrain to 0-100
                    simulation_paths.append(path[1:])  # Remove initial value
                
                simulation_paths = np.array(simulation_paths)
                final_values = simulation_paths[:, -1]
                
                # Calculate metrics
                stress_results[scenario_name] = {
                    'mean_final': np.mean(final_values),
                    'std_final': np.std(final_values),
                    'var_95': np.percentile(final_values, 5),
                    'var_99': np.percentile(final_values, 1),
                    'cvar_95': np.mean(final_values[final_values <= np.percentile(final_values, 5)]),
                    'cvar_99': np.mean(final_values[final_values <= np.percentile(final_values, 1)]),
                    'probability_below_30': np.mean(final_values < 30),
                    'probability_above_70': np.mean(final_values > 70)
                }
            
            return stress_results
            
        except Exception as e:
            print(f"Error in stress testing: {e}")
            return None
    
    def create_stress_test_chart(self, n_simulations=1000, time_horizon=30):
        """
        Create stress test visualization
        
        Args:
            n_simulations (int): Number of simulations
            time_horizon (int): Time horizon in days
            
        Returns:
            plotly.graph_objects.Figure: Stress test chart
        """
        try:
            stress_results = self.stress_test_scenarios(n_simulations, time_horizon)
            
            if not stress_results:
                return None
            
            # Create figure
            fig = go.Figure()
            
            scenarios = list(stress_results.keys())
            colors = ['#38A169', '#D69E2E', '#E53E3E', '#C53030', '#FC8181']
            
            # Add VaR 95% bars
            var_95_values = [stress_results[scenario]['var_95'] for scenario in scenarios]
            fig.add_trace(go.Bar(
                x=scenarios,
                y=var_95_values,
                name='VaR 95%',
                marker_color=colors,
                text=[f'{v:.1f}' for v in var_95_values],
                textposition='auto'
            ))
            
            # Add CVaR 95% bars
            cvar_95_values = [stress_results[scenario]['cvar_95'] for scenario in scenarios]
            fig.add_trace(go.Bar(
                x=scenarios,
                y=cvar_95_values,
                name='CVaR 95%',
                marker_color=[f'rgba({int(c[1:3], 16)}, {int(c[3:5], 16)}, {int(c[5:7], 16)}, 0.7)' for c in colors],
                text=[f'{v:.1f}' for v in cvar_95_values],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=dict(
                    text='Stress Test Scenarios - VaR and CVaR Analysis',
                    font=dict(color='#1A365D', size=18, family='Inter')
                ),
                xaxis_title='Stress Scenario',
                yaxis_title='BRI Level',
                height=600,
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                font=dict(color='#1A365D', family='Inter'),
                barmode='group'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating stress test chart: {e}")
            return None
    
    def generate_monte_carlo_report(self, n_simulations=1000, time_horizon=30):
        """
        Generate comprehensive Monte Carlo report
        
        Args:
            n_simulations (int): Number of simulations
            time_horizon (int): Time horizon in days
            
        Returns:
            dict: Comprehensive Monte Carlo report
        """
        try:
            # Run simulations for all methods
            methods = ['bootstrap', 'parametric', 'garch']
            simulation_results = {}
            
            for method in methods:
                simulation_results[method] = self.simulate_bri_paths(
                    n_simulations=n_simulations,
                    time_horizon=time_horizon,
                    method=method
                )
            
            # Create visualizations
            simulation_charts = {}
            for method in methods:
                simulation_charts[f'{method}_simulation'] = self.create_simulation_visualization(method)
                simulation_charts[f'{method}_risk_metrics'] = self.create_risk_metrics_chart(method)
            
            # Stress testing
            stress_results = self.stress_test_scenarios(n_simulations, time_horizon)
            stress_chart = self.create_stress_test_chart(n_simulations, time_horizon)
            
            return {
                'simulation_results': simulation_results,
                'simulation_charts': simulation_charts,
                'stress_results': stress_results,
                'stress_chart': stress_chart,
                'parameters': {
                    'n_simulations': n_simulations,
                    'time_horizon': time_horizon,
                    'methods': methods
                },
                'generated_at': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error generating Monte Carlo report: {e}")
            return None
