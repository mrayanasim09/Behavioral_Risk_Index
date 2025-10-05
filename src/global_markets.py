"""
Global Markets Support for BRI Dashboard
Implements support for S&P 500, NASDAQ, FTSE, and other global markets
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class GlobalMarkets:
    """Global markets data collection and analysis"""
    
    def __init__(self):
        self.market_symbols = {
            'US': {
                'SP500': '^GSPC',
                'NASDAQ': '^IXIC',
                'DOW': '^DJI',
                'RUSSELL2000': '^RUT',
                'VIX': '^VIX'
            },
            'Europe': {
                'FTSE100': '^FTSE',
                'DAX': '^GDAXI',
                'CAC40': '^FCHI',
                'STOXX50': '^STOXX50E',
                'IBEX35': '^IBEX'
            },
            'Asia': {
                'NIKKEI225': '^N225',
                'HANG_SENG': '^HSI',
                'SHANGHAI': '000001.SS',
                'KOSPI': '^KS11',
                'SENSEX': '^BSESN'
            },
            'Commodities': {
                'GOLD': 'GC=F',
                'SILVER': 'SI=F',
                'OIL': 'CL=F',
                'NATURAL_GAS': 'NG=F',
                'COPPER': 'HG=F'
            },
            'Currencies': {
                'EUR_USD': 'EURUSD=X',
                'GBP_USD': 'GBPUSD=X',
                'USD_JPY': 'USDJPY=X',
                'USD_CHF': 'USDCHF=X',
                'AUD_USD': 'AUDUSD=X'
            }
        }
        
        self.market_data = {}
        self.correlation_data = {}
        
    def fetch_market_data(self, symbol, start_date=None, end_date=None):
        """
        Fetch market data for a specific symbol
        
        Args:
            symbol (str): Market symbol
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: Market data
        """
        try:
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                return None
            
            # Reset index and clean data
            data = data.reset_index()
            data['Date'] = pd.to_datetime(data['Date'])
            data['date'] = data['Date'].dt.strftime('%Y-%m-%d')
            
            # Select relevant columns
            columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
            available_columns = [col for col in columns if col in data.columns]
            data = data[available_columns]
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def fetch_all_markets(self, start_date=None, end_date=None):
        """
        Fetch data for all global markets
        
        Args:
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            
        Returns:
            dict: Market data for all symbols
        """
        try:
            all_data = {}
            
            for region, symbols in self.market_symbols.items():
                region_data = {}
                for name, symbol in symbols.items():
                    print(f"Fetching {name} ({symbol})...")
                    data = self.fetch_market_data(symbol, start_date, end_date)
                    if data is not None:
                        region_data[name] = data
                        all_data[f"{region}_{name}"] = data
                
                self.market_data[region] = region_data
            
            return all_data
            
        except Exception as e:
            print(f"Error fetching all markets: {e}")
            return {}
    
    def calculate_market_correlations(self, bri_data):
        """
        Calculate correlations between BRI and global markets
        
        Args:
            bri_data (pd.DataFrame): BRI data
            
        Returns:
            dict: Correlation results
        """
        try:
            correlations = {}
            
            for region, symbols in self.market_data.items():
                region_correlations = {}
                
                for name, data in symbols.items():
                    if 'Close' in data.columns:
                        # Merge BRI and market data
                        merged = pd.merge(
                            bri_data, 
                            data, 
                            on='date', 
                            how='inner'
                        )
                        
                        if len(merged) > 10:  # Minimum data points
                            # Calculate correlation
                            correlation = merged['BRI'].corr(merged['Close'])
                            
                            # Calculate rolling correlation
                            rolling_corr = merged['BRI'].rolling(window=30).corr(merged['Close'])
                            
                            region_correlations[name] = {
                                'correlation': correlation,
                                'rolling_correlation_mean': rolling_corr.mean(),
                                'rolling_correlation_std': rolling_corr.std(),
                                'data_points': len(merged),
                                'date_range': f"{merged['date'].min()} to {merged['date'].max()}"
                            }
                
                correlations[region] = region_correlations
            
            self.correlation_data = correlations
            return correlations
            
        except Exception as e:
            print(f"Error calculating correlations: {e}")
            return {}
    
    def create_global_markets_overview(self):
        """
        Create global markets overview chart
        
        Returns:
            plotly.graph_objects.Figure: Global markets overview
        """
        try:
            if not self.market_data:
                return None
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('US Markets', 'European Markets', 'Asian Markets', 'Commodities'),
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            # US Markets
            if 'US' in self.market_data:
                for i, (name, data) in enumerate(self.market_data['US'].items()):
                    if 'Close' in data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=pd.to_datetime(data['date']),
                                y=data['Close'],
                                mode='lines',
                                name=name,
                                line=dict(color=colors[i % len(colors)], width=2)
                            ),
                            row=1, col=1
                        )
            
            # European Markets
            if 'Europe' in self.market_data:
                for i, (name, data) in enumerate(self.market_data['Europe'].items()):
                    if 'Close' in data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=pd.to_datetime(data['date']),
                                y=data['Close'],
                                mode='lines',
                                name=name,
                                line=dict(color=colors[i % len(colors)], width=2)
                            ),
                            row=1, col=2
                        )
            
            # Asian Markets
            if 'Asia' in self.market_data:
                for i, (name, data) in enumerate(self.market_data['Asia'].items()):
                    if 'Close' in data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=pd.to_datetime(data['date']),
                                y=data['Close'],
                                mode='lines',
                                name=name,
                                line=dict(color=colors[i % len(colors)], width=2)
                            ),
                            row=2, col=1
                        )
            
            # Commodities
            if 'Commodities' in self.market_data:
                for i, (name, data) in enumerate(self.market_data['Commodities'].items()):
                    if 'Close' in data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=pd.to_datetime(data['date']),
                                y=data['Close'],
                                mode='lines',
                                name=name,
                                line=dict(color=colors[i % len(colors)], width=2)
                            ),
                            row=2, col=2
                        )
            
            fig.update_layout(
                title=dict(
                    text='Global Markets Overview',
                    font=dict(color='#1A202C', size=18, family='Inter')
                ),
                height=800,
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                font=dict(color='#1A202C', family='Inter'),
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating global markets overview: {e}")
            return None
    
    def create_correlation_heatmap(self, bri_data):
        """
        Create correlation heatmap between BRI and global markets
        
        Args:
            bri_data (pd.DataFrame): BRI data
            
        Returns:
            plotly.graph_objects.Figure: Correlation heatmap
        """
        try:
            # Calculate correlations
            correlations = self.calculate_market_correlations(bri_data)
            
            if not correlations:
                return None
            
            # Prepare data for heatmap
            market_names = []
            correlation_values = []
            
            for region, region_correlations in correlations.items():
                for name, corr_data in region_correlations.items():
                    market_names.append(f"{region}_{name}")
                    correlation_values.append(corr_data['correlation'])
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=[correlation_values],
                x=market_names,
                y=['BRI Correlation'],
                colorscale=[[0, '#E53E3E'], [0.5, '#FFFFFF'], [1, '#38A169']],
                zmin=-1,
                zmax=1,
                text=[[f'{val:.3f}' for val in correlation_values]],
                texttemplate='%{text}',
                textfont={'size': 12, 'color': 'black'},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=dict(
                    text='BRI Correlation with Global Markets',
                    font=dict(color='#1A202C', size=18, family='Inter')
                ),
                height=400,
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                font=dict(color='#1A202C', family='Inter'),
                xaxis=dict(tickangle=45)
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating correlation heatmap: {e}")
            return None
    
    def create_market_performance_comparison(self):
        """
        Create market performance comparison chart
        
        Returns:
            plotly.graph_objects.Figure: Performance comparison
        """
        try:
            if not self.market_data:
                return None
            
            # Calculate performance metrics
            performance_data = []
            
            for region, symbols in self.market_data.items():
                for name, data in symbols.items():
                    if 'Close' in data.columns and len(data) > 1:
                        # Calculate returns
                        returns = data['Close'].pct_change().dropna()
                        
                        # Calculate metrics
                        total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                        volatility = returns.std() * np.sqrt(252) * 100
                        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
                        max_drawdown = self._calculate_max_drawdown(data['Close'])
                        
                        performance_data.append({
                            'Market': f"{region}_{name}",
                            'Total Return (%)': total_return,
                            'Volatility (%)': volatility,
                            'Sharpe Ratio': sharpe_ratio,
                            'Max Drawdown (%)': max_drawdown
                        })
            
            if not performance_data:
                return None
            
            # Create DataFrame
            df = pd.DataFrame(performance_data)
            
            # Create scatter plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['Volatility (%)'],
                y=df['Total Return (%)'],
                mode='markers+text',
                text=df['Market'],
                textposition='top center',
                marker=dict(
                    size=df['Sharpe Ratio'].abs() * 10 + 5,
                    color=df['Max Drawdown (%)'],
                    colorscale='RdYlGn',
                    colorbar=dict(title='Max Drawdown (%)'),
                    line=dict(width=2, color='black')
                ),
                hovertemplate='<b>%{text}</b><br>' +
                            'Total Return: %{y:.2f}%<br>' +
                            'Volatility: %{x:.2f}%<br>' +
                            'Sharpe Ratio: %{marker.size}<br>' +
                            'Max Drawdown: %{marker.color:.2f}%<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(
                    text='Global Markets Performance Comparison',
                    font=dict(color='#1A202C', size=18, family='Inter')
                ),
                xaxis_title='Volatility (%)',
                yaxis_title='Total Return (%)',
                height=600,
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                font=dict(color='#1A202C', family='Inter')
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating performance comparison: {e}")
            return None
    
    def _calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown"""
        try:
            peak = prices.expanding().max()
            drawdown = (prices - peak) / peak
            return drawdown.min() * 100
        except:
            return 0
    
    def get_market_summary(self):
        """
        Get summary of global markets data
        
        Returns:
            dict: Market summary
        """
        try:
            summary = {
                'total_markets': 0,
                'regions': {},
                'last_updated': datetime.now().isoformat()
            }
            
            for region, symbols in self.market_data.items():
                region_summary = {
                    'markets': len(symbols),
                    'symbols': list(symbols.keys()),
                    'data_quality': {}
                }
                
                for name, data in symbols.items():
                    if 'Close' in data.columns:
                        region_summary['data_quality'][name] = {
                            'data_points': len(data),
                            'date_range': f"{data['date'].min()} to {data['date'].max()}" if len(data) > 0 else 'No data',
                            'latest_price': data['Close'].iloc[-1] if len(data) > 0 else None
                        }
                
                summary['regions'][region] = region_summary
                summary['total_markets'] += len(symbols)
            
            return summary
            
        except Exception as e:
            print(f"Error getting market summary: {e}")
            return {'error': str(e)}
    
    def generate_global_markets_report(self, bri_data):
        """
        Generate comprehensive global markets report
        
        Args:
            bri_data (pd.DataFrame): BRI data
            
        Returns:
            dict: Comprehensive report
        """
        try:
            # Fetch all markets data
            all_data = self.fetch_all_markets()
            
            # Calculate correlations
            correlations = self.calculate_market_correlations(bri_data)
            
            # Create visualizations
            overview_chart = self.create_global_markets_overview()
            correlation_heatmap = self.create_correlation_heatmap(bri_data)
            performance_chart = self.create_market_performance_comparison()
            
            # Get summary
            summary = self.get_market_summary()
            
            report = {
                'market_data': all_data,
                'correlations': correlations,
                'visualizations': {
                    'overview_chart': overview_chart.to_dict() if overview_chart else None,
                    'correlation_heatmap': correlation_heatmap.to_dict() if correlation_heatmap else None,
                    'performance_chart': performance_chart.to_dict() if performance_chart else None
                },
                'summary': summary,
                'generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            print(f"Error generating global markets report: {e}")
            return {'error': str(e)}
