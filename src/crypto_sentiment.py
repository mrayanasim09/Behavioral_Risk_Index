"""
Cryptocurrency Sentiment Analysis for BRI Dashboard
Implements Bitcoin, Ethereum, and other crypto sentiment analysis
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import requests
import json
import warnings
warnings.filterwarnings('ignore')

class CryptoSentiment:
    """Cryptocurrency sentiment analysis and data collection"""
    
    def __init__(self):
        self.crypto_symbols = {
            'Bitcoin': 'BTC-USD',
            'Ethereum': 'ETH-USD',
            'Binance Coin': 'BNB-USD',
            'Cardano': 'ADA-USD',
            'Solana': 'SOL-USD',
            'Polkadot': 'DOT-USD',
            'Chainlink': 'LINK-USD',
            'Litecoin': 'LTC-USD',
            'Bitcoin Cash': 'BCH-USD',
            'Stellar': 'XLM-USD'
        }
        
        self.crypto_data = {}
        self.sentiment_data = {}
        
    def fetch_crypto_data(self, symbol, start_date=None, end_date=None):
        """
        Fetch cryptocurrency data
        
        Args:
            symbol (str): Crypto symbol
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: Crypto data
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
            
            # Calculate additional metrics
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=7).std()
            data['Volume_MA'] = data['Volume'].rolling(window=7).mean()
            
            # Select relevant columns
            columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Volatility', 'Volume_MA']
            available_columns = [col for col in columns if col in data.columns]
            data = data[available_columns]
            
            return data
            
        except Exception as e:
            print(f"Error fetching crypto data for {symbol}: {e}")
            return None
    
    def fetch_all_crypto_data(self, start_date=None, end_date=None):
        """
        Fetch data for all cryptocurrencies
        
        Args:
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            
        Returns:
            dict: Crypto data for all symbols
        """
        try:
            all_data = {}
            
            for name, symbol in self.crypto_symbols.items():
                print(f"Fetching {name} ({symbol})...")
                data = self.fetch_crypto_data(symbol, start_date, end_date)
                if data is not None:
                    all_data[name] = data
                    self.crypto_data[name] = data
            
            return all_data
            
        except Exception as e:
            print(f"Error fetching all crypto data: {e}")
            return {}
    
    def calculate_crypto_sentiment_indicators(self):
        """
        Calculate sentiment indicators for cryptocurrencies
        
        Returns:
            dict: Sentiment indicators
        """
        try:
            sentiment_indicators = {}
            
            for name, data in self.crypto_data.items():
                if 'Close' in data.columns and len(data) > 30:
                    # Calculate technical indicators
                    close_prices = data['Close']
                    
                    # RSI (Relative Strength Index)
                    rsi = self._calculate_rsi(close_prices, 14)
                    
                    # MACD (Moving Average Convergence Divergence)
                    macd_line, signal_line, histogram = self._calculate_macd(close_prices)
                    
                    # Bollinger Bands
                    bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close_prices, 20, 2)
                    
                    # Price position relative to Bollinger Bands
                    bb_position = (close_prices - bb_lower) / (bb_upper - bb_lower)
                    
                    # Volume analysis
                    volume_ratio = data['Volume'] / data['Volume_MA'] if 'Volume_MA' in data.columns else pd.Series([1] * len(data))
                    
                    # Sentiment score (0-100)
                    sentiment_score = self._calculate_sentiment_score(
                        rsi, bb_position, volume_ratio, data['Returns'] if 'Returns' in data.columns else pd.Series([0] * len(data))
                    )
                    
                    sentiment_indicators[name] = {
                        'rsi': rsi.iloc[-1] if len(rsi) > 0 else 50,
                        'macd': macd_line.iloc[-1] if len(macd_line) > 0 else 0,
                        'bb_position': bb_position.iloc[-1] if len(bb_position) > 0 else 0.5,
                        'volume_ratio': volume_ratio.iloc[-1] if len(volume_ratio) > 0 else 1,
                        'sentiment_score': sentiment_score.iloc[-1] if len(sentiment_score) > 0 else 50,
                        'price_change_24h': data['Returns'].iloc[-1] * 100 if 'Returns' in data.columns else 0,
                        'volatility': data['Volatility'].iloc[-1] * 100 if 'Volatility' in data.columns else 0
                    }
            
            self.sentiment_data = sentiment_indicators
            return sentiment_indicators
            
        except Exception as e:
            print(f"Error calculating sentiment indicators: {e}")
            return {}
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50] * len(prices))
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        except:
            return pd.Series([0] * len(prices)), pd.Series([0] * len(prices)), pd.Series([0] * len(prices))
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        try:
            rolling_mean = prices.rolling(window=window).mean()
            rolling_std = prices.rolling(window=window).std()
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            return upper_band, rolling_mean, lower_band
        except:
            return prices, prices, prices
    
    def _calculate_sentiment_score(self, rsi, bb_position, volume_ratio, returns):
        """Calculate overall sentiment score"""
        try:
            # RSI component (0-100)
            rsi_score = 100 - rsi  # Invert RSI (high RSI = overbought = negative sentiment)
            
            # Bollinger Bands position (0-100)
            bb_score = bb_position * 100
            
            # Volume component (0-100)
            volume_score = np.clip(volume_ratio * 50, 0, 100)
            
            # Returns component (0-100)
            returns_score = np.clip((returns * 1000) + 50, 0, 100)
            
            # Weighted average
            sentiment_score = (
                rsi_score * 0.3 +
                bb_score * 0.25 +
                volume_score * 0.2 +
                returns_score * 0.25
            )
            
            return sentiment_score
        except:
            return pd.Series([50] * len(rsi))
    
    def create_crypto_sentiment_dashboard(self):
        """
        Create cryptocurrency sentiment dashboard
        
        Returns:
            plotly.graph_objects.Figure: Crypto sentiment dashboard
        """
        try:
            if not self.sentiment_data:
                self.calculate_crypto_sentiment_indicators()
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Crypto Sentiment Scores', 'Price Changes (24h)', 'Volatility Analysis', 'RSI Distribution'),
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # Prepare data
            crypto_names = list(self.sentiment_data.keys())
            sentiment_scores = [self.sentiment_data[name]['sentiment_score'] for name in crypto_names]
            price_changes = [self.sentiment_data[name]['price_change_24h'] for name in crypto_names]
            volatilities = [self.sentiment_data[name]['volatility'] for name in crypto_names]
            rsi_values = [self.sentiment_data[name]['rsi'] for name in crypto_names]
            
            # Sentiment scores
            fig.add_trace(
                go.Bar(
                    x=crypto_names,
                    y=sentiment_scores,
                    name='Sentiment Score',
                    marker_color=['#38A169' if score > 60 else '#D69E2E' if score > 40 else '#E53E3E' for score in sentiment_scores]
                ),
                row=1, col=1
            )
            
            # Price changes
            fig.add_trace(
                go.Bar(
                    x=crypto_names,
                    y=price_changes,
                    name='Price Change (%)',
                    marker_color=['#38A169' if change > 0 else '#E53E3E' for change in price_changes]
                ),
                row=1, col=2
            )
            
            # Volatility
            fig.add_trace(
                go.Bar(
                    x=crypto_names,
                    y=volatilities,
                    name='Volatility (%)',
                    marker_color='#3182CE'
                ),
                row=2, col=1
            )
            
            # RSI distribution
            fig.add_trace(
                go.Bar(
                    x=crypto_names,
                    y=rsi_values,
                    name='RSI',
                    marker_color=['#E53E3E' if rsi > 70 else '#38A169' if rsi < 30 else '#D69E2E' for rsi in rsi_values]
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title=dict(
                    text='Cryptocurrency Sentiment Dashboard',
                    font=dict(color='#1A202C', size=18, family='Inter')
                ),
                height=800,
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                font=dict(color='#1A202C', family='Inter'),
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating crypto sentiment dashboard: {e}")
            return None
    
    def create_crypto_correlation_analysis(self, bri_data):
        """
        Create correlation analysis between BRI and crypto sentiment
        
        Args:
            bri_data (pd.DataFrame): BRI data
            
        Returns:
            plotly.graph_objects.Figure: Correlation analysis
        """
        try:
            if not self.crypto_data:
                return None
            
            # Calculate correlations
            correlations = {}
            
            for name, data in self.crypto_data.items():
                if 'Close' in data.columns:
                    # Merge BRI and crypto data
                    merged = pd.merge(
                        bri_data, 
                        data, 
                        on='date', 
                        how='inner'
                    )
                    
                    if len(merged) > 10:
                        # Calculate correlation with BRI
                        correlation = merged['BRI'].corr(merged['Close'])
                        correlations[name] = correlation
            
            if not correlations:
                return None
            
            # Create correlation chart
            crypto_names = list(correlations.keys())
            correlation_values = list(correlations.values())
            
            fig = go.Figure(data=go.Bar(
                x=crypto_names,
                y=correlation_values,
                marker_color=['#38A169' if corr > 0 else '#E53E3E' for corr in correlation_values],
                text=[f'{corr:.3f}' for corr in correlation_values],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=dict(
                    text='BRI Correlation with Cryptocurrency Prices',
                    font=dict(color='#1A202C', size=18, family='Inter')
                ),
                xaxis_title='Cryptocurrency',
                yaxis_title='Correlation with BRI',
                height=500,
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                font=dict(color='#1A202C', family='Inter'),
                xaxis=dict(tickangle=45)
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating crypto correlation analysis: {e}")
            return None
    
    def create_crypto_volatility_analysis(self):
        """
        Create cryptocurrency volatility analysis
        
        Returns:
            plotly.graph_objects.Figure: Volatility analysis
        """
        try:
            if not self.crypto_data:
                return None
            
            # Create volatility comparison
            fig = go.Figure()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            for i, (name, data) in enumerate(self.crypto_data.items()):
                if 'Volatility' in data.columns:
                    fig.add_trace(go.Scatter(
                        x=pd.to_datetime(data['date']),
                        y=data['Volatility'] * 100,
                        mode='lines',
                        name=name,
                        line=dict(color=colors[i % len(colors)], width=2)
                    ))
            
            fig.update_layout(
                title=dict(
                    text='Cryptocurrency Volatility Comparison',
                    font=dict(color='#1A202C', size=18, family='Inter')
                ),
                xaxis_title='Date',
                yaxis_title='Volatility (%)',
                height=600,
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                font=dict(color='#1A202C', family='Inter'),
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating crypto volatility analysis: {e}")
            return None
    
    def get_crypto_sentiment_summary(self):
        """
        Get summary of crypto sentiment data
        
        Returns:
            dict: Crypto sentiment summary
        """
        try:
            if not self.sentiment_data:
                self.calculate_crypto_sentiment_indicators()
            
            summary = {
                'total_cryptos': len(self.sentiment_data),
                'average_sentiment': np.mean([data['sentiment_score'] for data in self.sentiment_data.values()]),
                'bullish_cryptos': len([data for data in self.sentiment_data.values() if data['sentiment_score'] > 60]),
                'bearish_cryptos': len([data for data in self.sentiment_data.values() if data['sentiment_score'] < 40]),
                'neutral_cryptos': len([data for data in self.sentiment_data.values() if 40 <= data['sentiment_score'] <= 60]),
                'top_performer': max(self.sentiment_data.items(), key=lambda x: x[1]['price_change_24h'])[0],
                'most_volatile': max(self.sentiment_data.items(), key=lambda x: x[1]['volatility'])[0],
                'last_updated': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            print(f"Error getting crypto sentiment summary: {e}")
            return {'error': str(e)}
    
    def generate_crypto_sentiment_report(self, bri_data):
        """
        Generate comprehensive crypto sentiment report
        
        Args:
            bri_data (pd.DataFrame): BRI data
            
        Returns:
            dict: Comprehensive report
        """
        try:
            # Fetch all crypto data
            all_data = self.fetch_all_crypto_data()
            
            # Calculate sentiment indicators
            sentiment_indicators = self.calculate_crypto_sentiment_indicators()
            
            # Create visualizations
            sentiment_dashboard = self.create_crypto_sentiment_dashboard()
            correlation_analysis = self.create_crypto_correlation_analysis(bri_data)
            volatility_analysis = self.create_crypto_volatility_analysis()
            
            # Get summary
            summary = self.get_crypto_sentiment_summary()
            
            report = {
                'crypto_data': all_data,
                'sentiment_indicators': sentiment_indicators,
                'visualizations': {
                    'sentiment_dashboard': sentiment_dashboard.to_dict() if sentiment_dashboard else None,
                    'correlation_analysis': correlation_analysis.to_dict() if correlation_analysis else None,
                    'volatility_analysis': volatility_analysis.to_dict() if volatility_analysis else None
                },
                'summary': summary,
                'generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            print(f"Error generating crypto sentiment report: {e}")
            return {'error': str(e)}
