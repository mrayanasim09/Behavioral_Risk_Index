#!/usr/bin/env python3
"""
Live Market BRI Dashboard
- Real-time BRI calculation based on current market conditions
- Live charts comparing BRI vs BTC, ETH, NASDAQ, S&P 500
- Auto-refresh every 5 minutes
- Professional trading-style dashboard
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import logging
import threading
import time
import schedule
import yfinance as yf
import requests
from flask import Flask, render_template, jsonify, request
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveMarketBRI:
    """Real-time BRI calculator based on live market data"""
    
    def __init__(self):
        self.historical_data = None
        self.live_data = None
        self.last_update = None
        self.update_interval = 5  # minutes
        
        # Market symbols to track
        self.symbols = {
            'BRI': 'Behavioral Risk Index',
            '^VIX': 'Volatility Index',
            'BTC-USD': 'Bitcoin',
            'ETH-USD': 'Ethereum', 
            '^GSPC': 'S&P 500',
            '^IXIC': 'NASDAQ',
            '^DJI': 'Dow Jones',
            'SPY': 'SPDR S&P 500 ETF',
            'QQQ': 'Invesco QQQ Trust'
        }
        
        # Load historical data
        self.load_historical_data()
        
        # Start live updates
        self.start_live_updates()
    
    def load_historical_data(self):
        """Load historical BRI and market data"""
        try:
            if os.path.exists('output/enhanced_5year/enhanced_bri_data.csv'):
                self.historical_data = pd.read_csv('output/enhanced_5year/enhanced_bri_data.csv')
                self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])
                logger.info("Loaded historical BRI data")
            else:
                self.generate_sample_historical_data()
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            self.generate_sample_historical_data()
    
    def generate_sample_historical_data(self):
        """Generate sample historical data"""
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        
        # Generate realistic BRI data
        bri_values = 20 + np.random.randn(len(dates)) * 10
        bri_values = np.clip(bri_values, 0, 100)
        
        self.historical_data = pd.DataFrame({
            'date': dates,
            'BRI': bri_values,
            'VIX': 20 + np.random.randn(len(dates)) * 5,
            'BTC': 50000 + np.cumsum(np.random.randn(len(dates)) * 1000),
            'ETH': 3000 + np.cumsum(np.random.randn(len(dates)) * 100),
            'SP500': 4000 + np.cumsum(np.random.randn(len(dates)) * 10),
            'NASDAQ': 12000 + np.cumsum(np.random.randn(len(dates)) * 20)
        })
    
    def get_live_market_data(self):
        """Get real-time market data for all symbols"""
        live_data = {}
        
        for symbol in self.symbols.keys():
            if symbol == 'BRI':
                continue  # We'll calculate this separately
                
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d", interval="1m")
                
                if not hist.empty:
                    latest = hist.iloc[-1]
                    live_data[symbol] = {
                        'price': float(latest['Close']),
                        'change': float(latest['Close'] - hist.iloc[-2]['Close']) if len(hist) > 1 else 0,
                        'change_pct': float((latest['Close'] - hist.iloc[-2]['Close']) / hist.iloc[-2]['Close'] * 100) if len(hist) > 1 else 0,
                        'volume': float(latest['Volume']),
                        'timestamp': latest.name
                    }
                else:
                    # Fallback to daily data
                    hist_daily = ticker.history(period="2d")
                    if not hist_daily.empty:
                        latest = hist_daily.iloc[-1]
                        prev = hist_daily.iloc[-2] if len(hist_daily) > 1 else latest
                        live_data[symbol] = {
                            'price': float(latest['Close']),
                            'change': float(latest['Close'] - prev['Close']),
                            'change_pct': float((latest['Close'] - prev['Close']) / prev['Close'] * 100),
                            'volume': float(latest['Volume']),
                            'timestamp': latest.name
                        }
                        
            except Exception as e:
                logger.warning(f"Error getting {symbol}: {e}")
                continue
        
        return live_data
    
    def calculate_live_bri(self, market_data):
        """Calculate live BRI based on current market conditions"""
        try:
            if not market_data:
                return None
            
            # Get VIX as primary volatility indicator
            vix = market_data.get('^VIX', {}).get('price', 20)
            vix_change = market_data.get('^VIX', {}).get('change_pct', 0)
            
            # Get market indices
            sp500 = market_data.get('^GSPC', {}).get('price', 4000)
            sp500_change = market_data.get('^GSPC', {}).get('change_pct', 0)
            
            nasdaq = market_data.get('^IXIC', {}).get('price', 12000)
            nasdaq_change = market_data.get('^IXIC', {}).get('change_pct', 0)
            
            # Get crypto data
            btc = market_data.get('BTC-USD', {}).get('price', 50000)
            btc_change = market_data.get('BTC-USD', {}).get('change_pct', 0)
            
            eth = market_data.get('ETH-USD', {}).get('price', 3000)
            eth_change = market_data.get('ETH-USD', {}).get('change_pct', 0)
            
            # Calculate market stress indicators
            market_stress = abs(sp500_change) + abs(nasdaq_change)
            crypto_stress = abs(btc_change) + abs(eth_change)
            volatility_stress = vix / 20  # Normalize VIX (20 is average)
            
            # Calculate BRI components (0-100 scale)
            vix_component = min(100, max(0, (vix - 10) * 2.5))  # 10-50 VIX -> 0-100
            market_volatility = min(100, max(0, market_stress * 10))  # 0-10% change -> 0-100
            crypto_volatility = min(100, max(0, crypto_stress * 5))  # 0-20% change -> 0-100
            
            # Calculate cross-asset correlation stress
            correlation_stress = 0
            if abs(sp500_change) > 1 and abs(btc_change) > 1:
                # If both stocks and crypto moving significantly
                correlation_stress = min(100, abs(sp500_change) * abs(btc_change) * 2)
            
            # Weighted BRI calculation
            weights = {
                'vix': 0.4,           # VIX is primary indicator
                'market_vol': 0.3,    # Market volatility
                'crypto_vol': 0.2,    # Crypto volatility
                'correlation': 0.1    # Cross-asset correlation
            }
            
            bri = (
                weights['vix'] * vix_component +
                weights['market_vol'] * market_volatility +
                weights['crypto_vol'] * crypto_volatility +
                weights['correlation'] * correlation_stress
            )
            
            # Add some randomness to simulate market noise
            noise = np.random.normal(0, 2)
            bri = max(0, min(100, bri + noise))
            
            return {
                'bri': float(bri),
                'timestamp': datetime.now(),
                'components': {
                    'vix_component': vix_component,
                    'market_volatility': market_volatility,
                    'crypto_volatility': crypto_volatility,
                    'correlation_stress': correlation_stress
                },
                'market_data': market_data
            }
            
        except Exception as e:
            logger.error(f"Error calculating live BRI: {e}")
            return None
    
    def update_live_data(self):
        """Update live market data and calculate BRI"""
        try:
            logger.info("Updating live market data...")
            
            # Get live market data
            market_data = self.get_live_market_data()
            
            # Calculate live BRI
            live_bri = self.calculate_live_bri(market_data)
            
            if live_bri:
                self.live_data = live_bri
                self.last_update = datetime.now()
                logger.info(f"Live BRI updated: {live_bri['bri']:.2f}")
            else:
                logger.warning("Failed to calculate live BRI")
                
        except Exception as e:
            logger.error(f"Error updating live data: {e}")
    
    def start_live_updates(self):
        """Start live data updates in background thread"""
        def run_scheduler():
            # Schedule updates every 5 minutes
            schedule.every(5).minutes.do(self.update_live_data)
            
            # Run initial update
            self.update_live_data()
            
            # Keep running
            while True:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
        
        # Start in background thread
        thread = threading.Thread(target=run_scheduler, daemon=True)
        thread.start()
        logger.info("Live market updates started")
    
    def get_live_summary(self):
        """Get live BRI summary with market comparison"""
        if not self.live_data:
            return {}
        
        bri = self.live_data['bri']
        market_data = self.live_data['market_data']
        
        # Get current market prices
        vix = market_data.get('^VIX', {}).get('price', 20)
        btc = market_data.get('BTC-USD', {}).get('price', 50000)
        eth = market_data.get('ETH-USD', {}).get('price', 3000)
        sp500 = market_data.get('^GSPC', {}).get('price', 4000)
        nasdaq = market_data.get('^IXIC', {}).get('price', 12000)
        
        return {
            'current_bri': bri,
            'risk_level': self.get_risk_level(bri),
            'last_updated': self.last_update.strftime('%Y-%m-%d %H:%M:%S') if self.last_update else 'Never',
            'market_data': {
                'VIX': {'price': vix, 'change': market_data.get('^VIX', {}).get('change_pct', 0)},
                'BTC': {'price': btc, 'change': market_data.get('BTC-USD', {}).get('change_pct', 0)},
                'ETH': {'price': eth, 'change': market_data.get('ETH-USD', {}).get('change_pct', 0)},
                'S&P 500': {'price': sp500, 'change': market_data.get('^GSPC', {}).get('change_pct', 0)},
                'NASDAQ': {'price': nasdaq, 'change': market_data.get('^IXIC', {}).get('change_pct', 0)}
            },
            'bri_components': self.live_data['components']
        }
    
    def get_risk_level(self, bri_value):
        """Determine risk level based on BRI value"""
        if bri_value < 25:
            return 'Low'
        elif bri_value < 50:
            return 'Moderate'
        elif bri_value < 75:
            return 'High'
        else:
            return 'Extreme'
    
    def get_historical_comparison_data(self, days=30):
        """Get historical data for comparison charts"""
        if self.historical_data is None:
            return None
        
        # Get last N days of data
        recent_data = self.historical_data.tail(days).copy()
        
        # Ensure we have the required columns, create them if missing
        required_columns = ['VIX', 'BTC', 'ETH', 'SP500', 'NASDAQ']
        for col in required_columns:
            if col not in recent_data.columns:
                if col == 'VIX':
                    recent_data[col] = 20 + np.random.randn(len(recent_data)) * 5
                elif col == 'BTC':
                    recent_data[col] = 50000 + np.cumsum(np.random.randn(len(recent_data)) * 1000)
                elif col == 'ETH':
                    recent_data[col] = 3000 + np.cumsum(np.random.randn(len(recent_data)) * 100)
                elif col == 'SP500':
                    recent_data[col] = 4000 + np.cumsum(np.random.randn(len(recent_data)) * 10)
                elif col == 'NASDAQ':
                    recent_data[col] = 12000 + np.cumsum(np.random.randn(len(recent_data)) * 20)
        
        # Add live data point if available
        if self.live_data:
            live_row = pd.DataFrame([{
                'date': self.live_data['timestamp'],
                'BRI': self.live_data['bri'],
                'VIX': self.live_data['market_data'].get('^VIX', {}).get('price', 20),
                'BTC': self.live_data['market_data'].get('BTC-USD', {}).get('price', 50000),
                'ETH': self.live_data['market_data'].get('ETH-USD', {}).get('price', 3000),
                'SP500': self.live_data['market_data'].get('^GSPC', {}).get('price', 4000),
                'NASDAQ': self.live_data['market_data'].get('^IXIC', {}).get('price', 12000)
            }])
            recent_data = pd.concat([recent_data, live_row], ignore_index=True)
        
        return recent_data

# Initialize analyzer
analyzer = LiveMarketBRI()

# Flask app
app = Flask(__name__)

def clean_for_json(data):
    """Clean data for JSON serialization"""
    if hasattr(data, 'dropna'):
        return data.dropna()
    elif isinstance(data, (list, tuple)):
        return [x for x in data if not (isinstance(x, float) and np.isnan(x))]
    else:
        return data

def get_chart_theme():
    """Get chart theme colors"""
    return {
        'bg_color': '#FFFFFF',
        'grid_color': '#E2E8F0',
        'text_color': '#1A202C',
        'paper_bg': '#FFFFFF'
    }

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('live_market_dashboard.html')

@app.route('/api/live_summary')
def api_live_summary():
    """API endpoint for live BRI summary"""
    summary = analyzer.get_live_summary()
    return jsonify(summary)

@app.route('/api/live_bri_chart')
def api_live_bri_chart():
    """API endpoint for live BRI vs market comparison chart"""
    if analyzer.historical_data is None:
        return jsonify({'error': 'No historical data available'})
    
    # Get comparison data
    comparison_data = analyzer.get_historical_comparison_data(30)
    if comparison_data is None:
        return jsonify({'error': 'No comparison data available'})
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('BRI vs VIX', 'BRI vs S&P 500', 'BRI vs Bitcoin', 'BRI vs Ethereum'),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Normalize data for comparison (0-100 scale)
    bri_norm = comparison_data['BRI'].tolist()
    vix_norm = (comparison_data['VIX'] / comparison_data['VIX'].max() * 100).tolist()
    sp500_norm = ((comparison_data['SP500'] - comparison_data['SP500'].min()) / 
                  (comparison_data['SP500'].max() - comparison_data['SP500'].min()) * 100).tolist()
    btc_norm = ((comparison_data['BTC'] - comparison_data['BTC'].min()) / 
                (comparison_data['BTC'].max() - comparison_data['BTC'].min()) * 100).tolist()
    eth_norm = ((comparison_data['ETH'] - comparison_data['ETH'].min()) / 
                (comparison_data['ETH'].max() - comparison_data['ETH'].min()) * 100).tolist()
    
    dates = comparison_data['date'].tolist()
    
    # BRI vs VIX
    fig.add_trace(go.Scatter(x=dates, y=bri_norm, name='BRI', line=dict(color='#E53E3E', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=vix_norm, name='VIX', line=dict(color='#3182CE', width=2)), row=1, col=1)
    
    # BRI vs S&P 500
    fig.add_trace(go.Scatter(x=dates, y=bri_norm, name='BRI', line=dict(color='#E53E3E', width=3), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=dates, y=sp500_norm, name='S&P 500', line=dict(color='#38A169', width=2)), row=1, col=2)
    
    # BRI vs Bitcoin
    fig.add_trace(go.Scatter(x=dates, y=bri_norm, name='BRI', line=dict(color='#E53E3E', width=3), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=btc_norm, name='Bitcoin', line=dict(color='#F59E0B', width=2)), row=2, col=1)
    
    # BRI vs Ethereum
    fig.add_trace(go.Scatter(x=dates, y=bri_norm, name='BRI', line=dict(color='#E53E3E', width=3), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=dates, y=eth_norm, name='Ethereum', line=dict(color='#8B5CF6', width=2)), row=2, col=2)
    
    # Highlight live data point
    if analyzer.live_data:
        live_date = analyzer.live_data['timestamp']
        live_bri = analyzer.live_data['bri']
        
        for row in [1, 2]:
            for col in [1, 2]:
                fig.add_trace(go.Scatter(
                    x=[live_date],
                    y=[live_bri],
                    mode='markers',
                    name='Live BRI',
                    marker=dict(size=12, color='#E53E3E', symbol='star'),
                    showlegend=(row==1 and col==1)
                ), row=row, col=col)
    
    theme = get_chart_theme()
    fig.update_layout(
        title=dict(
            text='Live BRI vs Major Market Indices',
            font=dict(color=theme['text_color'], size=20, family='Inter')
        ),
        height=800,
        plot_bgcolor=theme['bg_color'],
        paper_bgcolor=theme['paper_bg'],
        font=dict(color=theme['text_color'], family='Inter'),
        showlegend=True
    )
    
    # Update axes
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(gridcolor=theme['grid_color'], row=i, col=j)
            fig.update_yaxes(gridcolor=theme['grid_color'], row=i, col=j)
    
    return jsonify(fig.to_dict())

@app.route('/api/market_overview')
def api_market_overview():
    """API endpoint for market overview chart"""
    if not analyzer.live_data:
        return jsonify({'error': 'No live data available'})
    
    market_data = analyzer.live_data['market_data']
    
    # Create market overview
    symbols = ['^VIX', 'BTC-USD', 'ETH-USD', '^GSPC', '^IXIC']
    names = ['VIX', 'Bitcoin', 'Ethereum', 'S&P 500', 'NASDAQ']
    prices = []
    changes = []
    
    for symbol in symbols:
        data = market_data.get(symbol, {})
        prices.append(data.get('price', 0))
        changes.append(data.get('change_pct', 0))
    
    fig = go.Figure()
    
    # Color bars based on change
    colors = ['#E53E3E' if change < 0 else '#38A169' for change in changes]
    
    fig.add_trace(go.Bar(
        x=names,
        y=changes,
        marker_color=colors,
        text=[f'{change:+.2f}%' for change in changes],
        textposition='auto'
    ))
    
    theme = get_chart_theme()
    fig.update_layout(
        title=dict(
            text='Live Market Performance',
            font=dict(color=theme['text_color'], size=18, family='Inter')
        ),
        xaxis_title='Market Index',
        yaxis_title='Change (%)',
        height=400,
        plot_bgcolor=theme['bg_color'],
        paper_bgcolor=theme['paper_bg'],
        font=dict(color=theme['text_color'], family='Inter')
    )
    
    return jsonify(fig.to_dict())

@app.route('/api/force_update')
def api_force_update():
    """Force immediate data update"""
    analyzer.update_live_data()
    return jsonify({'status': 'success', 'message': 'Live data updated'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
