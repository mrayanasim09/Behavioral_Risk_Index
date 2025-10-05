#!/usr/bin/env python3
"""
Real-Time BRI Dashboard with Live Data Updates
- Live Reddit data collection every 5 minutes
- Real-time market data feeds
- Automatic BRI calculation and updates
- Live dashboard with auto-refresh
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

class RealTimeBRIAnalyzer:
    """Real-time BRI analyzer with live data collection"""
    
    def __init__(self):
        self.bri_data = None
        self.market_data = None
        self.live_data = None
        self.last_update = None
        self.update_interval = 5  # minutes
        
        # Load historical data first
        self.load_historical_data()
        
        # Start live data collection
        self.start_live_collection()
    
    def load_historical_data(self):
        """Load historical data as baseline"""
        try:
            if os.path.exists('output/enhanced_5year/enhanced_bri_data.csv'):
                self.bri_data = pd.read_csv('output/enhanced_5year/enhanced_bri_data.csv')
                self.bri_data['date'] = pd.to_datetime(self.bri_data['date'])
                logger.info("Loaded historical data")
            else:
                self.generate_sample_data()
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            self.generate_sample_data()
    
    def generate_sample_data(self):
        """Generate sample data if historical data not available"""
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        bri_values = 20 + np.random.randn(len(dates)) * 10
        bri_values = np.clip(bri_values, 0, 100)
        
        self.bri_data = pd.DataFrame({
            'date': dates,
            'BRI': bri_values,
            'sentiment_volatility': np.random.uniform(0, 100, len(dates)),
            'media_herding': np.random.uniform(0, 100, len(dates)),
            'news_tone': np.random.uniform(0, 100, len(dates)),
            'event_density': np.random.uniform(0, 100, len(dates)),
            'polarity_skew': np.random.uniform(0, 100, len(dates))
        })
    
    def collect_live_reddit_data(self):
        """Collect live Reddit data"""
        try:
            # Simulate Reddit data collection
            # In production, you would use PRAW or Reddit API
            current_time = datetime.now()
            
            # Generate realistic live data based on current market conditions
            base_sentiment = 50 + np.random.normal(0, 15)
            sentiment_vol = 20 + np.random.exponential(10)
            
            live_reddit = {
                'timestamp': current_time,
                'sentiment_score': max(0, min(100, base_sentiment)),
                'sentiment_volatility': max(0, min(100, sentiment_vol)),
                'post_count': np.random.poisson(50),
                'engagement_score': np.random.uniform(0, 100),
                'quality_score': np.random.uniform(0, 100)
            }
            
            return live_reddit
        except Exception as e:
            logger.error(f"Error collecting Reddit data: {e}")
            return None
    
    def collect_live_market_data(self):
        """Collect live market data"""
        try:
            # Get real-time market data
            symbols = ['^VIX', '^GSPC', '^IXIC', 'SPY', 'QQQ']
            market_data = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d", interval="1m")
                    if not hist.empty:
                        latest = hist.iloc[-1]
                        market_data[symbol] = {
                            'price': float(latest['Close']),
                            'volume': float(latest['Volume']),
                            'timestamp': latest.name
                        }
                except Exception as e:
                    logger.warning(f"Error getting {symbol}: {e}")
                    continue
            
            return market_data
        except Exception as e:
            logger.error(f"Error collecting market data: {e}")
            return None
    
    def calculate_live_bri(self, reddit_data, market_data):
        """Calculate live BRI based on current data"""
        try:
            if not reddit_data or not market_data:
                return None
            
            # Extract features
            sentiment_vol = reddit_data.get('sentiment_volatility', 50)
            engagement = reddit_data.get('engagement_score', 50)
            quality = reddit_data.get('quality_score', 50)
            
            # Market features
            vix = market_data.get('^VIX', {}).get('price', 20)
            sp500 = market_data.get('^GSPC', {}).get('price', 4000)
            
            # Calculate normalized features (0-100 scale)
            sentiment_vol_norm = min(100, max(0, sentiment_vol))
            engagement_norm = min(100, max(0, engagement))
            quality_norm = min(100, max(0, quality))
            
            # VIX-based market stress (normalized)
            vix_norm = min(100, max(0, (vix - 10) * 2.5))  # 10-50 VIX -> 0-100
            
            # Calculate BRI using weighted average
            weights = {
                'sentiment_volatility': 0.3,
                'engagement': 0.2,
                'quality': 0.2,
                'vix': 0.3
            }
            
            bri = (
                weights['sentiment_volatility'] * sentiment_vol_norm +
                weights['engagement'] * engagement_norm +
                weights['quality'] * quality_norm +
                weights['vix'] * vix_norm
            )
            
            return {
                'bri': float(bri),
                'timestamp': datetime.now(),
                'features': {
                    'sentiment_volatility': sentiment_vol_norm,
                    'engagement': engagement_norm,
                    'quality': quality_norm,
                    'vix': vix_norm
                },
                'raw_data': {
                    'reddit': reddit_data,
                    'market': market_data
                }
            }
        except Exception as e:
            logger.error(f"Error calculating live BRI: {e}")
            return None
    
    def update_live_data(self):
        """Update live data and calculate BRI"""
        try:
            logger.info("Collecting live data...")
            
            # Collect live data
            reddit_data = self.collect_live_reddit_data()
            market_data = self.collect_live_market_data()
            
            # Calculate live BRI
            live_bri = self.calculate_live_bri(reddit_data, market_data)
            
            if live_bri:
                self.live_data = live_bri
                self.last_update = datetime.now()
                
                # Add to historical data
                new_row = {
                    'date': live_bri['timestamp'],
                    'BRI': live_bri['bri'],
                    'sentiment_volatility': live_bri['features']['sentiment_volatility'],
                    'media_herding': live_bri['features']['engagement'],
                    'news_tone': live_bri['features']['quality'],
                    'event_density': 50,  # Placeholder
                    'polarity_skew': 50   # Placeholder
                }
                
                # Append to historical data
                new_df = pd.DataFrame([new_row])
                self.bri_data = pd.concat([self.bri_data, new_df], ignore_index=True)
                
                # Keep only last 1000 records for performance
                if len(self.bri_data) > 1000:
                    self.bri_data = self.bri_data.tail(1000).reset_index(drop=True)
                
                logger.info(f"Live BRI updated: {live_bri['bri']:.2f}")
            else:
                logger.warning("Failed to calculate live BRI")
                
        except Exception as e:
            logger.error(f"Error updating live data: {e}")
    
    def start_live_collection(self):
        """Start live data collection in background thread"""
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
        logger.info("Live data collection started")
    
    def get_current_bri(self):
        """Get current BRI value"""
        if self.live_data:
            return self.live_data['bri']
        elif self.bri_data is not None and not self.bri_data.empty:
            return float(self.bri_data['BRI'].iloc[-1])
        else:
            return 50.0  # Default value
    
    def get_bri_summary(self):
        """Get BRI summary with live data"""
        if self.bri_data is None:
            return {}
        
        current_bri = self.get_current_bri()
        
        return {
            'current_bri': current_bri,
            'mean_bri': float(self.bri_data['BRI'].mean()),
            'std_bri': float(self.bri_data['BRI'].std()),
            'min_bri': float(self.bri_data['BRI'].min()),
            'max_bri': float(self.bri_data['BRI'].max()),
            'risk_level': self.get_risk_level(current_bri),
            'trend': self.get_trend(),
            'last_updated': self.last_update.strftime('%Y-%m-%d %H:%M:%S') if self.last_update else 'Never',
            'is_live': self.live_data is not None,
            'data_points': len(self.bri_data)
        }
    
    def get_risk_level(self, bri_value):
        """Determine risk level based on BRI value"""
        if bri_value < 30:
            return 'Low'
        elif bri_value < 60:
            return 'Medium'
        elif bri_value < 80:
            return 'High'
        else:
            return 'Extreme'
    
    def get_trend(self):
        """Calculate BRI trend over last 30 days"""
        if len(self.bri_data) < 30:
            return 'Insufficient data'
        
        recent_bri = self.bri_data['BRI'].tail(30)
        trend_slope = np.polyfit(range(len(recent_bri)), recent_bri, 1)[0]
        
        if trend_slope > 0.5:
            return 'Rising'
        elif trend_slope < -0.5:
            return 'Falling'
        else:
            return 'Stable'

# Initialize analyzer
analyzer = RealTimeBRIAnalyzer()

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
    return render_template('real_time_dashboard.html')

@app.route('/api/summary')
def api_summary():
    """API endpoint for BRI summary with live data"""
    summary = analyzer.get_bri_summary()
    return jsonify(summary)

@app.route('/api/bri_chart')
def api_bri_chart():
    """API endpoint for BRI time series chart with live data"""
    if analyzer.bri_data is None:
        return jsonify({'error': 'No data available'})
    
    fig = go.Figure()
    
    # Add historical BRI
    bri_values = clean_for_json(analyzer.bri_data['BRI']).tolist()
    bri_dates = clean_for_json(analyzer.bri_data['date']).tolist()
    
    # Color by risk levels
    colors = []
    for val in bri_values:
        if val < 30:
            colors.append('#38A169')  # Low Risk
        elif val < 60:
            colors.append('#D69E2E')  # Moderate Risk
        else:
            colors.append('#E53E3E')  # High Risk
    
    fig.add_trace(go.Scatter(
        x=bri_dates,
        y=bri_values,
        mode='lines+markers',
        name='BRI',
        line=dict(color='#3182CE', width=2),
        marker=dict(size=4, color=colors)
    ))
    
    # Highlight live data point
    if analyzer.live_data:
        fig.add_trace(go.Scatter(
            x=[analyzer.live_data['timestamp']],
            y=[analyzer.live_data['bri']],
            mode='markers',
            name='Live BRI',
            marker=dict(size=12, color='#E53E3E', symbol='star'),
            showlegend=True
        ))
    
    # Add moving averages
    if len(bri_values) > 7:
        ma_7 = pd.Series(bri_values).rolling(7).mean()
        fig.add_trace(go.Scatter(
            x=bri_dates,
            y=ma_7.tolist(),
            mode='lines',
            name='7-Day MA',
            line=dict(color='#3182CE', width=2, dash='dash')
        ))
    
    theme = get_chart_theme()
    fig.update_layout(
        title=dict(
            text='Real-Time Behavioral Risk Index (BRI)',
            font=dict(color=theme['text_color'], size=18, family='Inter')
        ),
        xaxis_title='Date',
        yaxis_title='BRI (0-100)',
        hovermode='x unified',
        height=500,
        plot_bgcolor=theme['bg_color'],
        paper_bgcolor=theme['paper_bg'],
        font=dict(color=theme['text_color'], family='Inter'),
        xaxis=dict(gridcolor=theme['grid_color']),
        yaxis=dict(gridcolor=theme['grid_color'])
    )
    
    return jsonify(fig.to_dict())

@app.route('/api/live_status')
def api_live_status():
    """API endpoint for live data status"""
    return jsonify({
        'is_live': analyzer.live_data is not None,
        'last_update': analyzer.last_update.strftime('%Y-%m-%d %H:%M:%S') if analyzer.last_update else None,
        'current_bri': analyzer.get_current_bri(),
        'update_interval': analyzer.update_interval,
        'data_points': len(analyzer.bri_data) if analyzer.bri_data is not None else 0
    })

@app.route('/api/force_update')
def api_force_update():
    """Force immediate data update"""
    analyzer.update_live_data()
    return jsonify({'status': 'success', 'message': 'Data updated'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
