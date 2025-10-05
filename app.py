#!/usr/bin/env python3
"""
Behavioral Risk Index (BRI) Web Application
Production-ready Flask app for Render deployment
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
from flask import Flask, render_template, jsonify, request
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class BRIAnalyzer:
    """Behavioral Risk Index Analyzer for web application"""
    
    def __init__(self):
        self.bri_data = None
        self.market_data = None
        self.load_data()
    
    def load_data(self):
        """Load BRI and market data"""
        try:
            # Try to load from fast pipeline output first
            if os.path.exists('output/fast/bri_timeseries.csv'):
                self.bri_data = pd.read_csv('output/fast/bri_timeseries.csv')
                self.market_data = pd.read_csv('output/fast/market_data.csv')
                logger.info("Loaded fast pipeline data")
            elif os.path.exists('output/research_grade/bri_timeseries.csv'):
                self.bri_data = pd.read_csv('output/research_grade/bri_timeseries.csv')
                self.market_data = pd.read_csv('output/research_grade/market_data.csv')
                logger.info("Loaded research-grade data")
            elif os.path.exists('output/complete/bri_timeseries.csv'):
                self.bri_data = pd.read_csv('output/complete/bri_timeseries.csv')
                self.market_data = pd.read_csv('output/complete/market_data.csv')
                logger.info("Loaded complete pipeline data")
            else:
                # Generate sample data for demonstration
                self.generate_sample_data()
                logger.info("Generated sample data for demonstration")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.generate_sample_data()
    
    def generate_sample_data(self):
        """Generate sample data for demonstration"""
        # Create date range
        dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
        
        # Generate sample BRI data
        np.random.seed(42)
        bri_values = 20 + np.random.randn(len(dates)) * 10
        bri_values = np.clip(bri_values, 0, 100)  # Ensure 0-100 range
        
        # Add some realistic spikes
        spike_indices = np.random.choice(len(dates), size=20, replace=False)
        bri_values[spike_indices] += np.random.uniform(20, 40, 20)
        bri_values = np.clip(bri_values, 0, 100)
        
        self.bri_data = pd.DataFrame({
            'date': dates,
            'BRI': bri_values,
            'sent_vol_score': np.random.uniform(0, 100, len(dates)),
            'news_tone_score': np.random.uniform(0, 100, len(dates)),
            'herding_score': np.random.uniform(0, 100, len(dates)),
            'polarity_skew_score': np.random.uniform(0, 100, len(dates)),
            'event_density_score': np.random.uniform(0, 100, len(dates))
        })
        
        # Generate sample market data
        vix_values = 20 + np.random.randn(len(dates)) * 5
        vix_values = np.clip(vix_values, 10, 50)
        
        self.market_data = pd.DataFrame({
            'Date': dates,
            'Close_^VIX': vix_values,
            '^GSPC_Close': 4000 + np.cumsum(np.random.randn(len(dates)) * 10)
        })
    
    def get_bri_summary(self):
        """Get BRI summary statistics"""
        if self.bri_data is None:
            return {}
        
        return {
            'current_bri': float(self.bri_data['BRI'].iloc[-1]),
            'mean_bri': float(self.bri_data['BRI'].mean()),
            'std_bri': float(self.bri_data['BRI'].std()),
            'min_bri': float(self.bri_data['BRI'].min()),
            'max_bri': float(self.bri_data['BRI'].max()),
            'risk_level': self.get_risk_level(self.bri_data['BRI'].iloc[-1]),
            'trend': self.get_trend(),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
    
    def get_correlation_data(self):
        """Get BRI-VIX correlation data"""
        if self.bri_data is None or self.market_data is None:
            return {}
        
        # Merge data
        merged = pd.merge(
            self.bri_data, 
            self.market_data, 
            left_on='date', 
            right_on='Date', 
            how='inner'
        )
        
        correlation = merged['BRI'].corr(merged['Close_^VIX'])
        
        return {
            'correlation': float(correlation),
            'r_squared': float(correlation ** 2),
            'data_points': len(merged)
        }
    
    def get_feature_importance(self):
        """Get feature importance data"""
        if self.bri_data is None:
            return {}
        
        feature_cols = ['sent_vol_score', 'news_tone_score', 'herding_score', 
                       'polarity_skew_score', 'event_density_score']
        
        feature_means = {}
        for col in feature_cols:
            feature_means[col.replace('_score', '')] = float(self.bri_data[col].mean())
        
        return feature_means

# Initialize analyzer
analyzer = BRIAnalyzer()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/summary')
def api_summary():
    """API endpoint for BRI summary"""
    summary = analyzer.get_bri_summary()
    correlation_data = analyzer.get_correlation_data()
    summary.update(correlation_data)
    return jsonify(summary)

@app.route('/api/correlation')
def api_correlation():
    """API endpoint for correlation data"""
    return jsonify(analyzer.get_correlation_data())

@app.route('/api/features')
def api_features():
    """API endpoint for feature importance"""
    return jsonify(analyzer.get_feature_importance())

@app.route('/api/bri_chart')
def api_bri_chart():
    """API endpoint for BRI time series chart"""
    if analyzer.bri_data is None:
        return jsonify({'error': 'No data available'})
    
    # Create time series chart
    fig = go.Figure()
    
    # Add BRI line
    fig.add_trace(go.Scatter(
        x=analyzer.bri_data['date'].tolist(),
        y=analyzer.bri_data['BRI'].tolist(),
        mode='lines',
        name='BRI',
        line=dict(color='blue', width=2)
    ))
    
    # Add 7-day moving average
    bri_smooth = analyzer.bri_data['BRI'].rolling(window=7, center=True).mean()
    fig.add_trace(go.Scatter(
        x=analyzer.bri_data['date'].tolist(),
        y=bri_smooth.tolist(),
        mode='lines',
        name='7-Day MA',
        line=dict(color='darkblue', width=3)
    ))
    
    # Add risk thresholds
    mean_bri = analyzer.bri_data['BRI'].mean()
    std_bri = analyzer.bri_data['BRI'].std()
    
    fig.add_hline(y=mean_bri, line_dash="dash", line_color="gray", 
                  annotation_text=f"Mean: {mean_bri:.1f}")
    fig.add_hline(y=mean_bri + std_bri, line_dash="dash", line_color="orange", 
                  annotation_text=f"+1σ: {mean_bri + std_bri:.1f}")
    fig.add_hline(y=mean_bri - std_bri, line_dash="dash", line_color="orange", 
                  annotation_text=f"-1σ: {mean_bri - std_bri:.1f}")
    
    fig.update_layout(
        title='Behavioral Risk Index (BRI) Over Time',
        xaxis_title='Date',
        yaxis_title='BRI (0-100)',
        hovermode='x unified',
        height=500
    )
    
    return jsonify(fig.to_dict())

@app.route('/api/correlation_chart')
def api_correlation_chart():
    """API endpoint for BRI-VIX correlation chart"""
    if analyzer.bri_data is None or analyzer.market_data is None:
        return jsonify({'error': 'No data available'})
    
    # Merge data
    merged = pd.merge(
        analyzer.bri_data, 
        analyzer.market_data, 
        left_on='date', 
        right_on='Date', 
        how='inner'
    )
    
    # Create scatter plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=merged['BRI'].tolist(),
        y=merged['Close_^VIX'].tolist(),
        mode='markers',
        name='BRI vs VIX',
        marker=dict(
            size=8,
            color=list(range(len(merged))),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Time Index")
        )
    ))
    
    # Add regression line
    z = np.polyfit(merged['BRI'], merged['Close_^VIX'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(merged['BRI'].min(), merged['BRI'].max(), 100)
    y_pred = p(x_range)
    
    fig.add_trace(go.Scatter(
        x=x_range.tolist(),
        y=y_pred.tolist(),
        mode='lines',
        name='Regression Line',
        line=dict(color='red', width=3)
    ))
    
    correlation = merged['BRI'].corr(merged['Close_^VIX'])
    
    fig.update_layout(
        title=f'BRI vs VIX Correlation (r = {correlation:.3f})',
        xaxis_title='Behavioral Risk Index (BRI)',
        yaxis_title='VIX (Volatility Index)',
        height=500
    )
    
    return jsonify(fig.to_dict())

@app.route('/api/feature_chart')
def api_feature_chart():
    """API endpoint for feature importance chart"""
    if analyzer.bri_data is None:
        return jsonify({'error': 'No data available'})
    
    feature_cols = ['sent_vol_score', 'news_tone_score', 'herding_score', 
                   'polarity_skew_score', 'event_density_score']
    
    feature_names = [col.replace('_score', '').replace('_', ' ').title() for col in feature_cols]
    feature_values = [analyzer.bri_data[col].mean() for col in feature_cols]
    
    fig = go.Figure(data=[
        go.Bar(
            x=feature_names,
            y=feature_values,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        )
    ])
    
    fig.update_layout(
        title='Feature Importance (Average Scores)',
        xaxis_title='Features',
        yaxis_title='Average Score',
        height=400
    )
    
    return jsonify(fig.to_dict())

@app.route('/api/distribution_chart')
def api_distribution_chart():
    """API endpoint for BRI distribution chart"""
    if analyzer.bri_data is None:
        return jsonify({'error': 'No data available'})
    
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=analyzer.bri_data['BRI'].tolist(),
        nbinsx=30,
        name='BRI Distribution',
        marker_color='green',
        opacity=0.7
    ))
    
    # Add mean line
    mean_bri = analyzer.bri_data['BRI'].mean()
    fig.add_vline(x=mean_bri, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {mean_bri:.1f}")
    
    fig.update_layout(
        title='BRI Distribution',
        xaxis_title='BRI Value',
        yaxis_title='Frequency',
        height=400
    )
    
    return jsonify(fig.to_dict())

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
