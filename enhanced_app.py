"""
Enhanced BRI Dashboard with Advanced Analytics
Includes risk heatmaps, volatility clustering, early warning system, and more
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from flask import Flask, render_template, jsonify, request
import json
import os
import sys
from datetime import datetime, timedelta
import logging
from src.advanced_analytics import AdvancedAnalytics
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def clean_for_json(data):
    """Clean data for JSON serialization by removing NaN values"""
    if hasattr(data, 'dropna'):
        return data.dropna()
    elif isinstance(data, (list, tuple)):
        return [x for x in data if not (isinstance(x, float) and np.isnan(x))]
    else:
        return data

def get_chart_theme():
    """Get chart theme colors based on current theme"""
    return {
        'bg_color': '#FFFFFF',
        'grid_color': '#E2E8F0',
        'text_color': '#1A202C',
        'paper_bg': '#FFFFFF'
    }

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedBRIAnalyzer:
    """Enhanced BRI analyzer with advanced analytics"""
    
    def __init__(self):
        self.bri_data = None
        self.market_data = None
        self.advanced_analytics = None
        self.load_data()
    
    def load_data(self):
        """Load BRI and market data"""
        try:
            # Try to load from fast pipeline first
            fast_bri_path = 'output/fast/bri_timeseries.csv'
            fast_market_path = 'output/fast/market_data.csv'
            
            if os.path.exists(fast_bri_path) and os.path.exists(fast_market_path):
                logger.info("Loading fast pipeline data")
                self.bri_data = pd.read_csv(fast_bri_path)
                self.market_data = pd.read_csv(fast_market_path)
            else:
                # Generate sample data
                logger.info("Generating sample data")
                self.generate_sample_data()
            
            # Initialize advanced analytics
            self.advanced_analytics = AdvancedAnalytics(self.bri_data, self.market_data)
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.generate_sample_data()
            self.advanced_analytics = AdvancedAnalytics(self.bri_data, self.market_data)
    
    def generate_sample_data(self):
        """Generate sample BRI and market data"""
        # Generate date range
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2024, 12, 31)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic BRI data with trends and volatility
        np.random.seed(42)
        n_days = len(dates)
        
        # Base trend with seasonal patterns
        trend = np.linspace(35, 45, n_days)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
        noise = np.random.normal(0, 5, n_days)
        
        # Add some crisis periods
        crisis_periods = [
            (datetime(2022, 3, 1), datetime(2022, 4, 30)),  # Market volatility
            (datetime(2022, 9, 1), datetime(2022, 10, 31)),  # Fed rate hikes
            (datetime(2023, 3, 1), datetime(2023, 4, 30)),  # Banking crisis
            (datetime(2023, 10, 1), datetime(2023, 11, 30)), # Geopolitical tensions
        ]
        
        bri_values = trend + seasonal + noise
        
        # Add crisis spikes
        for crisis_start, crisis_end in crisis_periods:
            crisis_mask = (dates >= crisis_start) & (dates <= crisis_end)
            bri_values[crisis_mask] += np.random.normal(15, 5, crisis_mask.sum())
        
        # Ensure values are within 0-100 range
        bri_values = np.clip(bri_values, 0, 100)
        
        # Create BRI dataframe
        self.bri_data = pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'BRI': bri_values
        })
        
        # Generate market data
        vix_base = 20 + 10 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
        vix_noise = np.random.normal(0, 3, n_days)
        vix_values = np.clip(vix_base + vix_noise, 10, 50)
        
        self.market_data = pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'VIX': vix_values,
            'SP500': 4000 + 200 * np.sin(2 * np.pi * np.arange(n_days) / 365.25) + np.random.normal(0, 50, n_days)
        })
    
    def get_current_bri(self):
        """Get current BRI value and risk level"""
        if self.bri_data is None or self.bri_data.empty:
            return {'bri': 0, 'risk_level': 'Unknown'}
        
        current_bri = self.bri_data['BRI'].iloc[-1]
        
        if current_bri < 30:
            risk_level = 'Low'
        elif current_bri < 60:
            risk_level = 'Moderate'
        else:
            risk_level = 'High'
        
        return {'bri': current_bri, 'risk_level': risk_level}
    
    def get_summary_stats(self):
        """Get comprehensive summary statistics"""
        if self.bri_data is None or self.bri_data.empty:
            return {}
        
        bri_stats = self.bri_data['BRI'].describe()
        current = self.get_current_bri()
        
        # Calculate correlation with VIX if available
        correlation = 0
        r_squared = 0
        if self.market_data is not None and not self.market_data.empty:
            # Check if market data has the expected columns
            if 'date' in self.market_data.columns and 'VIX' in self.market_data.columns:
                # Merge data for correlation calculation
                merged = pd.merge(
                    self.bri_data, 
                    self.market_data, 
                    on='date', 
                    how='inner'
                )
                if len(merged) > 1:
                    correlation = merged['BRI'].corr(merged['VIX'])
                    r_squared = correlation ** 2
        
        # Calculate trend
        recent_bri = self.bri_data['BRI'].tail(30)
        if len(recent_bri) > 1:
            trend_slope = np.polyfit(range(len(recent_bri)), recent_bri, 1)[0]
            trend = 'Rising' if trend_slope > 0.1 else 'Falling' if trend_slope < -0.1 else 'Stable'
        else:
            trend = 'Unknown'
        
        return {
            'current_bri': current['bri'],
            'risk_level': current['risk_level'],
            'mean_bri': bri_stats['mean'],
            'std_bri': bri_stats['std'],
            'min_bri': bri_stats['min'],
            'max_bri': bri_stats['max'],
            'correlation': correlation,
            'r_squared': r_squared,
            'trend': trend,
            'data_points': len(self.bri_data),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# Initialize Flask app
app = Flask(__name__)
analyzer = EnhancedBRIAnalyzer()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('enhanced_index.html')

@app.route('/api/summary')
def api_summary():
    """Get summary statistics"""
    try:
        stats = analyzer.get_summary_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error in summary API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/bri_chart')
def api_bri_chart():
    """Get BRI time series chart"""
    try:
        if analyzer.bri_data is None or analyzer.bri_data.empty:
            return jsonify({'error': 'No BRI data available'}), 404
        
        fig = go.Figure()
        
        # Add BRI line with risk-based coloring
        bri_values = clean_for_json(analyzer.bri_data['BRI']).tolist()
        bri_dates = clean_for_json(analyzer.bri_data['date']).tolist()
        
        # Create color array based on risk levels - Professional colors
        colors = []
        for val in bri_values:
            if val < 30:
                colors.append('#38A169')  # Low Risk - Professional Green
            elif val < 60:
                colors.append('#D69E2E')  # Moderate Risk - Professional Yellow
            else:
                colors.append('#E53E3E')  # High Risk - Professional Red
        
        fig.add_trace(go.Scatter(
            x=bri_dates,
            y=bri_values,
            mode='lines+markers',
            name='BRI',
            line=dict(color='#38A169', width=2),
            marker=dict(size=4, color=colors)
        ))
        
        # Add 7-day moving average
        bri_smooth = analyzer.bri_data['BRI'].rolling(window=7, center=True).mean()
        smooth_data = clean_for_json(bri_smooth)
        smooth_dates = analyzer.bri_data['date'].iloc[smooth_data.index]
        
        fig.add_trace(go.Scatter(
            x=clean_for_json(smooth_dates).tolist(),
            y=clean_for_json(smooth_data).tolist(),
            mode='lines',
            name='7-Day MA',
            line=dict(color='#3182CE', width=3)
        ))
        
        # Add risk thresholds
        mean_bri = analyzer.bri_data['BRI'].mean()
        std_bri = analyzer.bri_data['BRI'].std()
        
        fig.add_hline(y=mean_bri, line_dash="dash", line_color="#2D3748", 
                      annotation_text=f"Mean: {mean_bri:.1f}")
        fig.add_hline(y=mean_bri + std_bri, line_dash="dash", line_color="#E53E3E", 
                      annotation_text=f"+1σ: {mean_bri + std_bri:.1f}")
        fig.add_hline(y=mean_bri - std_bri, line_dash="dash", line_color="#E53E3E", 
                      annotation_text=f"-1σ: {mean_bri - std_bri:.1f}")
        
        theme = get_chart_theme()
        fig.update_layout(
            title=dict(
                text='Behavioral Risk Index (BRI) Over Time',
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
    
    except Exception as e:
        logger.error(f"Error in BRI chart API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/risk_heatmap')
def api_risk_heatmap():
    """Get risk heatmap visualization"""
    try:
        if analyzer.advanced_analytics is None:
            return jsonify({'error': 'Advanced analytics not available'}), 404
        
        heatmap = analyzer.advanced_analytics.create_risk_heatmap()
        return jsonify(heatmap.to_dict())
    
    except Exception as e:
        logger.error(f"Error in risk heatmap API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/volatility_clustering')
def api_volatility_clustering():
    """Get volatility clustering analysis"""
    try:
        if analyzer.advanced_analytics is None:
            return jsonify({'error': 'Advanced analytics not available'}), 404
        
        clustering = analyzer.advanced_analytics.identify_volatility_clusters()
        return jsonify({
            'figure': clustering['figure'].to_dict(),
            'cluster_stats': clustering['cluster_stats'].to_dict(),
            'cluster_labels': clustering['cluster_labels']
        })
    
    except Exception as e:
        logger.error(f"Error in volatility clustering API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/early_warning')
def api_early_warning():
    """Get early warning system visualization"""
    try:
        if analyzer.advanced_analytics is None:
            return jsonify({'error': 'Advanced analytics not available'}), 404
        
        warning_system = analyzer.advanced_analytics.create_early_warning_system()
        return jsonify({
            'figure': warning_system['figure'].to_dict(),
            'warning_stats': warning_system['warning_stats'],
            'spike_events': warning_system['spike_events']
        })
    
    except Exception as e:
        logger.error(f"Error in early warning API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/confidence_intervals')
def api_confidence_intervals():
    """Get confidence intervals for predictions"""
    try:
        if analyzer.advanced_analytics is None:
            return jsonify({'error': 'Advanced analytics not available'}), 404
        
        confidence_intervals = analyzer.advanced_analytics.calculate_confidence_intervals()
        return jsonify({
            'figure': confidence_intervals['figure'].to_dict(),
            'predictions': confidence_intervals['predictions'],
            'confidence_intervals': confidence_intervals['confidence_intervals'],
            'prediction_accuracy': confidence_intervals['prediction_accuracy']
        })
    
    except Exception as e:
        logger.error(f"Error in confidence intervals API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics_summary')
def api_analytics_summary():
    """Get comprehensive analytics summary"""
    try:
        if analyzer.advanced_analytics is None:
            return jsonify({'error': 'Advanced analytics not available'}), 404
        
        summary = analyzer.advanced_analytics.generate_analytics_summary()
        return jsonify({
            'risk_heatmap': summary['risk_heatmap'].to_dict(),
            'volatility_clustering': {
                'figure': summary['volatility_clustering']['figure'].to_dict(),
                'cluster_stats': summary['volatility_clustering']['cluster_stats'].to_dict(),
                'cluster_labels': summary['volatility_clustering']['cluster_labels']
            },
            'early_warning': {
                'figure': summary['early_warning']['figure'].to_dict(),
                'warning_stats': summary['early_warning']['warning_stats'],
                'spike_events': summary['early_warning']['spike_events']
            },
            'confidence_intervals': {
                'figure': summary['confidence_intervals']['figure'].to_dict(),
                'predictions': summary['confidence_intervals']['predictions'],
                'confidence_intervals': summary['confidence_intervals']['confidence_intervals'],
                'prediction_accuracy': summary['confidence_intervals']['prediction_accuracy']
            }
        })
    
    except Exception as e:
        logger.error(f"Error in analytics summary API: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
