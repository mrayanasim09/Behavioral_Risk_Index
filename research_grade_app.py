"""
Research-Grade BRI Dashboard with Complete Statistical Validation
Includes all advanced features: statistical rigor, volatility analysis, feature validation, and backtesting
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from flask import Flask, render_template, jsonify, request, send_file
import json
import os
import sys
from datetime import datetime, timedelta
import logging
import base64
import io
from src.advanced_analytics import AdvancedAnalytics
from src.advanced_charts import AdvancedCharts
from src.export_utils import ExportUtils
from src.forecasting_models import ForecastingModels
from src.monte_carlo_simulations import MonteCarloSimulations
from src.statistical_validation import StatisticalValidation
from src.advanced_backtesting import AdvancedBacktesting
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

class ResearchGradeBRIAnalyzer:
    """Research-grade BRI analyzer with complete statistical validation"""
    
    def __init__(self):
        self.bri_data = None
        self.market_data = None
        self.advanced_analytics = None
        self.advanced_charts = None
        self.export_utils = None
        self.forecasting_models = None
        self.monte_carlo = None
        self.statistical_validation = None
        self.advanced_backtesting = None
        self.chart_cache = {}
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
            
            # Initialize all modules
            self.advanced_analytics = AdvancedAnalytics(self.bri_data, self.market_data)
            self.advanced_charts = AdvancedCharts(self.bri_data, self.market_data)
            self.export_utils = ExportUtils(self.bri_data, self.market_data)
            self.forecasting_models = ForecastingModels(self.bri_data, self.market_data)
            self.monte_carlo = MonteCarloSimulations(self.bri_data, self.market_data)
            self.statistical_validation = StatisticalValidation(self.bri_data, self.market_data)
            self.advanced_backtesting = AdvancedBacktesting(self.bri_data, self.market_data)
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.generate_sample_data()
            self.advanced_analytics = AdvancedAnalytics(self.bri_data, self.market_data)
            self.advanced_charts = AdvancedCharts(self.bri_data, self.market_data)
            self.export_utils = ExportUtils(self.bri_data, self.market_data)
            self.forecasting_models = ForecastingModels(self.bri_data, self.market_data)
            self.monte_carlo = MonteCarloSimulations(self.bri_data, self.market_data)
            self.statistical_validation = StatisticalValidation(self.bri_data, self.market_data)
            self.advanced_backtesting = AdvancedBacktesting(self.bri_data, self.market_data)
    
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
        
        # Generate component scores
        sent_vol_score = np.random.uniform(20, 80, n_days)
        news_tone_score = np.random.uniform(30, 70, n_days)
        herding_score = np.random.uniform(25, 75, n_days)
        polarity_skew_score = np.random.uniform(10, 90, n_days)
        event_density_score = np.random.uniform(15, 85, n_days)
        
        # Create BRI dataframe
        self.bri_data = pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'BRI': bri_values,
            'sent_vol_score': sent_vol_score,
            'news_tone_score': news_tone_score,
            'herding_score': herding_score,
            'polarity_skew_score': polarity_skew_score,
            'event_density_score': event_density_score
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
    
    def get_cached_chart(self, chart_type):
        """Get cached chart or create new one"""
        if chart_type in self.chart_cache:
            return self.chart_cache[chart_type]
        
        try:
            if chart_type == 'bri_chart':
                chart = self.create_bri_chart()
            elif chart_type == 'candlestick':
                chart = self.advanced_charts.create_candlestick_chart()
            elif chart_type == 'box_plots':
                chart = self.advanced_charts.create_box_plot_analysis()
            elif chart_type == 'correlation_heatmap':
                chart = self.advanced_charts.create_correlation_heatmap()
            elif chart_type == 'violin_plots':
                chart = self.advanced_charts.create_violin_plot_analysis()
            elif chart_type == '3d_surface':
                chart = self.advanced_charts.create_3d_surface_plot()
            elif chart_type == 'forecasting_comparison':
                chart = self.forecasting_models.create_forecasting_comparison()
            elif chart_type == 'model_performance':
                chart = self.forecasting_models.create_model_performance_comparison()
            elif chart_type == 'monte_carlo_simulation':
                chart = self.monte_carlo.create_simulation_visualization()
            elif chart_type == 'risk_metrics':
                chart = self.monte_carlo.create_risk_metrics_chart()
            elif chart_type == 'stress_test':
                chart = self.monte_carlo.create_stress_test_chart()
            else:
                return None
            
            # Cache the chart
            self.chart_cache[chart_type] = chart.to_dict() if chart else None
            return self.chart_cache[chart_type]
            
        except Exception as e:
            logger.error(f"Error creating chart {chart_type}: {e}")
            return None
    
    def create_bri_chart(self):
        """Create optimized BRI chart"""
        try:
            fig = go.Figure()
            
            # Add BRI line with risk-based coloring
            bri_values = clean_for_json(self.bri_data['BRI']).tolist()
            bri_dates = clean_for_json(self.bri_data['date']).tolist()
            
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
            bri_smooth = self.bri_data['BRI'].rolling(window=7, center=True).mean()
            smooth_data = clean_for_json(bri_smooth)
            smooth_dates = self.bri_data['date'].iloc[smooth_data.index]
            
            fig.add_trace(go.Scatter(
                x=clean_for_json(smooth_dates).tolist(),
                y=clean_for_json(smooth_data).tolist(),
                mode='lines',
                name='7-Day MA',
                line=dict(color='#3182CE', width=3)
            ))
            
            # Add risk thresholds
            mean_bri = self.bri_data['BRI'].mean()
            std_bri = self.bri_data['BRI'].std()
            
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
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating BRI chart: {e}")
            return None

# Initialize Flask app
app = Flask(__name__)
analyzer = ResearchGradeBRIAnalyzer()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('ultimate_index.html')

@app.route('/api/summary')
def api_summary():
    """Get summary statistics"""
    try:
        stats = analyzer.get_summary_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error in summary API: {e}")
        return jsonify({'error': str(e)}), 500

# Statistical Validation Endpoints
@app.route('/api/out_of_sample_testing')
def api_out_of_sample_testing():
    """Get out-of-sample testing results"""
    try:
        if analyzer.statistical_validation is None:
            return jsonify({'error': 'Statistical validation not available'}), 404
        
        results = analyzer.statistical_validation.out_of_sample_testing()
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in out-of-sample testing API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/regime_detection')
def api_regime_detection():
    """Get regime detection analysis"""
    try:
        if analyzer.statistical_validation is None:
            return jsonify({'error': 'Statistical validation not available'}), 404
        
        results = analyzer.statistical_validation.regime_detection_analysis()
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in regime detection API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stationarity_tests')
def api_stationarity_tests():
    """Get stationarity test results"""
    try:
        if analyzer.statistical_validation is None:
            return jsonify({'error': 'Statistical validation not available'}), 404
        
        results = analyzer.statistical_validation.stationarity_tests()
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in stationarity tests API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/granger_causality')
def api_granger_causality():
    """Get Granger causality test results"""
    try:
        if analyzer.statistical_validation is None:
            return jsonify({'error': 'Statistical validation not available'}), 404
        
        results = analyzer.statistical_validation.granger_causality_test()
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in Granger causality API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/volatility_analysis')
def api_volatility_analysis():
    """Get volatility analysis results"""
    try:
        if analyzer.statistical_validation is None:
            return jsonify({'error': 'Statistical validation not available'}), 404
        
        results = analyzer.statistical_validation.volatility_analysis()
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in volatility analysis API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/feature_sensitivity')
def api_feature_sensitivity():
    """Get feature sensitivity analysis"""
    try:
        if analyzer.statistical_validation is None:
            return jsonify({'error': 'Statistical validation not available'}), 404
        
        results = analyzer.statistical_validation.feature_sensitivity_analysis()
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in feature sensitivity API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/multicollinearity_check')
def api_multicollinearity_check():
    """Get multicollinearity check results"""
    try:
        if analyzer.statistical_validation is None:
            return jsonify({'error': 'Statistical validation not available'}), 404
        
        results = analyzer.statistical_validation.multicollinearity_check()
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in multicollinearity check API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistical_validation_report')
def api_statistical_validation_report():
    """Get comprehensive statistical validation report"""
    try:
        if analyzer.statistical_validation is None:
            return jsonify({'error': 'Statistical validation not available'}), 404
        
        report = analyzer.statistical_validation.generate_comprehensive_report()
        return jsonify(report)
    except Exception as e:
        logger.error(f"Error in statistical validation report API: {e}")
        return jsonify({'error': str(e)}), 500

# Advanced Backtesting Endpoints
@app.route('/api/signal_quality_analysis')
def api_signal_quality_analysis():
    """Get signal quality analysis"""
    try:
        if analyzer.advanced_backtesting is None:
            return jsonify({'error': 'Advanced backtesting not available'}), 404
        
        results = analyzer.advanced_backtesting.signal_quality_analysis()
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in signal quality analysis API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trading_simulation')
def api_trading_simulation():
    """Get trading simulation results"""
    try:
        if analyzer.advanced_backtesting is None:
            return jsonify({'error': 'Advanced backtesting not available'}), 404
        
        results = analyzer.advanced_backtesting.trading_simulation()
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in trading simulation API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/indicator_comparison')
def api_indicator_comparison():
    """Get indicator comparison results"""
    try:
        if analyzer.advanced_backtesting is None:
            return jsonify({'error': 'Advanced backtesting not available'}), 404
        
        results = analyzer.advanced_backtesting.indicator_comparison()
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in indicator comparison API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/regime_switching_analysis')
def api_regime_switching_analysis():
    """Get regime switching analysis"""
    try:
        if analyzer.advanced_backtesting is None:
            return jsonify({'error': 'Advanced backtesting not available'}), 404
        
        results = analyzer.advanced_backtesting.regime_switching_analysis()
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in regime switching analysis API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtesting_report')
def api_backtesting_report():
    """Get comprehensive backtesting report"""
    try:
        if analyzer.advanced_backtesting is None:
            return jsonify({'error': 'Advanced backtesting not available'}), 404
        
        report = analyzer.advanced_backtesting.generate_comprehensive_backtesting_report()
        return jsonify(report)
    except Exception as e:
        logger.error(f"Error in backtesting report API: {e}")
        return jsonify({'error': str(e)}), 500

# Existing chart endpoints (inherited from fast_ultimate_app)
@app.route('/api/bri_chart')
def api_bri_chart():
    """Get BRI time series chart"""
    try:
        chart_data = analyzer.get_cached_chart('bri_chart')
        if chart_data:
            return jsonify(chart_data)
        else:
            return jsonify({'error': 'Failed to create BRI chart'}), 500
    except Exception as e:
        logger.error(f"Error in BRI chart API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/candlestick_chart')
def api_candlestick_chart():
    """Get candlestick chart"""
    try:
        chart_data = analyzer.get_cached_chart('candlestick')
        if chart_data:
            return jsonify(chart_data)
        else:
            return jsonify({'error': 'Failed to create candlestick chart'}), 500
    except Exception as e:
        logger.error(f"Error in candlestick chart API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/box_plots')
def api_box_plots():
    """Get box plot analysis"""
    try:
        chart_data = analyzer.get_cached_chart('box_plots')
        if chart_data:
            return jsonify(chart_data)
        else:
            return jsonify({'error': 'Failed to create box plots'}), 500
    except Exception as e:
        logger.error(f"Error in box plots API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/correlation_heatmap')
def api_correlation_heatmap():
    """Get correlation heatmap"""
    try:
        chart_data = analyzer.get_cached_chart('correlation_heatmap')
        if chart_data:
            return jsonify(chart_data)
        else:
            return jsonify({'error': 'Failed to create correlation heatmap'}), 500
    except Exception as e:
        logger.error(f"Error in correlation heatmap API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/violin_plots')
def api_violin_plots():
    """Get violin plot analysis"""
    try:
        chart_data = analyzer.get_cached_chart('violin_plots')
        if chart_data:
            return jsonify(chart_data)
        else:
            return jsonify({'error': 'Failed to create violin plots'}), 500
    except Exception as e:
        logger.error(f"Error in violin plots API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/3d_surface')
def api_3d_surface():
    """Get 3D surface plot"""
    try:
        chart_data = analyzer.get_cached_chart('3d_surface')
        if chart_data:
            return jsonify(chart_data)
        else:
            return jsonify({'error': 'Failed to create 3D surface plot'}), 500
    except Exception as e:
        logger.error(f"Error in 3D surface API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecasting_comparison')
def api_forecasting_comparison():
    """Get forecasting comparison"""
    try:
        chart_data = analyzer.get_cached_chart('forecasting_comparison')
        if chart_data:
            return jsonify(chart_data)
        else:
            return jsonify({'error': 'Failed to create forecasting comparison'}), 500
    except Exception as e:
        logger.error(f"Error in forecasting comparison API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_performance')
def api_model_performance():
    """Get model performance comparison"""
    try:
        chart_data = analyzer.get_cached_chart('model_performance')
        if chart_data:
            return jsonify(chart_data)
        else:
            return jsonify({'error': 'Failed to create model performance comparison'}), 500
    except Exception as e:
        logger.error(f"Error in model performance API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monte_carlo_visualization')
def api_monte_carlo_visualization():
    """Get Monte Carlo visualization"""
    try:
        chart_data = analyzer.get_cached_chart('monte_carlo_simulation')
        if chart_data:
            return jsonify(chart_data)
        else:
            return jsonify({'error': 'Failed to create Monte Carlo visualization'}), 500
    except Exception as e:
        logger.error(f"Error in Monte Carlo visualization API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/risk_metrics')
def api_risk_metrics():
    """Get risk metrics chart"""
    try:
        chart_data = analyzer.get_cached_chart('risk_metrics')
        if chart_data:
            return jsonify(chart_data)
        else:
            return jsonify({'error': 'Failed to create risk metrics chart'}), 500
    except Exception as e:
        logger.error(f"Error in risk metrics API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stress_test_chart')
def api_stress_test_chart():
    """Get stress test chart"""
    try:
        chart_data = analyzer.get_cached_chart('stress_test')
        if chart_data:
            return jsonify(chart_data)
        else:
            return jsonify({'error': 'Failed to create stress test chart'}), 500
    except Exception as e:
        logger.error(f"Error in stress test chart API: {e}")
        return jsonify({'error': str(e)}), 500

# Export endpoints
@app.route('/api/export_data')
def api_export_data():
    """Export data as CSV or JSON"""
    try:
        format_type = request.args.get('format', 'csv')
        include_metadata = request.args.get('include_metadata', 'true').lower() == 'true'
        
        if format_type == 'csv':
            csv_data = analyzer.export_utils.export_data_as_csv(
                analyzer.bri_data, 
                'bri_data_export', 
                include_metadata
            )
            return jsonify({'csv_data': csv_data})
        elif format_type == 'json':
            json_data = analyzer.export_utils.export_data_as_json(
                analyzer.bri_data.to_dict('records'), 
                'bri_data_export', 
                include_metadata
            )
            return jsonify({'json_data': json_data})
        else:
            return jsonify({'error': 'Unsupported format'}), 400
    
    except Exception as e:
        logger.error(f"Error in export data API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export_report')
def api_export_report():
    """Export automated report"""
    try:
        # Get summary stats for report
        stats = analyzer.get_summary_stats()
        
        # Create sample charts data
        charts_data = {
            'bri_chart': None,
            'risk_heatmap': None,
        }
        
        report_data = analyzer.export_utils.create_automated_report(
            charts_data, 
            stats, 
            'bri_automated_report'
        )
        
        if report_data:
            return jsonify({'report_data': report_data})
        else:
            return jsonify({'error': 'Failed to create automated report'}), 500
    
    except Exception as e:
        logger.error(f"Error in export report API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/daily_summary')
def api_daily_summary():
    """Get daily summary report"""
    try:
        date = request.args.get('date', None)
        summary = analyzer.export_utils.create_daily_summary_report(date)
        
        if summary:
            return jsonify(summary)
        else:
            return jsonify({'error': 'Failed to create daily summary'}), 500
    
    except Exception as e:
        logger.error(f"Error in daily summary API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/weekly_summary')
def api_weekly_summary():
    """Get weekly summary report"""
    try:
        week_start = request.args.get('week_start', None)
        summary = analyzer.export_utils.create_weekly_summary_report(week_start)
        
        if summary:
            return jsonify(summary)
        else:
            return jsonify({'error': 'Failed to create weekly summary'}), 500
    
    except Exception as e:
        logger.error(f"Error in weekly summary API: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
