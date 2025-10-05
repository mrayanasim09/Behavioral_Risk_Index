"""
Ultimate BRI Dashboard with All Advanced Features
Integrates advanced charts, export functionality, forecasting, and Monte Carlo simulations
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

class UltimateBRIAnalyzer:
    """Ultimate BRI analyzer with all advanced features"""
    
    def __init__(self):
        self.bri_data = None
        self.market_data = None
        self.advanced_analytics = None
        self.advanced_charts = None
        self.export_utils = None
        self.forecasting_models = None
        self.monte_carlo = None
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
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.generate_sample_data()
            self.advanced_analytics = AdvancedAnalytics(self.bri_data, self.market_data)
            self.advanced_charts = AdvancedCharts(self.bri_data, self.market_data)
            self.export_utils = ExportUtils(self.bri_data, self.market_data)
            self.forecasting_models = ForecastingModels(self.bri_data, self.market_data)
            self.monte_carlo = MonteCarloSimulations(self.bri_data, self.market_data)
    
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

# Initialize Flask app
app = Flask(__name__)
analyzer = UltimateBRIAnalyzer()

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

# Advanced Analytics Endpoints
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

# Advanced Charts Endpoints
@app.route('/api/candlestick_chart')
def api_candlestick_chart():
    """Get candlestick chart"""
    try:
        if analyzer.advanced_charts is None:
            return jsonify({'error': 'Advanced charts not available'}), 404
        
        candlestick = analyzer.advanced_charts.create_candlestick_chart()
        return jsonify(candlestick.to_dict())
    
    except Exception as e:
        logger.error(f"Error in candlestick chart API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/box_plots')
def api_box_plots():
    """Get box plot analysis"""
    try:
        if analyzer.advanced_charts is None:
            return jsonify({'error': 'Advanced charts not available'}), 404
        
        box_plots = analyzer.advanced_charts.create_box_plot_analysis()
        return jsonify(box_plots.to_dict())
    
    except Exception as e:
        logger.error(f"Error in box plots API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/correlation_heatmap')
def api_correlation_heatmap():
    """Get correlation heatmap"""
    try:
        if analyzer.advanced_charts is None:
            return jsonify({'error': 'Advanced charts not available'}), 404
        
        correlation_heatmap = analyzer.advanced_charts.create_correlation_heatmap()
        return jsonify(correlation_heatmap.to_dict())
    
    except Exception as e:
        logger.error(f"Error in correlation heatmap API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/violin_plots')
def api_violin_plots():
    """Get violin plot analysis"""
    try:
        if analyzer.advanced_charts is None:
            return jsonify({'error': 'Advanced charts not available'}), 404
        
        violin_plots = analyzer.advanced_charts.create_violin_plot_analysis()
        return jsonify(violin_plots.to_dict())
    
    except Exception as e:
        logger.error(f"Error in violin plots API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/3d_surface')
def api_3d_surface():
    """Get 3D surface plot"""
    try:
        if analyzer.advanced_charts is None:
            return jsonify({'error': 'Advanced charts not available'}), 404
        
        surface_3d = analyzer.advanced_charts.create_3d_surface_plot()
        return jsonify(surface_3d.to_dict())
    
    except Exception as e:
        logger.error(f"Error in 3D surface API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/advanced_histogram')
def api_advanced_histogram():
    """Get advanced histogram"""
    try:
        if analyzer.advanced_charts is None:
            return jsonify({'error': 'Advanced charts not available'}), 404
        
        histogram = analyzer.advanced_charts.create_advanced_histogram()
        return jsonify(histogram.to_dict())
    
    except Exception as e:
        logger.error(f"Error in advanced histogram API: {e}")
        return jsonify({'error': str(e)}), 500

# Forecasting Endpoints
@app.route('/api/forecasting_comparison')
def api_forecasting_comparison():
    """Get forecasting comparison"""
    try:
        if analyzer.forecasting_models is None:
            return jsonify({'error': 'Forecasting models not available'}), 404
        
        days_ahead = request.args.get('days_ahead', 30, type=int)
        comparison = analyzer.forecasting_models.create_forecasting_comparison(days_ahead)
        
        if comparison:
            return jsonify(comparison.to_dict())
        else:
            return jsonify({'error': 'Failed to create forecasting comparison'}), 500
    
    except Exception as e:
        logger.error(f"Error in forecasting comparison API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_performance')
def api_model_performance():
    """Get model performance comparison"""
    try:
        if analyzer.forecasting_models is None:
            return jsonify({'error': 'Forecasting models not available'}), 404
        
        performance = analyzer.forecasting_models.create_model_performance_comparison()
        
        if performance:
            return jsonify(performance.to_dict())
        else:
            return jsonify({'error': 'Failed to create model performance comparison'}), 500
    
    except Exception as e:
        logger.error(f"Error in model performance API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecasting_report')
def api_forecasting_report():
    """Get comprehensive forecasting report"""
    try:
        if analyzer.forecasting_models is None:
            return jsonify({'error': 'Forecasting models not available'}), 404
        
        days_ahead = request.args.get('days_ahead', 30, type=int)
        report = analyzer.forecasting_models.generate_forecasting_report(days_ahead)
        
        if report:
            # Convert Plotly figures to dict
            if 'comparison_chart' in report and report['comparison_chart']:
                report['comparison_chart'] = report['comparison_chart'].to_dict()
            if 'performance_chart' in report and report['performance_chart']:
                report['performance_chart'] = report['performance_chart'].to_dict()
            
            return jsonify(report)
        else:
            return jsonify({'error': 'Failed to generate forecasting report'}), 500
    
    except Exception as e:
        logger.error(f"Error in forecasting report API: {e}")
        return jsonify({'error': str(e)}), 500

# Monte Carlo Endpoints
@app.route('/api/monte_carlo_simulation')
def api_monte_carlo_simulation():
    """Get Monte Carlo simulation"""
    try:
        if analyzer.monte_carlo is None:
            return jsonify({'error': 'Monte Carlo simulations not available'}), 404
        
        method = request.args.get('method', 'bootstrap')
        n_simulations = request.args.get('n_simulations', 1000, type=int)
        time_horizon = request.args.get('time_horizon', 30, type=int)
        
        simulation = analyzer.monte_carlo.simulate_bri_paths(
            n_simulations=n_simulations,
            time_horizon=time_horizon,
            method=method
        )
        
        if simulation:
            return jsonify(simulation)
        else:
            return jsonify({'error': 'Failed to run Monte Carlo simulation'}), 500
    
    except Exception as e:
        logger.error(f"Error in Monte Carlo simulation API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monte_carlo_visualization')
def api_monte_carlo_visualization():
    """Get Monte Carlo visualization"""
    try:
        if analyzer.monte_carlo is None:
            return jsonify({'error': 'Monte Carlo simulations not available'}), 404
        
        method = request.args.get('method', 'bootstrap')
        n_paths = request.args.get('n_paths', 100, type=int)
        
        visualization = analyzer.monte_carlo.create_simulation_visualization(method, n_paths)
        
        if visualization:
            return jsonify(visualization.to_dict())
        else:
            return jsonify({'error': 'Failed to create Monte Carlo visualization'}), 500
    
    except Exception as e:
        logger.error(f"Error in Monte Carlo visualization API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/risk_metrics')
def api_risk_metrics():
    """Get risk metrics chart"""
    try:
        if analyzer.monte_carlo is None:
            return jsonify({'error': 'Monte Carlo simulations not available'}), 404
        
        method = request.args.get('method', 'bootstrap')
        
        risk_metrics = analyzer.monte_carlo.create_risk_metrics_chart(method)
        
        if risk_metrics:
            return jsonify(risk_metrics.to_dict())
        else:
            return jsonify({'error': 'Failed to create risk metrics chart'}), 500
    
    except Exception as e:
        logger.error(f"Error in risk metrics API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stress_test')
def api_stress_test():
    """Get stress test results"""
    try:
        if analyzer.monte_carlo is None:
            return jsonify({'error': 'Monte Carlo simulations not available'}), 404
        
        n_simulations = request.args.get('n_simulations', 1000, type=int)
        time_horizon = request.args.get('time_horizon', 30, type=int)
        
        stress_results = analyzer.monte_carlo.stress_test_scenarios(n_simulations, time_horizon)
        
        if stress_results:
            return jsonify(stress_results)
        else:
            return jsonify({'error': 'Failed to run stress test'}), 500
    
    except Exception as e:
        logger.error(f"Error in stress test API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stress_test_chart')
def api_stress_test_chart():
    """Get stress test chart"""
    try:
        if analyzer.monte_carlo is None:
            return jsonify({'error': 'Monte Carlo simulations not available'}), 404
        
        n_simulations = request.args.get('n_simulations', 1000, type=int)
        time_horizon = request.args.get('time_horizon', 30, type=int)
        
        stress_chart = analyzer.monte_carlo.create_stress_test_chart(n_simulations, time_horizon)
        
        if stress_chart:
            return jsonify(stress_chart.to_dict())
        else:
            return jsonify({'error': 'Failed to create stress test chart'}), 500
    
    except Exception as e:
        logger.error(f"Error in stress test chart API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monte_carlo_report')
def api_monte_carlo_report():
    """Get comprehensive Monte Carlo report"""
    try:
        if analyzer.monte_carlo is None:
            return jsonify({'error': 'Monte Carlo simulations not available'}), 404
        
        n_simulations = request.args.get('n_simulations', 1000, type=int)
        time_horizon = request.args.get('time_horizon', 30, type=int)
        
        report = analyzer.monte_carlo.generate_monte_carlo_report(n_simulations, time_horizon)
        
        if report:
            # Convert Plotly figures to dict
            if 'simulation_charts' in report:
                for chart_name, chart_fig in report['simulation_charts'].items():
                    if chart_fig:
                        report['simulation_charts'][chart_name] = chart_fig.to_dict()
            
            if 'stress_chart' in report and report['stress_chart']:
                report['stress_chart'] = report['stress_chart'].to_dict()
            
            return jsonify(report)
        else:
            return jsonify({'error': 'Failed to generate Monte Carlo report'}), 500
    
    except Exception as e:
        logger.error(f"Error in Monte Carlo report API: {e}")
        return jsonify({'error': str(e)}), 500

# Export Endpoints
@app.route('/api/export_chart')
def api_export_chart():
    """Export chart as PNG or PDF"""
    try:
        if analyzer.export_utils is None:
            return jsonify({'error': 'Export utils not available'}), 404
        
        chart_type = request.args.get('chart_type', 'bri_chart')
        format_type = request.args.get('format', 'png')
        width = request.args.get('width', 1200, type=int)
        height = request.args.get('height', 800, type=int)
        
        # Get chart data (this would need to be implemented based on chart_type)
        # For now, return a placeholder
        return jsonify({'error': 'Chart export not implemented yet'}), 501
    
    except Exception as e:
        logger.error(f"Error in export chart API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export_data')
def api_export_data():
    """Export data as CSV or JSON"""
    try:
        if analyzer.export_utils is None:
            return jsonify({'error': 'Export utils not available'}), 404
        
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
        if analyzer.export_utils is None:
            return jsonify({'error': 'Export utils not available'}), 404
        
        # Get summary stats for report
        stats = analyzer.get_summary_stats()
        
        # Create sample charts data (this would be replaced with actual charts)
        charts_data = {
            'bri_chart': None,  # Would be actual chart
            'risk_heatmap': None,  # Would be actual chart
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
        if analyzer.export_utils is None:
            return jsonify({'error': 'Export utils not available'}), 404
        
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
        if analyzer.export_utils is None:
            return jsonify({'error': 'Export utils not available'}), 404
        
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
