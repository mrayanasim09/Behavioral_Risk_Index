"""
Advanced Analytics Module for BRI Dashboard
Implements risk heatmaps, volatility clustering, and advanced visualizations
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnalytics:
    """Advanced analytics for BRI dashboard"""
    
    def __init__(self, bri_data, market_data=None):
        self.bri_data = bri_data.copy()
        self.market_data = market_data
        self.risk_thresholds = {
            'low': 30,
            'moderate': 60,
            'high': 100
        }
        
    def create_risk_heatmap(self, window_size=7):
        """
        Create a risk heatmap showing risk levels over time
        
        Args:
            window_size (int): Rolling window size for smoothing
            
        Returns:
            plotly.graph_objects.Figure: Risk heatmap
        """
        # Prepare data
        df = self.bri_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Create rolling risk levels
        df['risk_level'] = df['BRI'].rolling(window=window_size).mean()
        df['risk_category'] = pd.cut(df['risk_level'], 
                                   bins=[0, 30, 60, 100], 
                                   labels=['Low', 'Moderate', 'High'])
        
        # Create monthly aggregation for heatmap
        monthly_data = df.resample('M').agg({
            'BRI': ['mean', 'std', 'max'],
            'risk_category': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Low'
        }).round(2)
        
        # Flatten column names
        monthly_data.columns = ['mean_bri', 'std_bri', 'max_bri', 'dominant_risk']
        
        # Create heatmap data
        heatmap_data = monthly_data['mean_bri'].values.reshape(-1, 1)
        
        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=['Risk Level'],
            y=[d.strftime('%Y-%m') for d in monthly_data.index],
            colorscale=[[0, '#38A169'], [0.5, '#D69E2E'], [1, '#E53E3E']],
            showscale=True,
            colorbar=dict(
                title=dict(text="BRI Level", font=dict(color='#1A202C', family='Inter')),
                tickfont=dict(color='#1A202C', family='Inter')
            ),
            hovertemplate='<b>%{y}</b><br>Mean BRI: %{z:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text='Risk Heatmap - Monthly BRI Levels',
                font=dict(color='#1A202C', size=18, family='Inter')
            ),
            xaxis_title='Risk Categories',
            yaxis_title='Time Period',
            height=400,
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(color='#1A202C', family='Inter')
        )
        
        return fig
    
    def identify_volatility_clusters(self, n_clusters=3):
        """
        Identify high/low volatility periods using clustering
        
        Args:
            n_clusters (int): Number of volatility clusters
            
        Returns:
            dict: Clustering results and visualization
        """
        # Prepare features for clustering
        df = self.bri_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Calculate volatility features
        df['bri_volatility'] = df['BRI'].rolling(window=7).std()
        df['bri_change'] = df['BRI'].pct_change()
        df['bri_abs_change'] = df['bri_change'].abs()
        df['bri_ma_ratio'] = df['BRI'] / df['BRI'].rolling(window=30).mean()
        
        # Remove NaN values
        features_df = df[['bri_volatility', 'bri_abs_change', 'bri_ma_ratio']].dropna()
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Add cluster labels to dataframe
        features_df['cluster'] = clusters
        features_df['cluster_label'] = features_df['cluster'].map({
            0: 'Low Volatility',
            1: 'Moderate Volatility', 
            2: 'High Volatility'
        })
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('BRI Time Series with Volatility Clusters', 'Volatility Distribution by Cluster'),
            vertical_spacing=0.1
        )
        
        # Plot BRI with cluster colors
        colors = ['#38A169', '#D69E2E', '#E53E3E']
        for i, cluster in enumerate(features_df['cluster'].unique()):
            cluster_data = features_df[features_df['cluster'] == cluster]
            fig.add_trace(
                go.Scatter(
                    x=cluster_data.index,
                    y=cluster_data.index.map(lambda x: df.loc[x, 'BRI']),
                    mode='markers',
                    name=f'Cluster {i}',
                    marker=dict(color=colors[i], size=6),
                    hovertemplate=f'<b>{features_df.loc[cluster_data.index[0], "cluster_label"]}</b><br>' +
                                 'Date: %{x}<br>BRI: %{y:.1f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Plot volatility distribution
        for i, cluster in enumerate(features_df['cluster'].unique()):
            cluster_vol = features_df[features_df['cluster'] == cluster]['bri_volatility']
            fig.add_trace(
                go.Box(
                    y=cluster_vol,
                    name=f'Cluster {i}',
                    marker_color=colors[i],
                    boxpoints='outliers'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=dict(
                text='Volatility Clustering Analysis',
                font=dict(color='#1A202C', size=18, family='Inter')
            ),
            height=600,
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(color='#1A202C', family='Inter'),
            showlegend=True
        )
        
        # Calculate cluster statistics
        cluster_stats = features_df.groupby('cluster_label').agg({
            'bri_volatility': ['mean', 'std'],
            'bri_abs_change': ['mean', 'std'],
            'bri_ma_ratio': ['mean', 'std']
        }).round(3)
        
        return {
            'figure': fig,
            'cluster_stats': cluster_stats,
            'cluster_labels': features_df['cluster_label'].value_counts().to_dict()
        }
    
    def create_early_warning_system(self, threshold_percentile=90, lookback_days=30):
        """
        Create early warning system for risk spikes
        
        Args:
            threshold_percentile (int): Percentile for spike detection
            lookback_days (int): Days to look back for threshold calculation
            
        Returns:
            dict: Warning system results and visualization
        """
        df = self.bri_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Calculate dynamic threshold
        df['rolling_threshold'] = df['BRI'].rolling(window=lookback_days).quantile(threshold_percentile/100)
        # Handle NaN values in rolling threshold
        df['rolling_threshold'] = df['rolling_threshold'].fillna(df['BRI'].quantile(threshold_percentile/100))
        df['spike_detected'] = df['BRI'] > df['rolling_threshold']
        
        # Find spike events (remove NaN values first)
        df_clean = df.dropna()
        spike_events = df_clean[df_clean['spike_detected']].copy()
        
        # Create visualization
        fig = go.Figure()
        
        # Add BRI line
        fig.add_trace(go.Scatter(
            x=df_clean.index,
            y=df_clean['BRI'],
            mode='lines',
            name='BRI',
            line=dict(color='#3182CE', width=2)
        ))
        
        # Add threshold line
        fig.add_trace(go.Scatter(
            x=df_clean.index,
            y=df_clean['rolling_threshold'],
            mode='lines',
            name=f'{threshold_percentile}th Percentile Threshold',
            line=dict(color='#E53E3E', width=2, dash='dash')
        ))
        
        # Add spike markers
        if not spike_events.empty:
            fig.add_trace(go.Scatter(
                x=spike_events.index,
                y=spike_events['BRI'],
                mode='markers',
                name='Risk Spikes',
                marker=dict(
                    color='#E53E3E',
                    size=12,
                    symbol='triangle-up'
                ),
                hovertemplate='<b>Risk Spike Detected</b><br>' +
                             'Date: %{x}<br>BRI: %{y:.1f}<br>' +
                             f'Threshold: {threshold_percentile}th percentile<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text=f'Early Warning System - {threshold_percentile}th Percentile Threshold',
                font=dict(color='#1A202C', size=18, family='Inter')
            ),
            xaxis_title='Date',
            yaxis_title='BRI Level',
            height=500,
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(color='#1A202C', family='Inter'),
            hovermode='x unified'
        )
        
        # Calculate warning statistics
        warning_stats = {
            'total_spikes': len(spike_events),
            'spike_frequency': len(spike_events) / len(df_clean) * 100 if len(df_clean) > 0 else 0,
            'avg_spike_magnitude': spike_events['BRI'].mean() if not spike_events.empty else 0,
            'max_spike': spike_events['BRI'].max() if not spike_events.empty else 0,
            'recent_spikes': len(spike_events[spike_events.index >= df_clean.index[-30:]]) if not spike_events.empty and len(df_clean) >= 30 else 0
        }
        
        return {
            'figure': fig,
            'warning_stats': warning_stats,
            'spike_events': spike_events[['BRI', 'rolling_threshold']].to_dict('index') if not spike_events.empty else {}
        }
    
    def calculate_confidence_intervals(self, prediction_days=7, confidence_level=0.95):
        """
        Calculate confidence intervals for BRI predictions
        
        Args:
            prediction_days (int): Number of days to predict ahead
            confidence_level (float): Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            dict: Confidence interval results and visualization
        """
        df = self.bri_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Simple ARIMA-like prediction with confidence intervals
        bri_series = df['BRI'].dropna()
        
        # Calculate rolling statistics
        window = 30
        rolling_mean = bri_series.rolling(window=window).mean()
        rolling_std = bri_series.rolling(window=window).std()
        
        # Generate future dates
        last_date = bri_series.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days, freq='D')
        
        # Simple trend-based prediction
        recent_trend = bri_series.tail(7).mean() - bri_series.tail(14).head(7).mean()
        last_value = bri_series.iloc[-1]
        
        # Generate predictions
        predictions = []
        for i in range(prediction_days):
            pred_value = last_value + (recent_trend * (i + 1))
            predictions.append(pred_value)
        
        # Calculate confidence intervals
        recent_std = rolling_std.iloc[-1]
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        upper_bound = [p + z_score * recent_std for p in predictions]
        lower_bound = [p - z_score * recent_std for p in predictions]
        
        # Create visualization
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=bri_series.index,
            y=bri_series.values,
            mode='lines',
            name='Historical BRI',
            line=dict(color='#3182CE', width=2)
        ))
        
        # Add predictions
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name='Predicted BRI',
            line=dict(color='#E53E3E', width=2),
            marker=dict(size=6)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=future_dates.tolist() + future_dates.tolist()[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='tonexty',
            fillcolor='rgba(229, 62, 62, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{confidence_level*100:.0f}% Confidence Interval',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=dict(
                text=f'BRI Forecast with {confidence_level*100:.0f}% Confidence Intervals',
                font=dict(color='#1A202C', size=18, family='Inter')
            ),
            xaxis_title='Date',
            yaxis_title='BRI Level',
            height=500,
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(color='#1A202C', family='Inter'),
            hovermode='x unified'
        )
        
        return {
            'figure': fig,
            'predictions': dict(zip(future_dates.strftime('%Y-%m-%d'), predictions)),
            'confidence_intervals': {
                'upper': dict(zip(future_dates.strftime('%Y-%m-%d'), upper_bound)),
                'lower': dict(zip(future_dates.strftime('%Y-%m-%d'), lower_bound))
            },
            'prediction_accuracy': {
                'trend_magnitude': recent_trend,
                'confidence_level': confidence_level,
                'prediction_horizon': prediction_days
            }
        }
    
    def generate_analytics_summary(self):
        """Generate comprehensive analytics summary"""
        # Risk heatmap
        heatmap = self.create_risk_heatmap()
        
        # Volatility clustering
        clustering = self.identify_volatility_clusters()
        
        # Early warning system
        warning_system = self.create_early_warning_system()
        
        # Confidence intervals
        confidence_intervals = self.calculate_confidence_intervals()
        
        return {
            'risk_heatmap': heatmap,
            'volatility_clustering': clustering,
            'early_warning': warning_system,
            'confidence_intervals': confidence_intervals
        }
