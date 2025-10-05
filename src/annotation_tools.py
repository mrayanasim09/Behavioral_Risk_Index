"""
Annotation Tools for BRI Dashboard
Implements event markers, trend lines, and text annotations for charts
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AnnotationTools:
    """Annotation tools for marking significant events on charts"""
    
    def __init__(self, bri_data, market_data=None):
        self.bri_data = bri_data.copy()
        self.market_data = market_data
        self.significant_events = self._load_significant_events()
        
    def _load_significant_events(self):
        """Load significant market events for annotation"""
        events = [
            # 2022 Events
            {'date': '2022-02-24', 'event': 'Russia-Ukraine War', 'type': 'geopolitical', 'impact': 'high'},
            {'date': '2022-03-16', 'event': 'Fed Rate Hike', 'type': 'monetary', 'impact': 'high'},
            {'date': '2022-06-15', 'event': 'Fed 75bp Rate Hike', 'type': 'monetary', 'impact': 'high'},
            {'date': '2022-09-21', 'event': 'Fed 75bp Rate Hike', 'type': 'monetary', 'impact': 'high'},
            {'date': '2022-10-13', 'event': 'CPI Inflation Report', 'type': 'economic', 'impact': 'medium'},
            {'date': '2022-11-10', 'event': 'CPI Inflation Report', 'type': 'economic', 'impact': 'medium'},
            
            # 2023 Events
            {'date': '2023-01-06', 'event': 'Jobs Report', 'type': 'economic', 'impact': 'medium'},
            {'date': '2023-03-10', 'event': 'SVB Bank Collapse', 'type': 'banking', 'impact': 'high'},
            {'date': '2023-03-22', 'event': 'Fed Rate Decision', 'type': 'monetary', 'impact': 'high'},
            {'date': '2023-05-03', 'event': 'Fed Rate Decision', 'type': 'monetary', 'impact': 'high'},
            {'date': '2023-07-26', 'event': 'Fed Rate Decision', 'type': 'monetary', 'impact': 'high'},
            {'date': '2023-09-20', 'event': 'Fed Rate Decision', 'type': 'monetary', 'impact': 'high'},
            {'date': '2023-10-07', 'event': 'Israel-Hamas Conflict', 'type': 'geopolitical', 'impact': 'high'},
            {'date': '2023-11-01', 'event': 'Fed Rate Decision', 'type': 'monetary', 'impact': 'high'},
            {'date': '2023-12-13', 'event': 'Fed Rate Decision', 'type': 'monetary', 'impact': 'high'},
            
            # 2024 Events
            {'date': '2024-01-31', 'event': 'Fed Rate Decision', 'type': 'monetary', 'impact': 'high'},
            {'date': '2024-03-20', 'event': 'Fed Rate Decision', 'type': 'monetary', 'impact': 'high'},
            {'date': '2024-05-01', 'event': 'Fed Rate Decision', 'type': 'monetary', 'impact': 'high'},
            {'date': '2024-06-12', 'event': 'CPI Inflation Report', 'type': 'economic', 'impact': 'medium'},
            {'date': '2024-07-31', 'event': 'Fed Rate Decision', 'type': 'monetary', 'impact': 'high'},
            {'date': '2024-09-18', 'event': 'Fed Rate Decision', 'type': 'monetary', 'impact': 'high'},
            {'date': '2024-11-07', 'event': 'Fed Rate Decision', 'type': 'monetary', 'impact': 'high'},
            {'date': '2024-12-18', 'event': 'Fed Rate Decision', 'type': 'monetary', 'impact': 'high'},
        ]
        
        return pd.DataFrame(events)
    
    def add_event_annotations(self, fig, chart_type='bri'):
        """
        Add event annotations to a chart
        
        Args:
            fig: Plotly figure object
            chart_type (str): Type of chart ('bri', 'vix', 'correlation')
            
        Returns:
            plotly.graph_objects.Figure: Figure with annotations
        """
        try:
            # Filter events that fall within the data range
            bri_dates = pd.to_datetime(self.bri_data['date'])
            start_date = bri_dates.min()
            end_date = bri_dates.max()
            
            relevant_events = self.significant_events[
                (pd.to_datetime(self.significant_events['date']) >= start_date) &
                (pd.to_datetime(self.significant_events['date']) <= end_date)
            ]
            
            # Add annotations for each event
            for _, event in relevant_events.iterrows():
                event_date = event['date']
                event_name = event['event']
                event_type = event['type']
                impact = event['impact']
                
                # Determine annotation position based on chart type
                if chart_type == 'bri':
                    # Find BRI value on the event date
                    event_bri = self.bri_data[self.bri_data['date'] == event_date]['BRI'].values
                    if len(event_bri) > 0:
                        y_pos = event_bri[0]
                    else:
                        y_pos = self.bri_data['BRI'].mean()
                elif chart_type == 'vix':
                    # Find VIX value on the event date
                    if self.market_data is not None and 'VIX' in self.market_data.columns:
                        event_vix = self.market_data[self.market_data['date'] == event_date]['VIX'].values
                        if len(event_vix) > 0:
                            y_pos = event_vix[0]
                        else:
                            y_pos = self.market_data['VIX'].mean()
                    else:
                        y_pos = 20  # Default VIX level
                else:
                    y_pos = 0.5  # Default for correlation charts
                
                # Determine annotation style based on impact
                if impact == 'high':
                    bgcolor = 'rgba(231, 76, 60, 0.8)'
                    bordercolor = 'rgb(231, 76, 60)'
                    font_color = 'white'
                elif impact == 'medium':
                    bgcolor = 'rgba(214, 158, 46, 0.8)'
                    bordercolor = 'rgb(214, 158, 46)'
                    font_color = 'white'
                else:
                    bgcolor = 'rgba(56, 161, 105, 0.8)'
                    bordercolor = 'rgb(56, 161, 105)'
                    font_color = 'white'
                
                # Add annotation
                fig.add_annotation(
                    x=event_date,
                    y=y_pos,
                    text=event_name,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=bordercolor,
                    ax=0,
                    ay=-40,
                    bgcolor=bgcolor,
                    bordercolor=bordercolor,
                    borderwidth=1,
                    font=dict(color=font_color, size=10),
                    opacity=0.9
                )
            
            return fig
            
        except Exception as e:
            print(f"Error adding event annotations: {e}")
            return fig
    
    def add_trend_lines(self, fig, chart_type='bri'):
        """
        Add trend lines to a chart
        
        Args:
            fig: Plotly figure object
            chart_type (str): Type of chart
            
        Returns:
            plotly.graph_objects.Figure: Figure with trend lines
        """
        try:
            if chart_type == 'bri':
                # Add linear trend line
                bri_values = self.bri_data['BRI'].values
                dates = pd.to_datetime(self.bri_data['date'])
                x_numeric = np.arange(len(dates))
                
                # Calculate linear trend
                z = np.polyfit(x_numeric, bri_values, 1)
                p = np.poly1d(z)
                trend_line = p(x_numeric)
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=trend_line,
                    mode='lines',
                    name='Linear Trend',
                    line=dict(color='rgba(52, 73, 94, 0.8)', width=2, dash='dash'),
                    opacity=0.7
                ))
                
                # Add polynomial trend (degree 2)
                z2 = np.polyfit(x_numeric, bri_values, 2)
                p2 = np.poly1d(z2)
                trend_line2 = p2(x_numeric)
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=trend_line2,
                    mode='lines',
                    name='Polynomial Trend',
                    line=dict(color='rgba(155, 89, 182, 0.8)', width=2, dash='dot'),
                    opacity=0.7
                ))
            
            return fig
            
        except Exception as e:
            print(f"Error adding trend lines: {e}")
            return fig
    
    def add_support_resistance_lines(self, fig, chart_type='bri'):
        """
        Add support and resistance lines to a chart
        
        Args:
            fig: Plotly figure object
            chart_type (str): Type of chart
            
        Returns:
            plotly.graph_objects.Figure: Figure with support/resistance lines
        """
        try:
            if chart_type == 'bri':
                bri_values = self.bri_data['BRI'].values
                
                # Calculate support and resistance levels
                support_level = np.percentile(bri_values, 20)  # 20th percentile
                resistance_level = np.percentile(bri_values, 80)  # 80th percentile
                mean_level = np.mean(bri_values)
                
                dates = pd.to_datetime(self.bri_data['date'])
                
                # Add support line
                fig.add_hline(
                    y=support_level,
                    line_dash="dash",
                    line_color="rgba(56, 161, 105, 0.8)",
                    annotation_text=f"Support: {support_level:.1f}",
                    annotation_position="bottom right"
                )
                
                # Add resistance line
                fig.add_hline(
                    y=resistance_level,
                    line_dash="dash",
                    line_color="rgba(231, 76, 60, 0.8)",
                    annotation_text=f"Resistance: {resistance_level:.1f}",
                    annotation_position="top right"
                )
                
                # Add mean line
                fig.add_hline(
                    y=mean_level,
                    line_dash="dot",
                    line_color="rgba(52, 73, 94, 0.8)",
                    annotation_text=f"Mean: {mean_level:.1f}",
                    annotation_position="top left"
                )
            
            return fig
            
        except Exception as e:
            print(f"Error adding support/resistance lines: {e}")
            return fig
    
    def add_volatility_bands(self, fig, chart_type='bri', window=30):
        """
        Add volatility bands to a chart
        
        Args:
            fig: Plotly figure object
            chart_type (str): Type of chart
            window (int): Rolling window for volatility calculation
            
        Returns:
            plotly.graph_objects.Figure: Figure with volatility bands
        """
        try:
            if chart_type == 'bri':
                bri_values = self.bri_data['BRI'].values
                dates = pd.to_datetime(self.bri_data['date'])
                
                # Calculate rolling mean and standard deviation
                rolling_mean = pd.Series(bri_values).rolling(window=window, center=True).mean()
                rolling_std = pd.Series(bri_values).rolling(window=window, center=True).std()
                
                # Calculate bands
                upper_band = rolling_mean + (2 * rolling_std)
                lower_band = rolling_mean - (2 * rolling_std)
                
                # Add upper band
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=upper_band,
                    mode='lines',
                    name=f'Upper Band (+2σ)',
                    line=dict(color='rgba(231, 76, 60, 0.3)', width=1),
                    fill=None,
                    showlegend=True
                ))
                
                # Add lower band
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=lower_band,
                    mode='lines',
                    name=f'Lower Band (-2σ)',
                    line=dict(color='rgba(56, 161, 105, 0.3)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(231, 76, 60, 0.1)',
                    showlegend=True
                ))
            
            return fig
            
        except Exception as e:
            print(f"Error adding volatility bands: {e}")
            return fig
    
    def add_crisis_periods(self, fig, chart_type='bri'):
        """
        Add crisis period shading to a chart
        
        Args:
            fig: Plotly figure object
            chart_type (str): Type of chart
            
        Returns:
            plotly.graph_objects.Figure: Figure with crisis period shading
        """
        try:
            # Define crisis periods
            crisis_periods = [
                {'start': '2022-02-24', 'end': '2022-04-30', 'name': 'Russia-Ukraine War'},
                {'start': '2022-09-01', 'end': '2022-10-31', 'name': 'Fed Rate Hikes'},
                {'start': '2023-03-01', 'end': '2023-04-30', 'name': 'Banking Crisis'},
                {'start': '2023-10-01', 'end': '2023-11-30', 'name': 'Geopolitical Tensions'},
            ]
            
            for crisis in crisis_periods:
                fig.add_vrect(
                    x0=crisis['start'],
                    x1=crisis['end'],
                    fillcolor="rgba(231, 76, 60, 0.2)",
                    layer="below",
                    line_width=0,
                    annotation_text=crisis['name'],
                    annotation_position="top left",
                    annotation_font_size=10,
                    annotation_font_color="rgba(231, 76, 60, 0.8)"
                )
            
            return fig
            
        except Exception as e:
            print(f"Error adding crisis periods: {e}")
            return fig
    
    def create_annotated_bri_chart(self):
        """
        Create a BRI chart with all annotations
        
        Returns:
            plotly.graph_objects.Figure: Fully annotated BRI chart
        """
        try:
            # Create base BRI chart
            fig = go.Figure()
            
            # Add BRI line
            bri_values = self.bri_data['BRI'].values
            bri_dates = pd.to_datetime(self.bri_data['date'])
            
            fig.add_trace(go.Scatter(
                x=bri_dates,
                y=bri_values,
                mode='lines+markers',
                name='BRI',
                line=dict(color='#38A169', width=2),
                marker=dict(size=4)
            ))
            
            # Add all annotations
            fig = self.add_event_annotations(fig, 'bri')
            fig = self.add_trend_lines(fig, 'bri')
            fig = self.add_support_resistance_lines(fig, 'bri')
            fig = self.add_volatility_bands(fig, 'bri')
            fig = self.add_crisis_periods(fig, 'bri')
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text='BRI with Annotations - Events, Trends, and Crisis Periods',
                    font=dict(color='#1A202C', size=18, family='Inter')
                ),
                xaxis_title='Date',
                yaxis_title='BRI (0-100)',
                hovermode='x unified',
                height=600,
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                font=dict(color='#1A202C', family='Inter'),
                xaxis=dict(gridcolor='#E2E8F0'),
                yaxis=dict(gridcolor='#E2E8F0'),
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating annotated BRI chart: {e}")
            return None
    
    def create_annotation_summary(self):
        """
        Create a summary of all annotations
        
        Returns:
            dict: Annotation summary
        """
        try:
            # Count events by type and impact
            event_summary = self.significant_events.groupby(['type', 'impact']).size().unstack(fill_value=0)
            
            # Calculate crisis period statistics
            crisis_periods = [
                {'start': '2022-02-24', 'end': '2022-04-30', 'name': 'Russia-Ukraine War'},
                {'start': '2022-09-01', 'end': '2022-10-31', 'name': 'Fed Rate Hikes'},
                {'start': '2023-03-01', 'end': '2023-04-30', 'name': 'Banking Crisis'},
                {'start': '2023-10-01', 'end': '2023-11-30', 'name': 'Geopolitical Tensions'},
            ]
            
            total_crisis_days = 0
            for crisis in crisis_periods:
                start = pd.to_datetime(crisis['start'])
                end = pd.to_datetime(crisis['end'])
                days = (end - start).days
                total_crisis_days += days
            
            summary = {
                'total_events': len(self.significant_events),
                'events_by_type': self.significant_events['type'].value_counts().to_dict(),
                'events_by_impact': self.significant_events['impact'].value_counts().to_dict(),
                'crisis_periods': len(crisis_periods),
                'total_crisis_days': total_crisis_days,
                'crisis_percentage': (total_crisis_days / len(self.bri_data)) * 100,
                'event_summary_table': event_summary.to_dict()
            }
            
            return summary
            
        except Exception as e:
            print(f"Error creating annotation summary: {e}")
            return {'error': str(e)}
