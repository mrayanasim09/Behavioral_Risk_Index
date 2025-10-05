"""
Advanced Chart Types for BRI Dashboard
Implements candlestick charts, box plots, correlation heatmaps, and more
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedCharts:
    """Advanced chart types for enhanced visualizations"""
    
    def __init__(self, bri_data, market_data=None):
        self.bri_data = bri_data.copy()
        self.market_data = market_data
        self.professional_colors = {
            'primary': '#1A365D',
            'secondary': '#2D3748',
            'accent': '#C53030',
            'risk_low': '#38A169',
            'risk_moderate': '#D69E2E',
            'risk_high': '#E53E3E',
            'neutral': '#6B7280'
        }
        
    def create_candlestick_chart(self, symbol='SP500', period=90):
        """
        Create candlestick chart for market data
        
        Args:
            symbol (str): Market symbol to display
            period (int): Number of days to display
            
        Returns:
            plotly.graph_objects.Figure: Candlestick chart
        """
        # Generate sample OHLC data if market data not available
        if self.market_data is None or 'date' not in self.market_data.columns:
            dates = pd.date_range(start='2024-01-01', periods=period, freq='D')
            np.random.seed(42)
            
            # Generate realistic OHLC data
            base_price = 4000
            prices = []
            current_price = base_price
            
            for i in range(period):
                # Random walk with trend
                change = np.random.normal(0, 0.02)
                current_price *= (1 + change)
                prices.append(current_price)
            
            # Create OHLC data
            ohlc_data = []
            for i, price in enumerate(prices):
                high = price * (1 + abs(np.random.normal(0, 0.01)))
                low = price * (1 - abs(np.random.normal(0, 0.01)))
                open_price = prices[i-1] if i > 0 else price
                close_price = price
                
                ohlc_data.append({
                    'Date': dates[i],
                    'Open': open_price,
                    'High': high,
                    'Low': low,
                    'Close': close_price
                })
            
            df = pd.DataFrame(ohlc_data)
        else:
            # Use actual market data if available
            df = self.market_data.tail(period).copy()
            df['Date'] = pd.to_datetime(df['date'])
            df = df.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close'})
        
        # Create candlestick chart
        fig = go.Figure(data=go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=symbol,
            increasing_line_color=self.professional_colors['risk_low'],
            decreasing_line_color=self.professional_colors['risk_high'],
            increasing_fillcolor=self.professional_colors['risk_low'],
            decreasing_fillcolor=self.professional_colors['risk_high']
        ))
        
        # Add volume if available
        if 'Volume' in df.columns:
            fig.add_trace(go.Bar(
                x=df['Date'],
                y=df['Volume'],
                name='Volume',
                yaxis='y2',
                marker_color=self.professional_colors['neutral'],
                opacity=0.3
            ))
            
            fig.update_layout(
                yaxis2=dict(
                    title='Volume',
                    overlaying='y',
                    side='right',
                    showgrid=False
                )
            )
        
        fig.update_layout(
            title=dict(
                text=f'{symbol} Candlestick Chart - Last {period} Days',
                font=dict(color=self.professional_colors['primary'], size=18, family='Inter')
            ),
            xaxis_title='Date',
            yaxis_title='Price',
            height=500,
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(color=self.professional_colors['primary'], family='Inter'),
            xaxis=dict(gridcolor='#E2E8F0'),
            yaxis=dict(gridcolor='#E2E8F0'),
            hovermode='x unified'
        )
        
        return fig
    
    def create_box_plot_analysis(self):
        """
        Create comprehensive box plot analysis for BRI components
        
        Returns:
            plotly.graph_objects.Figure: Box plot analysis
        """
        # Prepare data for box plots
        df = self.bri_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')
        df['quarter'] = df['date'].dt.to_period('Q')
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'BRI Distribution by Month',
                'BRI Components Distribution',
                'BRI Distribution by Quarter',
                'Risk Level Distribution'
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. BRI by Month
        monthly_data = []
        months = []
        for month in df['month'].unique():
            month_data = df[df['month'] == month]['BRI']
            monthly_data.append(month_data.values)
            months.append(str(month))
        
        for i, (month, data) in enumerate(zip(months, monthly_data)):
            fig.add_trace(
                go.Box(
                    y=data,
                    name=month,
                    marker_color=self.professional_colors['primary'],
                    boxpoints='outliers',
                    jitter=0.3
                ),
                row=1, col=1
            )
        
        # 2. BRI Components
        components = ['sent_vol_score', 'news_tone_score', 'herding_score', 'polarity_skew_score', 'event_density_score']
        component_names = ['Sentiment Volatility', 'News Tone', 'Media Herding', 'Polarity Skew', 'Event Density']
        colors = [self.professional_colors['risk_low'], self.professional_colors['risk_moderate'], 
                 self.professional_colors['risk_high'], self.professional_colors['accent'], self.professional_colors['neutral']]
        
        for component, name, color in zip(components, component_names, colors):
            if component in df.columns:
                fig.add_trace(
                    go.Box(
                        y=df[component],
                        name=name,
                        marker_color=color,
                        boxpoints='outliers'
                    ),
                    row=1, col=2
                )
        
        # 3. BRI by Quarter
        quarterly_data = []
        quarters = []
        for quarter in df['quarter'].unique():
            quarter_data = df[df['quarter'] == quarter]['BRI']
            quarterly_data.append(quarter_data.values)
            quarters.append(str(quarter))
        
        for i, (quarter, data) in enumerate(zip(quarters, quarterly_data)):
            fig.add_trace(
                go.Box(
                    y=data,
                    name=quarter,
                    marker_color=self.professional_colors['secondary'],
                    boxpoints='outliers'
                ),
                row=2, col=1
            )
        
        # 4. Risk Level Distribution
        df['risk_level'] = pd.cut(df['BRI'], bins=[0, 30, 60, 100], labels=['Low', 'Moderate', 'High'])
        risk_colors = [self.professional_colors['risk_low'], self.professional_colors['risk_moderate'], self.professional_colors['risk_high']]
        
        for risk_level, color in zip(['Low', 'Moderate', 'High'], risk_colors):
            risk_data = df[df['risk_level'] == risk_level]['BRI']
            if not risk_data.empty:
                fig.add_trace(
                    go.Box(
                        y=risk_data,
                        name=f'{risk_level} Risk',
                        marker_color=color,
                        boxpoints='outliers'
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title=dict(
                text='BRI Distribution Analysis - Box Plots',
                font=dict(color=self.professional_colors['primary'], size=18, family='Inter')
            ),
            height=800,
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(color=self.professional_colors['primary'], family='Inter'),
            showlegend=True
        )
        
        return fig
    
    def create_correlation_heatmap(self):
        """
        Create correlation heatmap for BRI components and market data
        
        Returns:
            plotly.graph_objects.Figure: Correlation heatmap
        """
        # Prepare correlation data
        df = self.bri_data.copy()
        
        # Select numeric columns for correlation
        numeric_cols = ['BRI', 'sent_vol_score', 'news_tone_score', 'herding_score', 'polarity_skew_score', 'event_density_score']
        correlation_data = df[numeric_cols].corr()
        
        # Create custom colorscale
        colorscale = [
            [0.0, self.professional_colors['risk_high']],  # Red for negative correlation
            [0.5, '#FFFFFF'],  # White for no correlation
            [1.0, self.professional_colors['risk_low']]   # Green for positive correlation
        ]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_data.values,
            x=correlation_data.columns,
            y=correlation_data.columns,
            colorscale=colorscale,
            zmin=-1,
            zmax=1,
            text=np.round(correlation_data.values, 2),
            texttemplate='%{text}',
            textfont={'size': 12, 'color': 'black'},
            hoverongaps=False,
            hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text='BRI Components Correlation Matrix',
                font=dict(color=self.professional_colors['primary'], size=18, family='Inter')
            ),
            height=600,
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(color=self.professional_colors['primary'], family='Inter'),
            xaxis=dict(side='bottom'),
            yaxis=dict(autorange='reversed')
        )
        
        return fig
    
    def create_violin_plot_analysis(self):
        """
        Create violin plots for BRI distribution analysis
        
        Returns:
            plotly.graph_objects.Figure: Violin plot analysis
        """
        df = self.bri_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['risk_level'] = pd.cut(df['BRI'], bins=[0, 30, 60, 100], labels=['Low', 'Moderate', 'High'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'BRI Distribution by Risk Level',
                'BRI Distribution by Quarter',
                'BRI Distribution by Month',
                'BRI Components Violin Plot'
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. Risk Level Distribution
        risk_colors = [self.professional_colors['risk_low'], self.professional_colors['risk_moderate'], self.professional_colors['risk_high']]
        for risk_level, color in zip(['Low', 'Moderate', 'High'], risk_colors):
            risk_data = df[df['risk_level'] == risk_level]['BRI']
            if not risk_data.empty:
                fig.add_trace(
                    go.Violin(
                        y=risk_data,
                        name=f'{risk_level} Risk',
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=color,
                        opacity=0.7,
                        line_color=color
                    ),
                    row=1, col=1
                )
        
        # 2. Quarterly Distribution
        quarter_colors = [self.professional_colors['primary'], self.professional_colors['secondary'], 
                         self.professional_colors['accent'], self.professional_colors['neutral']]
        for quarter, color in zip([1, 2, 3, 4], quarter_colors):
            quarter_data = df[df['quarter'] == quarter]['BRI']
            if not quarter_data.empty:
                fig.add_trace(
                    go.Violin(
                        y=quarter_data,
                        name=f'Q{quarter}',
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=color,
                        opacity=0.7,
                        line_color=color
                    ),
                    row=1, col=2
                )
        
        # 3. Monthly Distribution
        month_colors = px.colors.qualitative.Set3
        for month in range(1, 13):
            month_data = df[df['month'] == month]['BRI']
            if not month_data.empty:
                fig.add_trace(
                    go.Violin(
                        y=month_data,
                        name=f'Month {month}',
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=month_colors[month % len(month_colors)],
                        opacity=0.6,
                        line_color=month_colors[month % len(month_colors)]
                    ),
                    row=2, col=1
                )
        
        # 4. Components Violin Plot
        components = ['sent_vol_score', 'news_tone_score', 'herding_score', 'polarity_skew_score', 'event_density_score']
        component_names = ['Sentiment Volatility', 'News Tone', 'Media Herding', 'Polarity Skew', 'Event Density']
        component_colors = [self.professional_colors['risk_low'], self.professional_colors['risk_moderate'], 
                           self.professional_colors['risk_high'], self.professional_colors['accent'], self.professional_colors['neutral']]
        
        for component, name, color in zip(components, component_names, component_colors):
            if component in df.columns:
                fig.add_trace(
                    go.Violin(
                        y=df[component],
                        name=name,
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=color,
                        opacity=0.7,
                        line_color=color
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title=dict(
                text='BRI Distribution Analysis - Violin Plots',
                font=dict(color=self.professional_colors['primary'], size=18, family='Inter')
            ),
            height=800,
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(color=self.professional_colors['primary'], family='Inter'),
            showlegend=True
        )
        
        return fig
    
    def create_3d_surface_plot(self):
        """
        Create 3D surface plot for BRI analysis
        
        Returns:
            plotly.graph_objects.Figure: 3D surface plot
        """
        df = self.bri_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Create 3D surface data
        months = df['month'].unique()
        days = df['day_of_year'].unique()
        
        # Create grid for surface plot
        X, Y = np.meshgrid(months, days)
        Z = np.zeros_like(X, dtype=float)
        
        # Fill Z values with BRI data
        for i, month in enumerate(months):
            for j, day in enumerate(days):
                month_day_data = df[(df['month'] == month) & (df['day_of_year'] == day)]
                if not month_day_data.empty:
                    Z[j, i] = month_day_data['BRI'].mean()
                else:
                    Z[j, i] = np.nan
        
        # Create 3D surface plot
        fig = go.Figure(data=go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale=[[0, self.professional_colors['risk_low']], 
                       [0.5, self.professional_colors['risk_moderate']], 
                       [1, self.professional_colors['risk_high']]],
            name='BRI Surface',
            hovertemplate='<b>Month:</b> %{x}<br><b>Day of Year:</b> %{y}<br><b>BRI:</b> %{z:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text='BRI 3D Surface Analysis',
                font=dict(color=self.professional_colors['primary'], size=18, family='Inter')
            ),
            scene=dict(
                xaxis_title='Month',
                yaxis_title='Day of Year',
                zaxis_title='BRI Level',
                bgcolor='#FFFFFF',
                xaxis=dict(gridcolor='#E2E8F0'),
                yaxis=dict(gridcolor='#E2E8F0'),
                zaxis=dict(gridcolor='#E2E8F0')
            ),
            height=600,
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(color=self.professional_colors['primary'], family='Inter')
        )
        
        return fig
    
    def create_advanced_histogram(self):
        """
        Create advanced histogram with multiple distributions
        
        Returns:
            plotly.graph_objects.Figure: Advanced histogram
        """
        df = self.bri_data.copy()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'BRI Distribution',
                'BRI Components Distribution',
                'Risk Level Frequency',
                'BRI vs Normal Distribution'
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. BRI Distribution
        fig.add_trace(
            go.Histogram(
                x=df['BRI'],
                nbinsx=30,
                name='BRI Distribution',
                marker_color=self.professional_colors['primary'],
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # 2. Components Distribution
        components = ['sent_vol_score', 'news_tone_score', 'herding_score', 'polarity_skew_score', 'event_density_score']
        component_names = ['Sentiment Volatility', 'News Tone', 'Media Herding', 'Polarity Skew', 'Event Density']
        colors = [self.professional_colors['risk_low'], self.professional_colors['risk_moderate'], 
                 self.professional_colors['risk_high'], self.professional_colors['accent'], self.professional_colors['neutral']]
        
        for component, name, color in zip(components, component_names, colors):
            if component in df.columns:
                fig.add_trace(
                    go.Histogram(
                        x=df[component],
                        nbinsx=20,
                        name=name,
                        marker_color=color,
                        opacity=0.6
                    ),
                    row=1, col=2
                )
        
        # 3. Risk Level Frequency
        df['risk_level'] = pd.cut(df['BRI'], bins=[0, 30, 60, 100], labels=['Low', 'Moderate', 'High'])
        risk_counts = df['risk_level'].value_counts()
        
        fig.add_trace(
            go.Bar(
                x=risk_counts.index,
                y=risk_counts.values,
                name='Risk Level Frequency',
                marker_color=[self.professional_colors['risk_low'], self.professional_colors['risk_moderate'], self.professional_colors['risk_high']]
            ),
            row=2, col=1
        )
        
        # 4. BRI vs Normal Distribution
        bri_mean = df['BRI'].mean()
        bri_std = df['BRI'].std()
        x_range = np.linspace(df['BRI'].min(), df['BRI'].max(), 100)
        normal_dist = stats.norm.pdf(x_range, bri_mean, bri_std)
        
        fig.add_trace(
            go.Histogram(
                x=df['BRI'],
                nbinsx=30,
                name='BRI Data',
                marker_color=self.professional_colors['primary'],
                opacity=0.7,
                histnorm='probability density'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=normal_dist,
                mode='lines',
                name='Normal Distribution',
                line=dict(color=self.professional_colors['accent'], width=3)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=dict(
                text='BRI Distribution Analysis - Advanced Histograms',
                font=dict(color=self.professional_colors['primary'], size=18, family='Inter')
            ),
            height=800,
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(color=self.professional_colors['primary'], family='Inter'),
            showlegend=True
        )
        
        return fig
    
    def generate_all_advanced_charts(self):
        """Generate all advanced chart types"""
        return {
            'candlestick': self.create_candlestick_chart(),
            'box_plots': self.create_box_plot_analysis(),
            'correlation_heatmap': self.create_correlation_heatmap(),
            'violin_plots': self.create_violin_plot_analysis(),
            '3d_surface': self.create_3d_surface_plot(),
            'advanced_histogram': self.create_advanced_histogram()
        }
