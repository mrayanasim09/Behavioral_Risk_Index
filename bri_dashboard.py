"""
Behavioral Risk Index Dashboard
Real-time BRI monitoring with Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import os
import json

# Page configuration
st.set_page_config(
    page_title="Behavioral Risk Index Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        color: #ff4444;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffaa00;
        font-weight: bold;
    }
    .risk-low {
        color: #00aa44;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_bri_data():
    """Load BRI data with caching"""
    try:
        # Try to load enhanced BRI data first
        if os.path.exists('output/enhanced/bri_timeseries_enhanced.csv'):
            return pd.read_csv('output/enhanced/bri_timeseries_enhanced.csv')
        elif os.path.exists('output/complete/bri_timeseries.csv'):
            return pd.read_csv('output/complete/bri_timeseries.csv')
        else:
            return None
    except Exception as e:
        st.error(f"Error loading BRI data: {e}")
        return None

@st.cache_data
def load_market_data():
    """Load market data with caching"""
    try:
        if os.path.exists('output/enhanced/market_data.csv'):
            return pd.read_csv('output/enhanced/market_data.csv')
        elif os.path.exists('output/complete/market_data.csv'):
            return pd.read_csv('output/complete/market_data.csv')
        else:
            return None
    except Exception as e:
        st.error(f"Error loading market data: {e}")
        return None

@st.cache_data
def load_validation_results():
    """Load validation results with caching"""
    try:
        if os.path.exists('output/enhanced/predictive_modeling_results.json'):
            with open('output/enhanced/predictive_modeling_results.json', 'r') as f:
                return json.load(f)
        elif os.path.exists('output/complete/validation_results.json'):
            with open('output/complete/validation_results.json', 'r') as f:
                return json.load(f)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading validation results: {e}")
        return None

def get_risk_level(bri_value):
    """Determine risk level based on BRI value"""
    if bri_value >= 80:
        return "HIGH", "risk-high"
    elif bri_value >= 60:
        return "MEDIUM", "risk-medium"
    else:
        return "LOW", "risk-low"

def create_bri_timeseries_plot(bri_data):
    """Create BRI time series plot"""
    fig = go.Figure()
    
    # Add BRI line
    fig.add_trace(go.Scatter(
        x=bri_data['date'],
        y=bri_data['BRI'],
        mode='lines',
        name='BRI',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Date:</b> %{x}<br><b>BRI:</b> %{y:.2f}<extra></extra>'
    ))
    
    # Add risk level bands
    fig.add_hline(y=80, line_dash="dash", line_color="red", 
                  annotation_text="High Risk (80+)", annotation_position="top right")
    fig.add_hline(y=60, line_dash="dash", line_color="orange", 
                  annotation_text="Medium Risk (60+)", annotation_position="top right")
    fig.add_hline(y=40, line_dash="dash", line_color="green", 
                  annotation_text="Low Risk (40-)", annotation_position="bottom right")
    
    # Add crisis period highlights
    crisis_periods = [
        ('2022-02-24', '2022-03-31', 'Russia-Ukraine War', 'red'),
        ('2022-06-01', '2022-10-31', 'Bear Market', 'orange'),
        ('2023-03-01', '2023-03-31', 'Banking Crisis', 'purple'),
        ('2023-05-01', '2023-06-30', 'Debt Ceiling', 'brown')
    ]
    
    for start, end, label, color in crisis_periods:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=color, opacity=0.1,
            annotation_text=label, annotation_position="top left"
        )
    
    fig.update_layout(
        title="Behavioral Risk Index Over Time",
        xaxis_title="Date",
        yaxis_title="BRI (0-100)",
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_bri_vix_comparison(bri_data, market_data):
    """Create BRI vs VIX comparison plot"""
    # Merge data
    bri_data['date'] = pd.to_datetime(bri_data['date'])
    market_data['Date'] = pd.to_datetime(market_data['Date'])
    
    # Get VIX column
    vix_col = 'Close_^VIX' if 'Close_^VIX' in market_data.columns else 'Close'
    
    merged_data = pd.merge(
        bri_data[['date', 'BRI']], 
        market_data[['Date', vix_col]], 
        left_on='date', 
        right_on='Date', 
        how='inner'
    )
    
    if merged_data.empty:
        return None
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('BRI vs VIX Time Series', 'BRI vs VIX Scatter', 
                       'Rolling Correlation', 'Distribution Comparison'),
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Time series comparison
    fig.add_trace(
        go.Scatter(x=merged_data['date'], y=merged_data['BRI'], 
                  name='BRI', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=merged_data['date'], y=merged_data[vix_col], 
                  name='VIX', line=dict(color='red')),
        row=1, col=1, secondary_y=True
    )
    
    # Scatter plot
    fig.add_trace(
        go.Scatter(x=merged_data['BRI'], y=merged_data[vix_col], 
                  mode='markers', name='BRI vs VIX', 
                  marker=dict(color='green', opacity=0.6)),
        row=1, col=2
    )
    
    # Rolling correlation
    rolling_corr = merged_data['BRI'].rolling(window=30).corr(merged_data[vix_col])
    fig.add_trace(
        go.Scatter(x=merged_data['date'], y=rolling_corr, 
                  name='30-Day Rolling Correlation', line=dict(color='purple')),
        row=2, col=1
    )
    
    # Distribution comparison
    fig.add_trace(
        go.Histogram(x=merged_data['BRI'], name='BRI Distribution', 
                    opacity=0.7, nbinsx=30),
        row=2, col=2
    )
    fig.add_trace(
        go.Histogram(x=merged_data[vix_col], name='VIX Distribution', 
                    opacity=0.7, nbinsx=30),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="BRI vs VIX Comprehensive Analysis",
        height=800,
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="BRI", row=1, col=1)
    fig.update_yaxes(title_text="VIX", row=1, col=1, secondary_y=True)
    fig.update_xaxes(title_text="BRI", row=1, col=2)
    fig.update_yaxes(title_text="VIX", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Rolling Correlation", row=2, col=1)
    fig.update_xaxes(title_text="Value", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    return fig

def create_feature_analysis(bri_data):
    """Create feature analysis plots"""
    # Get feature columns
    feature_cols = [col for col in bri_data.columns if col.endswith('_score')]
    
    if not feature_cols:
        return None
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Feature Importance', 'Feature Correlation Matrix', 
                       'Feature Time Series', 'Feature Distribution'),
        specs=[[{"type": "bar"}, {"type": "heatmap"}],
               [{"type": "scatter"}, {"type": "histogram"}]]
    )
    
    # Feature importance
    feature_importance = bri_data[feature_cols].mean()
    fig.add_trace(
        go.Bar(x=feature_importance.index, y=feature_importance.values, 
               name='Feature Importance'),
        row=1, col=1
    )
    
    # Feature correlation matrix
    corr_matrix = bri_data[feature_cols].corr()
    fig.add_trace(
        go.Heatmap(z=corr_matrix.values, 
                  x=corr_matrix.columns, 
                  y=corr_matrix.columns,
                  colorscale='RdBu', zmid=0),
        row=1, col=2
    )
    
    # Feature time series (first 3 features)
    for i, col in enumerate(feature_cols[:3]):
        fig.add_trace(
            go.Scatter(x=bri_data['date'], y=bri_data[col], 
                      name=col, mode='lines'),
            row=2, col=1
        )
    
    # Feature distribution
    for col in feature_cols[:2]:  # First 2 features
        fig.add_trace(
            go.Histogram(x=bri_data[col], name=col, opacity=0.7),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Feature Analysis",
        height=800,
        showlegend=True
    )
    
    return fig

def main():
    """Main dashboard function"""
    # Header
    st.markdown('<h1 class="main-header">📊 Behavioral Risk Index Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    bri_data = load_bri_data()
    market_data = load_market_data()
    validation_results = load_validation_results()
    
    if bri_data is None:
        st.error("No BRI data found. Please run the pipeline first.")
        return
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    
    # Date range selector
    if not bri_data.empty:
        bri_data['date'] = pd.to_datetime(bri_data['date'])
        min_date = bri_data['date'].min().date()
        max_date = bri_data['date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Filter data by date range
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_data = bri_data[
                (bri_data['date'].dt.date >= start_date) & 
                (bri_data['date'].dt.date <= end_date)
            ]
        else:
            filtered_data = bri_data
    else:
        filtered_data = bri_data
    
    # Main metrics
    if not filtered_data.empty:
        current_bri = filtered_data['BRI'].iloc[-1]
        avg_bri = filtered_data['BRI'].mean()
        max_bri = filtered_data['BRI'].max()
        min_bri = filtered_data['BRI'].min()
        
        risk_level, risk_class = get_risk_level(current_bri)
        
        # Create metric columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current BRI",
                f"{current_bri:.2f}",
                delta=f"{current_bri - avg_bri:.2f}",
                help="Current Behavioral Risk Index value"
            )
        
        with col2:
            st.metric(
                "Risk Level",
                risk_level,
                help="Current risk level based on BRI value"
            )
        
        with col3:
            st.metric(
                "Average BRI",
                f"{avg_bri:.2f}",
                help="Average BRI over selected period"
            )
        
        with col4:
            st.metric(
                "BRI Range",
                f"{min_bri:.1f} - {max_bri:.1f}",
                help="BRI range over selected period"
            )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 BRI Overview", 
        "🔍 BRI vs VIX Analysis", 
        "⚙️ Feature Analysis", 
        "📊 Model Performance", 
        "ℹ️ About"
    ])
    
    with tab1:
        st.header("BRI Overview")
        
        if not filtered_data.empty:
            # BRI time series plot
            fig = create_bri_timeseries_plot(filtered_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # BRI statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("BRI Statistics")
                stats_data = {
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th Percentile', '75th Percentile'],
                    'Value': [
                        f"{filtered_data['BRI'].mean():.2f}",
                        f"{filtered_data['BRI'].median():.2f}",
                        f"{filtered_data['BRI'].std():.2f}",
                        f"{filtered_data['BRI'].min():.2f}",
                        f"{filtered_data['BRI'].max():.2f}",
                        f"{filtered_data['BRI'].quantile(0.25):.2f}",
                        f"{filtered_data['BRI'].quantile(0.75):.2f}"
                    ]
                }
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
            
            with col2:
                st.subheader("Risk Level Distribution")
                risk_counts = {
                    'High Risk (80+)': len(filtered_data[filtered_data['BRI'] >= 80]),
                    'Medium Risk (60-79)': len(filtered_data[(filtered_data['BRI'] >= 60) & (filtered_data['BRI'] < 80)]),
                    'Low Risk (<60)': len(filtered_data[filtered_data['BRI'] < 60])
                }
                
                fig_pie = px.pie(
                    values=list(risk_counts.values()),
                    names=list(risk_counts.keys()),
                    title="Risk Level Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("No data available for the selected date range.")
    
    with tab2:
        st.header("BRI vs VIX Analysis")
        
        if market_data is not None and not filtered_data.empty:
            fig = create_bri_vix_comparison(filtered_data, market_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation analysis
                bri_data_temp = filtered_data.copy()
                bri_data_temp['date'] = pd.to_datetime(bri_data_temp['date'])
                market_data_temp = market_data.copy()
                market_data_temp['Date'] = pd.to_datetime(market_data_temp['Date'])
                
                vix_col = 'Close_^VIX' if 'Close_^VIX' in market_data_temp.columns else 'Close'
                
                merged_temp = pd.merge(
                    bri_data_temp[['date', 'BRI']], 
                    market_data_temp[['Date', vix_col]], 
                    left_on='date', 
                    right_on='Date', 
                    how='inner'
                )
                
                if not merged_temp.empty:
                    correlation = merged_temp['BRI'].corr(merged_temp[vix_col])
                    st.metric("BRI-VIX Correlation", f"{correlation:.4f}")
            else:
                st.warning("Could not create BRI vs VIX comparison plot.")
        else:
            st.warning("Market data not available for comparison.")
    
    with tab3:
        st.header("Feature Analysis")
        
        if not filtered_data.empty:
            fig = create_feature_analysis(filtered_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No feature data available for analysis.")
        else:
            st.warning("No data available for feature analysis.")
    
    with tab4:
        st.header("Model Performance")
        
        if validation_results:
            st.subheader("Predictive Modeling Results")
            
            if 'model_comparison' in validation_results:
                comparison_df = pd.DataFrame(validation_results['model_comparison'])
                st.dataframe(comparison_df, use_container_width=True)
                
                # Best model metrics
                best_model = validation_results.get('best_model', 'N/A')
                best_r2 = validation_results.get('best_r2', 'N/A')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Best Model", best_model)
                with col2:
                    st.metric("Best R² Score", f"{best_r2:.4f}")
            
            if 'data_info' in validation_results:
                st.subheader("Data Information")
                data_info = validation_results['data_info']
                st.json(data_info)
        else:
            st.warning("No validation results available.")
    
    with tab5:
        st.header("About the Behavioral Risk Index")
        
        st.markdown("""
        ### What is the Behavioral Risk Index (BRI)?
        
        The Behavioral Risk Index is a novel quantitative measure that captures narrative concentration 
        and herding behavior in financial markets through the analysis of:
        
        - **Social Media Sentiment**: Reddit posts from 48 finance subreddits
        - **News Tone**: GDELT global news events with sentiment analysis
        - **Market Attention**: Media attention and herding behavior patterns
        
        ### Key Features
        
        - **Scale**: 0-100 behavioral risk index
        - **Frequency**: Daily observations
        - **Components**: 5 core behavioral risk indicators
        - **Optimization**: Data-driven weight optimization
        - **Validation**: Machine learning-based predictive modeling
        
        ### Risk Levels
        
        - **High Risk (80+)**: Extreme behavioral risk, potential market instability
        - **Medium Risk (60-79)**: Elevated behavioral risk, increased caution needed
        - **Low Risk (<60)**: Normal behavioral risk, stable market conditions
        
        ### Applications
        
        - **Risk Management**: Portfolio risk assessment
        - **Trading Strategies**: BRI-based algorithmic trading
        - **Regulatory Monitoring**: Market stability indicators
        - **Investment Research**: Sentiment-driven analysis
        
        ### Methodology
        
        The BRI is calculated using a weighted aggregation of five behavioral indicators:
        
        1. **Sentiment Volatility** (30%): Reddit sentiment standard deviation
        2. **News Tone** (20%): GDELT average tone
        3. **Herding Intensity** (20%): Media attention patterns
        4. **Event Density** (20%): Number of major events per day
        5. **Polarity Skew** (10%): Asymmetry of sentiment distribution
        
        Weights are optimized using PCA analysis, grid search, and advanced optimization techniques.
        
        ### Data Sources
        
        - **Market Data**: Yahoo Finance (22 financial instruments)
        - **News Data**: GDELT export files (65 financial events)
        - **Social Media**: Reddit API (48 finance subreddits)
        
        ### Technical Implementation
        
        - **Backend**: Python with scikit-learn, XGBoost, TensorFlow
        - **Frontend**: Streamlit dashboard
        - **Optimization**: PCA, Grid Search, Advanced Optimization
        - **Modeling**: Random Forest, XGBoost, LSTM
        
        ---
        
        *This dashboard provides real-time monitoring of the Behavioral Risk Index for research and practical applications.*
        """)

if __name__ == "__main__":
    main()
