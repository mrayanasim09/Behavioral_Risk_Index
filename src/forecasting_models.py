"""
Advanced Forecasting Models for BRI Dashboard
Implements LSTM, Random Forest, XGBoost, and ARIMA models for BRI prediction
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ForecastingModels:
    """Advanced forecasting models for BRI prediction"""
    
    def __init__(self, bri_data, market_data=None):
        self.bri_data = bri_data.copy()
        self.market_data = market_data
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        
    def prepare_features(self, lookback_days=30):
        """
        Prepare features for machine learning models
        
        Args:
            lookback_days (int): Number of days to look back for features
            
        Returns:
            tuple: (X, y) features and targets
        """
        df = self.bri_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Create lagged features
        for i in range(1, lookback_days + 1):
            df[f'bri_lag_{i}'] = df['BRI'].shift(i)
            df[f'sent_vol_lag_{i}'] = df['sent_vol_score'].shift(i) if 'sent_vol_score' in df.columns else 0
            df[f'news_tone_lag_{i}'] = df['news_tone_score'].shift(i) if 'news_tone_score' in df.columns else 0
            df[f'herding_lag_{i}'] = df['herding_score'].shift(i) if 'herding_score' in df.columns else 0
        
        # Create technical indicators
        df['bri_ma_7'] = df['BRI'].rolling(window=7).mean()
        df['bri_ma_30'] = df['BRI'].rolling(window=30).mean()
        df['bri_std_7'] = df['BRI'].rolling(window=7).std()
        df['bri_std_30'] = df['BRI'].rolling(window=30).std()
        df['bri_momentum'] = df['BRI'].pct_change(7)
        df['bri_volatility'] = df['BRI'].rolling(window=7).std()
        
        # Create target variable (next day BRI)
        df['target'] = df['BRI'].shift(-1)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Select features
        feature_cols = [col for col in df.columns if col not in ['date', 'BRI', 'target']]
        X = df[feature_cols].values
        y = df['target'].values
        
        return X, y, feature_cols
    
    def train_random_forest(self, lookback_days=30, test_size=0.2):
        """
        Train Random Forest model
        
        Args:
            lookback_days (int): Number of days to look back
            test_size (float): Test set size
            
        Returns:
            dict: Model performance metrics
        """
        try:
            X, y, feature_cols = self.prepare_features(lookback_days)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = rf_model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Store model and scaler
            self.models['random_forest'] = rf_model
            self.scalers['random_forest'] = scaler
            self.performance_metrics['random_forest'] = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': rmse,
                'feature_importance': dict(zip(feature_cols, rf_model.feature_importances_))
            }
            
            return self.performance_metrics['random_forest']
            
        except Exception as e:
            print(f"Error training Random Forest: {e}")
            return None
    
    def train_xgboost(self, lookback_days=30, test_size=0.2):
        """
        Train XGBoost model
        
        Args:
            lookback_days (int): Number of days to look back
            test_size (float): Test set size
            
        Returns:
            dict: Model performance metrics
        """
        try:
            X, y, feature_cols = self.prepare_features(lookback_days)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            xgb_model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = xgb_model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Store model and scaler
            self.models['xgboost'] = xgb_model
            self.scalers['xgboost'] = scaler
            self.performance_metrics['xgboost'] = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': rmse,
                'feature_importance': dict(zip(feature_cols, xgb_model.feature_importances_))
            }
            
            return self.performance_metrics['xgboost']
            
        except Exception as e:
            print(f"Error training XGBoost: {e}")
            return None
    
    def train_arima(self, order=(1, 1, 1)):
        """
        Train ARIMA model
        
        Args:
            order (tuple): ARIMA order (p, d, q)
            
        Returns:
            dict: Model performance metrics
        """
        try:
            df = self.bri_data.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Use BRI data for ARIMA
            bri_series = df['BRI'].values
            
            # Split data
            split_point = int(len(bri_series) * 0.8)
            train_data = bri_series[:split_point]
            test_data = bri_series[split_point:]
            
            # Train ARIMA model
            arima_model = ARIMA(train_data, order=order)
            arima_fitted = arima_model.fit()
            
            # Make predictions
            predictions = arima_fitted.forecast(steps=len(test_data))
            
            # Calculate metrics
            mse = mean_squared_error(test_data, predictions)
            mae = mean_absolute_error(test_data, predictions)
            r2 = r2_score(test_data, predictions)
            rmse = np.sqrt(mse)
            
            # Store model
            self.models['arima'] = arima_fitted
            self.performance_metrics['arima'] = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': rmse,
                'aic': arima_fitted.aic,
                'bic': arima_fitted.bic
            }
            
            return self.performance_metrics['arima']
            
        except Exception as e:
            print(f"Error training ARIMA: {e}")
            return None
    
    def train_exponential_smoothing(self):
        """
        Train Exponential Smoothing model
        
        Returns:
            dict: Model performance metrics
        """
        try:
            df = self.bri_data.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Use BRI data for exponential smoothing
            bri_series = df['BRI'].values
            
            # Split data
            split_point = int(len(bri_series) * 0.8)
            train_data = bri_series[:split_point]
            test_data = bri_series[split_point:]
            
            # Train exponential smoothing model
            exp_model = ExponentialSmoothing(train_data, trend='add', seasonal=None)
            exp_fitted = exp_model.fit()
            
            # Make predictions
            predictions = exp_fitted.forecast(steps=len(test_data))
            
            # Calculate metrics
            mse = mean_squared_error(test_data, predictions)
            mae = mean_absolute_error(test_data, predictions)
            r2 = r2_score(test_data, predictions)
            rmse = np.sqrt(mse)
            
            # Store model
            self.models['exponential_smoothing'] = exp_fitted
            self.performance_metrics['exponential_smoothing'] = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': rmse,
                'aic': exp_fitted.aic,
                'bic': exp_fitted.bic
            }
            
            return self.performance_metrics['exponential_smoothing']
            
        except Exception as e:
            print(f"Error training Exponential Smoothing: {e}")
            return None
    
    def predict_future(self, model_name, days_ahead=30):
        """
        Predict future BRI values using trained model
        
        Args:
            model_name (str): Name of the model to use
            days_ahead (int): Number of days to predict ahead
            
        Returns:
            dict: Predictions and confidence intervals
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found. Train the model first.")
            
            df = self.bri_data.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            if model_name in ['random_forest', 'xgboost']:
                # Prepare features for ML models
                X, y, feature_cols = self.prepare_features()
                scaler = self.scalers[model_name]
                model = self.models[model_name]
                
                # Get last available features
                last_features = X[-1:].copy()
                predictions = []
                
                for _ in range(days_ahead):
                    # Make prediction
                    pred = model.predict(scaler.transform(last_features))[0]
                    predictions.append(pred)
                    
                    # Update features for next prediction
                    # Shift all lagged features
                    for i in range(len(feature_cols) - 1, 0, -1):
                        if i < len(feature_cols) - 1:
                            last_features[0, i] = last_features[0, i-1]
                    last_features[0, 0] = pred
                
            elif model_name == 'arima':
                # ARIMA prediction
                model = self.models[model_name]
                predictions = model.forecast(steps=days_ahead)
                
            elif model_name == 'exponential_smoothing':
                # Exponential smoothing prediction
                model = self.models[model_name]
                predictions = model.forecast(steps=days_ahead)
            
            # Generate future dates
            last_date = df['date'].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq='D')
            
            # Calculate confidence intervals (simple approach)
            recent_std = df['BRI'].tail(30).std()
            confidence_level = 0.95
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            
            upper_bound = [p + z_score * recent_std for p in predictions]
            lower_bound = [p - z_score * recent_std for p in predictions]
            
            return {
                'dates': future_dates.strftime('%Y-%m-%d').tolist(),
                'predictions': predictions,
                'upper_bound': upper_bound,
                'lower_bound': lower_bound,
                'confidence_level': confidence_level,
                'model_name': model_name
            }
            
        except Exception as e:
            print(f"Error predicting future: {e}")
            return None
    
    def create_forecasting_comparison(self, days_ahead=30):
        """
        Create comparison of all forecasting models
        
        Args:
            days_ahead (int): Number of days to predict ahead
            
        Returns:
            plotly.graph_objects.Figure: Forecasting comparison chart
        """
        try:
            # Train all models
            self.train_random_forest()
            self.train_xgboost()
            self.train_arima()
            self.train_exponential_smoothing()
            
            # Get predictions from all models
            predictions = {}
            for model_name in ['random_forest', 'xgboost', 'arima', 'exponential_smoothing']:
                pred_data = self.predict_future(model_name, days_ahead)
                if pred_data:
                    predictions[model_name] = pred_data
            
            # Create figure
            fig = go.Figure()
            
            # Add historical data
            df = self.bri_data.copy()
            df['date'] = pd.to_datetime(df['date'])
            
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['BRI'],
                mode='lines',
                name='Historical BRI',
                line=dict(color='#1A365D', width=2)
            ))
            
            # Add predictions from each model
            colors = ['#E53E3E', '#D69E2E', '#38A169', '#3182CE']
            model_names = ['Random Forest', 'XGBoost', 'ARIMA', 'Exponential Smoothing']
            
            for i, (model_name, model_display_name) in enumerate(zip(predictions.keys(), model_names)):
                pred_data = predictions[model_name]
                
                # Add prediction line
                fig.add_trace(go.Scatter(
                    x=pred_data['dates'],
                    y=pred_data['predictions'],
                    mode='lines',
                    name=f'{model_display_name} Prediction',
                    line=dict(color=colors[i], width=2, dash='dash')
                ))
                
                # Add confidence interval
                fig.add_trace(go.Scatter(
                    x=pred_data['dates'] + pred_data['dates'][::-1],
                    y=pred_data['upper_bound'] + pred_data['lower_bound'][::-1],
                    fill='tonexty' if i > 0 else 'tonext',
                    fillcolor=f'rgba({int(colors[i][1:3], 16)}, {int(colors[i][3:5], 16)}, {int(colors[i][5:7], 16)}, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{model_display_name} Confidence Interval',
                    hoverinfo='skip'
                ))
            
            fig.update_layout(
                title=dict(
                    text=f'BRI Forecasting Comparison - {days_ahead} Days Ahead',
                    font=dict(color='#1A365D', size=18, family='Inter')
                ),
                xaxis_title='Date',
                yaxis_title='BRI Level',
                height=600,
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                font=dict(color='#1A365D', family='Inter'),
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating forecasting comparison: {e}")
            return None
    
    def create_model_performance_comparison(self):
        """
        Create model performance comparison chart
        
        Returns:
            plotly.graph_objects.Figure: Performance comparison chart
        """
        try:
            # Train all models if not already trained
            if not self.performance_metrics:
                self.train_random_forest()
                self.train_xgboost()
                self.train_arima()
                self.train_exponential_smoothing()
            
            # Prepare data for comparison
            models = list(self.performance_metrics.keys())
            metrics = ['mse', 'mae', 'rmse', 'r2']
            metric_names = ['Mean Squared Error', 'Mean Absolute Error', 'RMSE', 'R² Score']
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=metric_names,
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            colors = ['#E53E3E', '#D69E2E', '#38A169', '#3182CE']
            
            for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                values = [self.performance_metrics[model].get(metric, 0) for model in models]
                model_names = [model.replace('_', ' ').title() for model in models]
                
                fig.add_trace(
                    go.Bar(
                        x=model_names,
                        y=values,
                        name=metric_name,
                        marker_color=colors[i],
                        text=[f'{v:.4f}' for v in values],
                        textposition='auto'
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title=dict(
                    text='Model Performance Comparison',
                    font=dict(color='#1A365D', size=18, family='Inter')
                ),
                height=600,
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                font=dict(color='#1A365D', family='Inter'),
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating performance comparison: {e}")
            return None
    
    def get_best_model(self):
        """
        Get the best performing model based on R² score
        
        Returns:
            str: Name of the best model
        """
        if not self.performance_metrics:
            return None
        
        best_model = max(self.performance_metrics.keys(), 
                        key=lambda x: self.performance_metrics[x].get('r2', -float('inf')))
        return best_model
    
    def generate_forecasting_report(self, days_ahead=30):
        """
        Generate comprehensive forecasting report
        
        Args:
            days_ahead (int): Number of days to predict ahead
            
        Returns:
            dict: Comprehensive forecasting report
        """
        try:
            # Train all models
            self.train_random_forest()
            self.train_xgboost()
            self.train_arima()
            self.train_exponential_smoothing()
            
            # Get best model
            best_model = self.get_best_model()
            
            # Get predictions from best model
            best_predictions = self.predict_future(best_model, days_ahead)
            
            # Create comparison chart
            comparison_chart = self.create_forecasting_comparison(days_ahead)
            
            # Create performance chart
            performance_chart = self.create_model_performance_comparison()
            
            return {
                'best_model': best_model,
                'best_model_performance': self.performance_metrics[best_model],
                'all_model_performance': self.performance_metrics,
                'best_predictions': best_predictions,
                'comparison_chart': comparison_chart,
                'performance_chart': performance_chart,
                'forecasting_horizon': days_ahead,
                'generated_at': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error generating forecasting report: {e}")
            return None
