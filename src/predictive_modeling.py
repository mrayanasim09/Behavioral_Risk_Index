"""
Predictive Modeling for Behavioral Risk Index
Implements LSTM, Random Forest, and XGBoost for volatility prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class BRIPredictiveModeler:
    """Advanced predictive modeling for BRI-based volatility prediction"""
    
    def __init__(self, bri_data: pd.DataFrame, market_data: pd.DataFrame):
        """
        Initialize predictive modeler
        
        Args:
            bri_data: DataFrame with BRI and component features
            market_data: DataFrame with market data (VIX, returns, etc.)
        """
        self.bri_data = bri_data
        self.market_data = market_data
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def prepare_data(self, target_column: str = 'vix', lookback_days: int = 5) -> tuple:
        """Prepare data for predictive modeling"""
        print(f"Preparing data for {target_column} prediction...")
        
        # Merge BRI and market data
        merged_data = pd.merge(
            self.bri_data[['date', 'BRI'] + [col for col in self.bri_data.columns if col.endswith('_score')]], 
            self.market_data[['Date', target_column]], 
            left_on='date', 
            right_on='Date', 
            how='inner'
        ).dropna()
        
        if merged_data.empty:
            print(f"No overlapping data for {target_column}")
            return None, None, None, None
        
        # Create features
        feature_columns = [col for col in merged_data.columns if col.endswith('_score')]
        
        # Add lagged features
        for col in feature_columns + ['BRI']:
            for lag in range(1, lookback_days + 1):
                merged_data[f'{col}_lag_{lag}'] = merged_data[col].shift(lag)
        
        # Add market features
        if 'vix' in merged_data.columns:
            merged_data['vix_lag_1'] = merged_data['vix'].shift(1)
            merged_data['vix_ma_5'] = merged_data['vix'].rolling(5).mean()
            merged_data['vix_volatility'] = merged_data['vix'].rolling(5).std()
        
        # Calculate returns if available
        if '^GSPC_Close' in self.market_data.columns:
            sp500_data = pd.merge(
                merged_data[['date']], 
                self.market_data[['Date', '^GSPC_Close']], 
                left_on='date', 
                right_on='Date', 
                how='left'
            )
            sp500_data['sp500_returns'] = sp500_data['^GSPC_Close'].pct_change()
            merged_data = pd.merge(merged_data, sp500_data[['date', 'sp500_returns']], on='date', how='left')
        
        # Prepare features and target
        feature_cols = [col for col in merged_data.columns if col not in ['date', 'Date', target_column]]
        X = merged_data[feature_cols].fillna(0).values
        y = merged_data[target_column].values
        
        # Remove rows with NaN values
        valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Split data (80% train, 20% test)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Data prepared: {len(X_train)} training samples, {len(X_test)} test samples")
        print(f"Features: {len(feature_cols)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
    
    def train_random_forest(self, X_train, X_test, y_train, y_test, feature_cols) -> dict:
        """Train Random Forest model"""
        print("Training Random Forest model...")
        
        # Grid search for optimal parameters
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42)
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Grid search
        grid_search = GridSearchCV(
            rf, param_grid, cv=tscv, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Train final model with best parameters
        best_rf = grid_search.best_estimator_
        best_rf.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = best_rf.predict(X_train)
        y_pred_test = best_rf.predict(X_test)
        
        # Metrics
        train_metrics = {
            'mse': mean_squared_error(y_train, y_pred_train),
            'mae': mean_absolute_error(y_train, y_pred_train),
            'r2': r2_score(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train))
        }
        
        test_metrics = {
            'mse': mean_squared_error(y_test, y_pred_test),
            'mae': mean_absolute_error(y_test, y_pred_test),
            'r2': r2_score(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.models['random_forest'] = best_rf
        self.results['random_forest'] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance,
            'best_params': grid_search.best_params_
        }
        
        print(f"Random Forest - Test R²: {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.4f}")
        
        return {
            'model': best_rf,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance
        }
    
    def train_xgboost(self, X_train, X_test, y_train, y_test, feature_cols) -> dict:
        """Train XGBoost model"""
        print("Training XGBoost model...")
        
        # XGBoost parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Grid search
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Train final model
        best_xgb = grid_search.best_estimator_
        best_xgb.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = best_xgb.predict(X_train)
        y_pred_test = best_xgb.predict(X_test)
        
        # Metrics
        train_metrics = {
            'mse': mean_squared_error(y_train, y_pred_train),
            'mae': mean_absolute_error(y_train, y_pred_train),
            'r2': r2_score(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train))
        }
        
        test_metrics = {
            'mse': mean_squared_error(y_test, y_pred_test),
            'mae': mean_absolute_error(y_test, y_pred_test),
            'r2': r2_score(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_xgb.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.models['xgboost'] = best_xgb
        self.results['xgboost'] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance,
            'best_params': grid_search.best_params_
        }
        
        print(f"XGBoost - Test R²: {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.4f}")
        
        return {
            'model': best_xgb,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance
        }
    
    def train_lstm(self, X_train, X_test, y_train, y_test, feature_cols, lookback_days: int = 5) -> dict:
        """Train LSTM model"""
        print("Training LSTM model...")
        
        # Reshape data for LSTM (samples, timesteps, features)
        def create_sequences(X, y, lookback):
            X_seq, y_seq = [], []
            for i in range(lookback, len(X)):
                X_seq.append(X[i-lookback:i])
                y_seq.append(y[i])
            return np.array(X_seq), np.array(y_seq)
        
        # Create sequences
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, lookback_days)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, lookback_days)
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback_days, X_train.shape[1])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Train model
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=100,
            batch_size=32,
            validation_data=(X_test_seq, y_test_seq),
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )
        
        # Predictions
        y_pred_train = model.predict(X_train_seq, verbose=0).flatten()
        y_pred_test = model.predict(X_test_seq, verbose=0).flatten()
        
        # Metrics
        train_metrics = {
            'mse': mean_squared_error(y_train_seq, y_pred_train),
            'mae': mean_absolute_error(y_train_seq, y_pred_train),
            'r2': r2_score(y_train_seq, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train_seq, y_pred_train))
        }
        
        test_metrics = {
            'mse': mean_squared_error(y_test_seq, y_pred_test),
            'mae': mean_absolute_error(y_test_seq, y_pred_test),
            'r2': r2_score(y_test_seq, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test_seq, y_pred_test))
        }
        
        self.models['lstm'] = model
        self.results['lstm'] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'history': history.history
        }
        
        print(f"LSTM - Test R²: {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.4f}")
        
        return {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'history': history.history
        }
    
    def create_baseline_model(self, X_train, X_test, y_train, y_test) -> dict:
        """Create baseline model using only VIX lagged values"""
        print("Creating baseline model (VIX-only)...")
        
        # Find VIX-related features
        vix_features = [col for col in range(len(X_train[0])) if 'vix' in str(col).lower()]
        
        if not vix_features:
            print("No VIX features found for baseline")
            return {}
        
        # Use only VIX features
        X_train_baseline = X_train[:, vix_features]
        X_test_baseline = X_test[:, vix_features]
        
        # Simple linear regression baseline
        from sklearn.linear_model import LinearRegression
        baseline_model = LinearRegression()
        baseline_model.fit(X_train_baseline, y_train)
        
        # Predictions
        y_pred_train = baseline_model.predict(X_train_baseline)
        y_pred_test = baseline_model.predict(X_test_baseline)
        
        # Metrics
        train_metrics = {
            'mse': mean_squared_error(y_train, y_pred_train),
            'mae': mean_absolute_error(y_train, y_pred_train),
            'r2': r2_score(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train))
        }
        
        test_metrics = {
            'mse': mean_squared_error(y_test, y_pred_test),
            'mae': mean_absolute_error(y_test, y_pred_test),
            'r2': r2_score(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))
        }
        
        self.models['baseline'] = baseline_model
        self.results['baseline'] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
        
        print(f"Baseline (VIX-only) - Test R²: {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.4f}")
        
        return {
            'model': baseline_model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models"""
        print("Comparing model performance...")
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Train_R2': results['train_metrics']['r2'],
                'Test_R2': results['test_metrics']['r2'],
                'Train_RMSE': results['train_metrics']['rmse'],
                'Test_RMSE': results['test_metrics']['rmse'],
                'Train_MAE': results['train_metrics']['mae'],
                'Test_MAE': results['test_metrics']['mae']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test_R2', ascending=False)
        
        print("\nModel Comparison:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def create_prediction_visualizations(self, output_dir: str, X_test, y_test):
        """Create visualizations for model predictions"""
        print("Creating prediction visualizations...")
        
        plots_dir = f"{output_dir}/prediction_plots"
        import os
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot predictions for each model
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (model_name, model) in enumerate(self.models.items()):
            if i >= 4:  # Limit to 4 plots
                break
                
            ax = axes[i]
            
            # Get predictions
            if model_name == 'lstm':
                # LSTM needs sequence data
                lookback_days = 5
                X_test_seq, _ = self.create_sequences(X_test, y_test, lookback_days)
                y_pred = model.predict(X_test_seq, verbose=0).flatten()
                y_actual = y_test[lookback_days:]
            else:
                y_pred = model.predict(X_test)
                y_actual = y_test
            
            # Plot actual vs predicted
            ax.scatter(y_actual, y_pred, alpha=0.6, s=20)
            ax.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{model_name.title()} - Actual vs Predicted')
            ax.grid(True, alpha=0.3)
            
            # Add R² score
            r2 = r2_score(y_actual, y_pred)
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/model_predictions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance plot (for tree-based models)
        tree_models = ['random_forest', 'xgboost']
        for model_name in tree_models:
            if model_name in self.results and 'feature_importance' in self.results[model_name]:
                plt.figure(figsize=(12, 8))
                importance_df = self.results[model_name]['feature_importance'].head(15)
                plt.barh(range(len(importance_df)), importance_df['importance'])
                plt.yticks(range(len(importance_df)), importance_df['feature'])
                plt.xlabel('Feature Importance')
                plt.title(f'{model_name.title()} - Feature Importance')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{plots_dir}/{model_name}_feature_importance.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"Prediction visualizations saved to {plots_dir}/")
    
    def run_complete_analysis(self, target_column: str = 'vix', lookback_days: int = 5, output_dir: str = 'output'):
        """Run complete predictive modeling analysis"""
        print("Running complete predictive modeling analysis...")
        
        # Prepare data
        data = self.prepare_data(target_column, lookback_days)
        if data is None:
            return None
        
        X_train, X_test, y_train, y_test, feature_cols = data
        
        # Train all models
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        # Baseline model
        self.create_baseline_model(X_train, X_test, y_train, y_test)
        
        # Machine learning models
        self.train_random_forest(X_train, X_test, y_train, y_test, feature_cols)
        self.train_xgboost(X_train, X_test, y_train, y_test, feature_cols)
        
        # Deep learning model
        self.train_lstm(X_train, X_test, y_train, y_test, feature_cols, lookback_days)
        
        # Compare models
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        comparison_df = self.compare_models()
        
        # Create visualizations
        self.create_prediction_visualizations(output_dir, X_test, y_test)
        
        # Save results
        results_summary = {
            'model_comparison': comparison_df.to_dict('records'),
            'best_model': comparison_df.iloc[0]['Model'],
            'best_r2': comparison_df.iloc[0]['Test_R2'],
            'feature_columns': feature_cols,
            'data_info': {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features': len(feature_cols)
            }
        }
        
        return results_summary
