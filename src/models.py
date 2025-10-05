"""
Model training and persistence module for BRI pipeline.
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from utils import ensure_directory

class ModelTrainer:
    """Model training and persistence class for BRI pipeline."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Model parameters
        self.random_state = self.config.get('random_state', 42)
        self.test_size = self.config.get('test_size', 0.2)
        
        # Initialize models
        self.models = {}
        self.trained_models = {}
        
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize various models for training."""
        models = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0, random_state=self.random_state),
            'lasso_regression': Lasso(alpha=0.1, random_state=self.random_state),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=self.random_state
            ),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        self.models = models
        return models
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, 
                    model_names: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Train multiple models and return performance metrics."""
        self.logger.info(f"Training models on {len(X)} samples")
        
        if model_names is None:
            model_names = list(self.models.keys())
        
        results = {}
        
        for name in model_names:
            if name not in self.models:
                self.logger.warning(f"Model {name} not found")
                continue
            
            try:
                model = self.models[name]
                
                # Train model
                model.fit(X, y)
                
                # Make predictions
                y_pred = model.predict(X)
                
                # Calculate metrics
                mse = mean_squared_error(y, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                # Store trained model
                self.trained_models[name] = model
                
                # Store results
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'predictions': y_pred
                }
                
                self.logger.info(f"Trained {name}: RMSE={rmse:.4f}, R²={r2:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def cross_validate_models(self, X: pd.DataFrame, y: pd.Series, 
                            cv_folds: int = 5) -> Dict[str, Dict]:
        """Perform cross-validation on models."""
        from sklearn.model_selection import cross_val_score
        
        self.logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        cv_results = {}
        
        for name, model in self.models.items():
            try:
                # Perform cross-validation
                cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
                
                cv_results[name] = {
                    'mean_cv_score': -cv_scores.mean(),
                    'std_cv_score': cv_scores.std(),
                    'cv_scores': -cv_scores
                }
                
                self.logger.info(f"{name} CV RMSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                self.logger.error(f"Error in cross-validation for {name}: {e}")
                cv_results[name] = {'error': str(e)}
        
        return cv_results
    
    def select_best_model(self, results: Dict[str, Dict], metric: str = 'rmse') -> str:
        """Select the best model based on specified metric."""
        best_model = None
        best_score = float('inf')
        
        for name, result in results.items():
            if 'error' in result:
                continue
            
            score = result.get(metric, float('inf'))
            if score < best_score:
                best_score = score
                best_model = name
        
        if best_model:
            self.logger.info(f"Best model: {best_model} ({metric}={best_score:.4f})")
        
        return best_model
    
    def predict(self, X: pd.DataFrame, model_name: str) -> np.ndarray:
        """Make predictions using a trained model."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")
        
        return self.trained_models[model_name].predict(X)
    
    def get_feature_importance(self, model_name: str, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance for tree-based models."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            self.logger.warning(f"Model {model_name} does not support feature importance")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_models(self, output_dir: str = "models"):
        """Save trained models to disk."""
        ensure_directory(output_dir)
        
        for name, model in self.trained_models.items():
            model_path = os.path.join(output_dir, f"{name}.pkl")
            joblib.dump(model, model_path)
            self.logger.info(f"Saved model {name} to {model_path}")
    
    def load_models(self, model_dir: str = "models"):
        """Load trained models from disk."""
        for name in self.models.keys():
            model_path = os.path.join(model_dir, f"{name}.pkl")
            if os.path.exists(model_path):
                self.trained_models[name] = joblib.load(model_path)
                self.logger.info(f"Loaded model {name} from {model_path}")
    
    def create_ensemble_model(self, X: pd.DataFrame, y: pd.Series, 
                            model_names: List[str]) -> Dict:
        """Create an ensemble model from multiple trained models."""
        self.logger.info(f"Creating ensemble from {model_names}")
        
        # Train individual models
        individual_results = self.train_models(X, y, model_names)
        
        # Get predictions from each model
        predictions = {}
        for name in model_names:
            if name in individual_results and 'error' not in individual_results[name]:
                predictions[name] = individual_results[name]['predictions']
        
        if not predictions:
            self.logger.error("No successful models for ensemble")
            return {}
        
        # Create ensemble predictions (simple average)
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        # Calculate ensemble metrics
        mse = mean_squared_error(y, ensemble_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, ensemble_pred)
        r2 = r2_score(y, ensemble_pred)
        
        return {
            'ensemble_predictions': ensemble_pred,
            'individual_predictions': predictions,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, 
                            model_name: str, param_grid: Dict) -> Dict:
        """Perform hyperparameter tuning using grid search."""
        from sklearn.model_selection import GridSearchCV
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        self.logger.info(f"Performing hyperparameter tuning for {model_name}")
        
        model = self.models[model_name]
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        # Get best parameters and model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
        # Train best model and get metrics
        y_pred = best_model.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Store best model
        self.trained_models[f"{model_name}_tuned"] = best_model
        
        return {
            'best_params': best_params,
            'best_model': best_model,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def create_model_comparison_report(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """Create a comparison report of all models."""
        comparison_data = []
        
        for name, result in results.items():
            if 'error' in result:
                comparison_data.append({
                    'model': name,
                    'mse': np.nan,
                    'rmse': np.nan,
                    'mae': np.nan,
                    'r2': np.nan,
                    'status': 'error'
                })
            else:
                comparison_data.append({
                    'model': name,
                    'mse': result.get('mse', np.nan),
                    'rmse': result.get('rmse', np.nan),
                    'mae': result.get('mae', np.nan),
                    'r2': result.get('r2', np.nan),
                    'status': 'success'
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('rmse')
        
        return comparison_df
