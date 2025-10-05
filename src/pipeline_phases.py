"""
Refactored BRI Pipeline Phases

This module breaks down the large bri_pipeline.py into smaller, focused functions
organized by pipeline phases. Each phase is a separate module for better
maintainability and testability.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import required libraries
from transformers import pipeline
from textblob import TextBlob
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class Phase1DataCollection:
    """Phase 1: Data Collection - Market, GDELT, and Reddit data"""
    
    def __init__(self, data_collector, gdelt_processor):
        self.data_collector = data_collector
        self.gdelt_processor = gdelt_processor
    
    def collect_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Collect market data from Yahoo Finance"""
        logger.info("Collecting market data from Yahoo Finance...")
        market_data = self.data_collector.collect_market_data(start_date, end_date)
        logger.info(f"Collected market data: {market_data.shape}")
        return market_data
    
    def process_gdelt_data(self, gdelt_file: str) -> pd.DataFrame:
        """Process GDELT export file"""
        logger.info("Processing GDELT export file...")
        gdelt_events = self.gdelt_processor.process_export_file(gdelt_file)
        logger.info(f"Processed GDELT events: {gdelt_events.shape}")
        return gdelt_events
    
    def collect_reddit_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Collect Reddit data"""
        logger.info("Collecting Reddit data...")
        reddit_data = self.data_collector.collect_reddit_data(start_date, end_date)
        logger.info(f"Collected Reddit data: {reddit_data.shape}")
        return reddit_data

class Phase2DataPreprocessing:
    """Phase 2: Data Preprocessing - Cleaning and sentiment analysis"""
    
    def __init__(self, text_preprocessor, config=None):
        self.text_preprocessor = text_preprocessor
        self.config = config or self._load_default_config()
    
    def _load_default_config(self):
        """Load default configuration"""
        return {
            'text_processing': {
                'min_text_length': 10,
                'max_text_length': 512,
                'engagement_weight': 2
            },
            'normalization': {
                'goldstein_scale_min': -10,
                'goldstein_scale_max': 10,
                'goldstein_norm_min': 0,
                'goldstein_norm_max': 1,
                'sentiment_normalization_factor': 100
            }
        }
    
    def clean_gdelt_data(self, gdelt_events: pd.DataFrame) -> pd.DataFrame:
        """Clean GDELT data as specified"""
        if gdelt_events.empty:
            logger.warning("No GDELT events to clean")
            return pd.DataFrame()
        
        # Flexible schema: map optional columns and fallback to zeros when missing
        df = gdelt_events.copy()
        # Ensure required logical columns exist
        if 'date' not in df.columns and 'SQLDATE' in df.columns:
            df['date'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d', errors='coerce')
        if 'GoldsteinScale' not in df.columns and 'goldstein_scale' in df.columns:
            df['GoldsteinScale'] = df['goldstein_scale']
        if 'AvgTone' not in df.columns and 'tone' in df.columns:
            df['AvgTone'] = df['tone']

        # Drop nulls on core fields if present
        core_subset = [c for c in ['date', 'GoldsteinScale', 'AvgTone'] if c in df.columns]
        if core_subset:
            df = df.dropna(subset=core_subset)
        
        # Convert SQLDATE → datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Normalize GoldsteinScale (–10 to +10 → 0–1 scale)
        goldstein_min = self.config['normalization']['goldstein_scale_min']
        goldstein_max = self.config['normalization']['goldstein_scale_max']
        norm_min = self.config['normalization']['goldstein_norm_min']
        norm_max = self.config['normalization']['goldstein_norm_max']
        
        df["GoldsteinNorm"] = (df["GoldsteinScale"] - goldstein_min) / (goldstein_max - goldstein_min)
        df["GoldsteinNorm"] = df["GoldsteinNorm"].clip(norm_min, norm_max)
        
        # Group by day and take average tone
        agg_dict = {"GoldsteinNorm": "mean", "AvgTone": "mean"}
        if 'NumMentions' in df.columns: agg_dict['NumMentions'] = 'sum'
        if 'NumSources' in df.columns: agg_dict['NumSources'] = 'sum'
        if 'NumArticles' in df.columns: agg_dict['NumArticles'] = 'sum'
        if 'GLOBALEVENTID' in df.columns: agg_dict['GLOBALEVENTID'] = 'count'

        daily_gdelt = df.groupby("date").agg(agg_dict).reset_index()

        # Build standardized columns with safe defaults
        daily_gdelt = daily_gdelt.rename(columns={
            'GoldsteinNorm': 'avg_goldstein_tone',
            'AvgTone': 'avg_tone',
            'NumMentions': 'total_mentions',
            'NumSources': 'total_sources',
            'NumArticles': 'total_articles',
            'GLOBALEVENTID': 'event_count'
        })
        for col, default in (
            ('total_mentions', 0), ('total_sources', 0), ('total_articles', 0), ('event_count', 0)
        ):
            if col not in daily_gdelt.columns:
                daily_gdelt[col] = default
        
        logger.info(f"Cleaned GDELT data: {len(daily_gdelt)} days")
        return daily_gdelt
    
    def clean_reddit_text(self, reddit_data: pd.DataFrame) -> pd.DataFrame:
        """Clean Reddit text as specified"""
        if reddit_data.empty:
            logger.warning("No Reddit data to clean")
            return pd.DataFrame()
        
        # Process each post
        cleaned_posts = []
        min_text_length = self.config['text_processing']['min_text_length']
        for idx, row in reddit_data.iterrows():
            text = str(row.get('combined_text', ''))
            if text and len(text) > min_text_length:
                try:
                    # Clean text (remove emojis, URLs, symbols, lowercase, stopwords)
                    cleaned_text = self.text_preprocessor.preprocess_text(text)
                    if cleaned_text:
                        cleaned_posts.append({
                            'date': row['date'],
                            'text': cleaned_text,
                            'subreddit': row.get('subreddit', 'unknown'),
                            'score': row.get('score', 0),
                            'num_comments': row.get('num_comments', 0),
                            'url': row.get('url', '')
                        })
                except Exception as e:
                    logger.warning(f"Error cleaning Reddit text: {e}")
                    continue
        
        df = pd.DataFrame(cleaned_posts)
        logger.info(f"Cleaned Reddit data: {len(df)} posts")
        return df
    
    def perform_sentiment_analysis(self, reddit_clean: pd.DataFrame, 
                                 gdelt_clean: pd.DataFrame) -> pd.DataFrame:
        """Perform sentiment analysis as specified"""
        sentiment_data = []
        
        # Initialize FinBERT sentiment pipeline
        try:
            sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            logger.info("Loaded FinBERT sentiment model")
            use_finbert = True
        except:
            logger.warning("FinBERT not available, using TextBlob")
            sentiment_pipeline = None
            use_finbert = False
        
        # Analyze Reddit sentiment
        if not reddit_clean.empty:
            logger.info("Analyzing Reddit sentiment...")
            sentiment_data.extend(self._analyze_reddit_sentiment(
                reddit_clean, sentiment_pipeline, use_finbert
            ))
        
        # Analyze GDELT sentiment
        if not gdelt_clean.empty:
            logger.info("Analyzing GDELT sentiment...")
            sentiment_data.extend(self._analyze_gdelt_sentiment(gdelt_clean))
        
        df = pd.DataFrame(sentiment_data)
        logger.info(f"Generated sentiment data: {len(df)} records")
        return df
    
    def _analyze_reddit_sentiment(self, reddit_clean: pd.DataFrame, 
                                sentiment_pipeline, use_finbert: bool) -> List[Dict]:
        """Analyze Reddit sentiment"""
        sentiment_data = []
        
        for idx, row in reddit_clean.iterrows():
            text = row['text']
            try:
                if use_finbert:
                    # Use FinBERT
                    max_text_length = self.config['text_processing']['max_text_length']
                    result = sentiment_pipeline(text[:max_text_length])  # Limit text length
                    sentiment_score = result[0]['score'] if result[0]['label'] == 'POSITIVE' else -result[0]['score']
                    confidence = result[0]['score']
                else:
                    # Use TextBlob
                    blob = TextBlob(text)
                    sentiment_score = blob.sentiment.polarity
                    confidence = abs(blob.sentiment.polarity)
                
                sentiment_data.append({
                    'date': row['date'],
                    'sentiment': sentiment_score,
                    'confidence': confidence,
                    'source': 'reddit',
                    'subreddit': row['subreddit'],
                    'engagement': row['score'] + row['num_comments'] * self.config['text_processing']['engagement_weight']
                })
            except Exception as e:
                logger.warning(f"Error analyzing Reddit sentiment: {e}")
                continue
        
        return sentiment_data
    
    def _analyze_gdelt_sentiment(self, gdelt_clean: pd.DataFrame) -> List[Dict]:
        """Analyze GDELT sentiment"""
        sentiment_data = []
        
        for idx, row in gdelt_clean.iterrows():
            try:
                # Use AvgTone as sentiment score
                sentiment_normalization_factor = self.config['normalization']['sentiment_normalization_factor']
                sentiment_score = row['avg_tone'] / sentiment_normalization_factor  # Normalize to -1 to 1
                confidence = abs(sentiment_score)
                
                sentiment_data.append({
                    'date': row['date'],
                    'sentiment': sentiment_score,
                    'confidence': confidence,
                    'source': 'gdelt',
                    'subreddit': 'news',
                    'engagement': row['total_mentions']
                })
            except Exception as e:
                logger.warning(f"Error analyzing GDELT sentiment: {e}")
                continue
        
        return sentiment_data

class Phase3FeatureEngineering:
    """Phase 3: Feature Engineering - Create behavioral risk indicators"""
    
    def __init__(self, config=None):
        self.config = config or self._load_default_config()
    
    def _load_default_config(self):
        """Load default configuration"""
        return {
            'rolling_windows': {
                'media_herding': 7,
                'polarity_skew': 7,
                'min_periods': 1
            },
            'text_processing': {
                'engagement_weight': 2
            }
        }
    
    def create_behavioral_features(self, sentiment_data: pd.DataFrame, 
                                 gdelt_clean: pd.DataFrame, 
                                 reddit_clean: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral risk indicators as specified"""
        logger.info("Creating behavioral risk indicators...")
        
        # Normalize all date columns to pandas datetime (midnight) to avoid dtype mismatches
        for df_ref in (sentiment_data, gdelt_clean, reddit_clean):
            if df_ref is not None and not df_ref.empty and 'date' in df_ref.columns:
                df_ref['date'] = pd.to_datetime(df_ref['date']).dt.normalize()
        
        # Group sentiment by date; if empty, build a neutral daily frame from GDELT dates
        if sentiment_data is None or sentiment_data.empty or 'date' not in sentiment_data.columns:
            if gdelt_clean is not None and not gdelt_clean.empty and 'date' in gdelt_clean.columns:
                dates = (
                    pd.to_datetime(gdelt_clean['date']).dt.normalize().dropna().sort_values().unique()
                )
                daily_sentiment = pd.DataFrame({
                    'date': dates,
                    'sentiment_mean': 0.0,
                    'sentiment_std': 0.0,
                    'sentiment_count': 0,
                    'confidence_mean': 0.0,
                    'engagement_sum': 0.0
                })
            else:
                # No basis for dates; return empty
                logger.warning("No sentiment or GDELT dates available to build features")
                return pd.DataFrame()
        else:
            daily_sentiment = sentiment_data.groupby('date').agg({
                'sentiment': ['mean', 'std', 'count'],
                'confidence': 'mean',
                'engagement': 'sum'
            }).reset_index()
        
        # Flatten column names
        if isinstance(daily_sentiment.columns, pd.MultiIndex):
            daily_sentiment.columns = ['date', 'sentiment_mean', 'sentiment_std', 
                                     'sentiment_count', 'confidence_mean', 'engagement_sum']
        
        # Calculate behavioral features
        features = self._calculate_behavioral_features(daily_sentiment, gdelt_clean, reddit_clean)
        
        logger.info(f"Created behavioral features: {features.shape}")
        return features
    
    def _calculate_behavioral_features(self, daily_sentiment: pd.DataFrame,
                                     gdelt_clean: pd.DataFrame,
                                     reddit_clean: pd.DataFrame) -> pd.DataFrame:
        """Calculate specific behavioral features"""
        features = daily_sentiment.copy()
        
        # 1. Sentiment Volatility (std of sentiment)
        features['sentiment_volatility'] = features['sentiment_std'].fillna(0)
        
        # 2. News Tone (from GDELT)
        if not gdelt_clean.empty:
            features = features.merge(
                gdelt_clean[['date', 'avg_tone']], 
                on='date', 
                how='left'
            )
            features['news_tone'] = features['avg_tone'].fillna(0)
        else:
            features['news_tone'] = 0
        
        # 3. Media Herding (correlation between sources)
        features['media_herding'] = self._calculate_media_herding(features)
        
        # 4. Polarity Skew (skewness of sentiment distribution)
        features['polarity_skew'] = self._calculate_polarity_skew(features)
        
        # 5. Event Density (number of events per day)
        features['event_density'] = self._calculate_event_density(features, gdelt_clean)
        
        # 6. Engagement Index (weighted by user engagement)
        features['engagement_index'] = self._calculate_engagement_index(features, reddit_clean)
        
        # 7. Fear Index (negative sentiment + high volatility)
        features['fear_index'] = self._calculate_fear_index(features)
        
        # 8. Overconfidence Index (positive sentiment + low volatility)
        features['overconfidence_index'] = self._calculate_overconfidence_index(features)
        
        return features
    
    def _calculate_media_herding(self, features: pd.DataFrame) -> pd.Series:
        """Calculate media herding indicator"""
        # Simple implementation - can be enhanced
        window = self.config['rolling_windows']['media_herding']
        min_periods = self.config['rolling_windows']['min_periods']
        return features['sentiment_std'].rolling(window=window, min_periods=min_periods).std().fillna(0)
    
    def _calculate_polarity_skew(self, features: pd.DataFrame) -> pd.Series:
        """Calculate polarity skewness"""
        # Simple implementation - can be enhanced
        window = self.config['rolling_windows']['polarity_skew']
        min_periods = self.config['rolling_windows']['min_periods']
        return features['sentiment_mean'].rolling(window=window, min_periods=min_periods).skew().fillna(0)
    
    def _calculate_event_density(self, features: pd.DataFrame, gdelt_clean: pd.DataFrame) -> pd.Series:
        """Calculate event density"""
        if gdelt_clean.empty:
            return pd.Series(0, index=features.index)
        
        event_counts = (
            gdelt_clean.groupby('date', as_index=False)['event_count'].sum()
        )
        features = features.merge(event_counts, on='date', how='left')
        return features['event_count'].fillna(0)
    
    def _calculate_engagement_index(self, features: pd.DataFrame, reddit_clean: pd.DataFrame) -> pd.Series:
        """Calculate engagement index"""
        if reddit_clean.empty:
            return pd.Series(0, index=features.index)
        
        daily_engagement = reddit_clean.groupby('date').agg({
            'score': 'sum',
            'num_comments': 'sum'
        }).reset_index()
        
        engagement_weight = self.config['text_processing']['engagement_weight']
        daily_engagement['engagement'] = daily_engagement['score'] + daily_engagement['num_comments'] * engagement_weight
        
        features = features.merge(daily_engagement[['date', 'engagement']], on='date', how='left')
        return features['engagement'].fillna(0)
    
    def _calculate_fear_index(self, features: pd.DataFrame) -> pd.Series:
        """Calculate fear index"""
        # Negative sentiment + high volatility
        fear = -features['sentiment_mean'] + features['sentiment_volatility']
        return fear.fillna(0)
    
    def _calculate_overconfidence_index(self, features: pd.DataFrame) -> pd.Series:
        """Calculate overconfidence index"""
        # Positive sentiment + low volatility
        overconfidence = features['sentiment_mean'] - features['sentiment_volatility']
        return overconfidence.fillna(0)

class Phase4BRICalculation:
    """Phase 4: BRI Calculation - Normalize and aggregate features"""
    
    def __init__(self, config=None):
        self.config = config or self._load_default_config()
    
    def _load_default_config(self):
        """Load default configuration"""
        return {
            'feature_weights': {
                'sentiment_volatility': 0.2,
                'news_tone': 0.15,
                'media_herding': 0.15,
                'polarity_skew': 0.1,
                'event_density': 0.1,
                'engagement_index': 0.1,
                'fear_index': 0.1,
                'overconfidence_index': 0.1
            },
            'normalization': {
                'bri_clip_min': -3,
                'bri_clip_max': 3,
                'bri_scale_min': 0,
                'bri_scale_max': 100
            }
        }
    
    def calculate_bri(self, behavioral_features: pd.DataFrame) -> pd.DataFrame:
        """Calculate the Behavioral Risk Index"""
        logger.info("Calculating Behavioral Risk Index...")
        
        # Select features for BRI calculation
        feature_columns = [
            'sentiment_volatility', 'news_tone', 'media_herding', 
            'polarity_skew', 'event_density', 'engagement_index',
            'fear_index', 'overconfidence_index'
        ]
        
        # Filter available features
        available_features = [col for col in feature_columns if col in behavioral_features.columns]
        
        if not available_features:
            logger.error("No features available for BRI calculation")
            return pd.DataFrame()
        
        # Normalize features to 0-1 scale
        normalized_features = self._normalize_features(behavioral_features[available_features])
        
        # Calculate weighted BRI
        weights = self._get_feature_weights(available_features)
        bri_scores = self._calculate_weighted_bri(normalized_features, weights)
        
        # Create final BRI dataframe
        bri_data = behavioral_features[['date']].copy()
        bri_data['bri'] = bri_scores
        bri_data['bri_normalized'] = self._normalize_bri_to_0_100(bri_scores)
        
        logger.info(f"Calculated BRI: {bri_data.shape}")
        return bri_data
    
    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize features to 0-1 scale"""
        scaler = MinMaxScaler()
        normalized = pd.DataFrame(
            scaler.fit_transform(features.fillna(0)),
            columns=features.columns,
            index=features.index
        )
        return normalized
    
    def _get_feature_weights(self, features: List[str]) -> Dict[str, float]:
        """Get feature weights for BRI calculation"""
        # Get weights from configuration
        default_weights = self.config['feature_weights']
        
        # Return weights for available features
        return {feature: default_weights.get(feature, 0.1) for feature in features}
    
    def _calculate_weighted_bri(self, normalized_features: pd.DataFrame, 
                              weights: Dict[str, float]) -> pd.Series:
        """Calculate weighted BRI score"""
        bri_scores = pd.Series(0, index=normalized_features.index)
        
        for feature, weight in weights.items():
            if feature in normalized_features.columns:
                bri_scores += normalized_features[feature] * weight
        
        return bri_scores
    
    def _normalize_bri_to_0_100(self, bri_scores: pd.Series) -> pd.Series:
        """Normalize BRI scores to 0-100 scale"""
        # Get normalization parameters from config
        clip_min = self.config['normalization']['bri_clip_min']
        clip_max = self.config['normalization']['bri_clip_max']
        scale_min = self.config['normalization']['bri_scale_min']
        scale_max = self.config['normalization']['bri_scale_max']
        
        # Clip extreme values
        clipped_scores = bri_scores.clip(clip_min, clip_max)
        
        # Normalize to 0-100
        normalized = (clipped_scores - clip_min) / (clip_max - clip_min) * (scale_max - scale_min) + scale_min
        return normalized.clip(scale_min, scale_max)

class Phase5AnalysisValidation:
    """Phase 5: Analysis & Validation - Compare with VIX and backtest"""
    
    def __init__(self, data_collector):
        self.data_collector = data_collector
    
    def collect_vix_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Collect VIX data for comparison"""
        logger.info("Collecting VIX data...")
        vix_data = self.data_collector.collect_vix_data(start_date, end_date)
        logger.info(f"Collected VIX data: {vix_data.shape}")
        return vix_data
    
    def run_validation_analysis(self, bri_data: pd.DataFrame, 
                              vix_data: pd.DataFrame, 
                              market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run validation analysis comparing BRI with VIX"""
        logger.info("Running validation analysis...")
        
        # Merge data
        merged_data = self._merge_validation_data(bri_data, vix_data, market_data)
        
        if merged_data.empty:
            logger.warning("No data available for validation")
            return {}
        
        # Calculate correlations
        correlations = self._calculate_correlations(merged_data)
        
        # Calculate lag analysis
        lag_analysis = self._calculate_lag_analysis(merged_data)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(merged_data)
        
        validation_results = {
            'correlations': correlations,
            'lag_analysis': lag_analysis,
            'performance_metrics': performance_metrics,
            'data_points': len(merged_data)
        }
        
        logger.info("Validation analysis completed")
        return validation_results
    
    def run_economic_backtesting(self, bri_data: pd.DataFrame, 
                               market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run economic event backtesting"""
        logger.info("Running economic event backtesting...")
        
        # Define economic events (simplified)
        economic_events = self._define_economic_events()
        
        # Run backtesting
        backtest_results = self._run_backtest(bri_data, market_data, economic_events)
        
        logger.info("Economic backtesting completed")
        return backtest_results
    
    def _merge_validation_data(self, bri_data: pd.DataFrame, 
                             vix_data: pd.DataFrame, 
                             market_data: pd.DataFrame) -> pd.DataFrame:
        """Merge data for validation"""
        # Start with BRI data and normalize dates to tz-naive datetimes
        merged = bri_data.copy()
        for df_ref in (merged, vix_data, market_data):
            if df_ref is not None and not df_ref.empty and 'date' in df_ref.columns:
                df_ref['date'] = pd.to_datetime(df_ref['date'], utc=True, errors='coerce').dt.tz_convert(None).dt.normalize()
        
        # Add VIX data
        if not vix_data.empty:
            merged = merged.merge(vix_data[['date', 'vix']], on='date', how='left')
        
        # Add market data
        if not market_data.empty:
            market_cols = ['date', 'close', 'volume']
            available_cols = [col for col in market_cols if col in market_data.columns]
            merged = merged.merge(market_data[available_cols], on='date', how='left')
        
        return merged.dropna()
    
    def _calculate_correlations(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlations between BRI and other indicators"""
        correlations = {}
        
        if 'vix' in data.columns:
            correlations['bri_vix'] = data['bri'].corr(data['vix'])
        
        if 'close' in data.columns:
            correlations['bri_market'] = data['bri'].corr(data['close'])
        
        return correlations
    
    def _calculate_lag_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate lag analysis between BRI and VIX"""
        if 'vix' not in data.columns:
            return {}
        
        # Calculate correlations at different lags
        lags = range(-5, 6)  # -5 to +5 days
        lag_correlations = {}
        
        for lag in lags:
            if lag < 0:
                # BRI leads VIX
                bri_shifted = data['bri'].shift(lag)
                vix_original = data['vix']
            else:
                # VIX leads BRI
                bri_original = data['bri']
                vix_shifted = data['vix'].shift(lag)
            
            if lag < 0:
                corr = bri_shifted.corr(vix_original)
            else:
                corr = bri_original.corr(vix_shifted)
            
            lag_correlations[lag] = corr
        
        # Find best lag
        best_lag = max(lag_correlations, key=lag_correlations.get)
        best_correlation = lag_correlations[best_lag]
        
        return {
            'lag_correlations': lag_correlations,
            'best_lag': best_lag,
            'best_correlation': best_correlation
        }
    
    def _calculate_performance_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics"""
        metrics = {}
        
        if 'bri' in data.columns:
            metrics['bri_mean'] = data['bri'].mean()
            metrics['bri_std'] = data['bri'].std()
            metrics['bri_min'] = data['bri'].min()
            metrics['bri_max'] = data['bri'].max()
        
        if 'vix' in data.columns:
            metrics['vix_mean'] = data['vix'].mean()
            metrics['vix_std'] = data['vix'].std()
        
        return metrics
    
    def _define_economic_events(self) -> List[Dict[str, Any]]:
        """Define economic events for backtesting"""
        # Simplified economic events - can be expanded
        events = [
            {'date': '2020-03-15', 'event': 'COVID-19 Market Crash', 'type': 'crisis'},
            {'date': '2020-03-23', 'event': 'Market Recovery Start', 'type': 'recovery'},
            {'date': '2021-01-06', 'event': 'Capitol Riots', 'type': 'crisis'},
            {'date': '2022-02-24', 'event': 'Russia-Ukraine War', 'type': 'crisis'},
            {'date': '2022-03-16', 'event': 'Fed Rate Hike', 'type': 'policy'}
        ]
        
        return events
    
    def _run_backtest(self, bri_data: pd.DataFrame, 
                     market_data: pd.DataFrame, 
                     economic_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run backtest on economic events"""
        backtest_results = {
            'events_analyzed': len(economic_events),
            'event_results': []
        }
        
        for event in economic_events:
            event_date = pd.to_datetime(event['date'])
            
            # Find BRI value around event
            event_window = bri_data[
                (bri_data['date'] >= event_date - timedelta(days=5)) &
                (bri_data['date'] <= event_date + timedelta(days=5))
            ]
            
            if not event_window.empty:
                event_result = {
                    'event': event['event'],
                    'date': event['date'],
                    'type': event['type'],
                    'bri_before': event_window['bri'].iloc[0] if len(event_window) > 0 else None,
                    'bri_after': event_window['bri'].iloc[-1] if len(event_window) > 0 else None,
                    'bri_change': None
                }
                
                if event_result['bri_before'] is not None and event_result['bri_after'] is not None:
                    event_result['bri_change'] = event_result['bri_after'] - event_result['bri_before']
                
                backtest_results['event_results'].append(event_result)
        
        return backtest_results

class Phase6Visualization:
    """Phase 6: Visualization Dashboard - Create charts and visualizations"""
    
    def create_visualizations(self, bri_data: pd.DataFrame, 
                            vix_data: pd.DataFrame, 
                            market_data: pd.DataFrame, 
                            output_dir: str) -> None:
        """Create visualizations for the dashboard"""
        logger.info("Creating visualizations...")
        
        # Create output directory for charts
        charts_dir = os.path.join(output_dir, 'charts')
        os.makedirs(charts_dir, exist_ok=True)
        
        # Create BRI time series chart
        self._create_bri_timeseries_chart(bri_data, charts_dir)
        
        # Create correlation chart
        if not vix_data.empty:
            self._create_correlation_chart(bri_data, vix_data, charts_dir)
        
        # Create market comparison chart
        if not market_data.empty:
            self._create_market_comparison_chart(bri_data, market_data, charts_dir)
        
        logger.info(f"Visualizations saved to: {charts_dir}")
    
    def _create_bri_timeseries_chart(self, bri_data: pd.DataFrame, charts_dir: str) -> None:
        """Create BRI time series chart"""
        plt.figure(figsize=(12, 6))
        plt.plot(bri_data['date'], bri_data['bri'], linewidth=2, color='blue')
        plt.title('Behavioral Risk Index (BRI) Over Time')
        plt.xlabel('Date')
        plt.ylabel('BRI (0-100)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'bri_timeseries.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_correlation_chart(self, bri_data: pd.DataFrame, 
                                vix_data: pd.DataFrame, charts_dir: str) -> None:
        """Create BRI vs VIX correlation chart"""
        # Merge data
        merged = bri_data.merge(vix_data[['date', 'vix']], on='date', how='inner')
        
        if merged.empty:
            return
        
        plt.figure(figsize=(10, 6))
        plt.scatter(merged['bri'], merged['vix'], alpha=0.6)
        plt.xlabel('Behavioral Risk Index (BRI)')
        plt.ylabel('VIX (Volatility Index)')
        plt.title('BRI vs VIX Correlation')
        
        # Add correlation coefficient
        corr = merged['bri'].corr(merged['vix'])
        plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=plt.gca().transAxes, fontsize=12)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'bri_vix_correlation.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_market_comparison_chart(self, bri_data: pd.DataFrame, 
                                      market_data: pd.DataFrame, charts_dir: str) -> None:
        """Create market comparison chart"""
        # Merge data
        merged = bri_data.merge(market_data[['date', 'close']], on='date', how='inner')
        
        if merged.empty:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # BRI chart
        ax1.plot(merged['date'], merged['bri'], linewidth=2, color='blue')
        ax1.set_title('Behavioral Risk Index (BRI)')
        ax1.set_ylabel('BRI (0-100)')
        ax1.grid(True, alpha=0.3)
        
        # Market chart
        ax2.plot(merged['date'], merged['close'], linewidth=2, color='green')
        ax2.set_title('Market Close Price')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'market_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

class Phase7FinalDeliverables:
    """Phase 7: Final Deliverables - Save results and generate reports"""
    
    def save_all_deliverables(self, output_dir: str, deliverables: Dict[str, Any]) -> None:
        """Save all pipeline deliverables"""
        logger.info("Saving all deliverables...")
        
        # Save BRI data
        if 'bri_timeseries' in deliverables:
            deliverables['bri_timeseries'].to_csv(
                os.path.join(output_dir, 'bri_timeseries.csv'), index=False
            )
        
        # Save validation results
        if 'validation_results' in deliverables:
            with open(os.path.join(output_dir, 'validation_results.json'), 'w') as f:
                json.dump(deliverables['validation_results'], f, indent=2, default=str)
        
        # Save backtest results
        if 'backtest_results' in deliverables:
            with open(os.path.join(output_dir, 'backtest_results.json'), 'w') as f:
                json.dump(deliverables['backtest_results'], f, indent=2, default=str)
        
        logger.info(f"All deliverables saved to: {output_dir}")
    
    def generate_final_report(self, output_dir: str, 
                            validation_results: Dict[str, Any], 
                            backtest_results: Dict[str, Any]) -> None:
        """Generate final report"""
        logger.info("Generating final report...")
        
        report_path = os.path.join(output_dir, 'final_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Behavioral Risk Index (BRI) Final Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Validation results
            if validation_results:
                f.write("## Validation Results\n\n")
                f.write(f"- Data Points: {validation_results.get('data_points', 'N/A')}\n")
                
                correlations = validation_results.get('correlations', {})
                for key, value in correlations.items():
                    f.write(f"- {key}: {value:.3f}\n")
                
                f.write("\n")
            
            # Backtest results
            if backtest_results:
                f.write("## Backtest Results\n\n")
                f.write(f"- Events Analyzed: {backtest_results.get('events_analyzed', 'N/A')}\n")
                
                event_results = backtest_results.get('event_results', [])
                for event in event_results:
                    f.write(f"- {event['event']}: {event.get('bri_change', 'N/A')}\n")
                
                f.write("\n")
        
        logger.info(f"Final report saved to: {report_path}")

# Import os and json for the final deliverables
import os
import json
