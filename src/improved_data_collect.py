"""
Improved Data Collection Module with Enhanced Error Handling and Data Validation

This module provides robust data collection capabilities with:
- Comprehensive error handling and retry logic
- Data quality validation at ingestion
- Rate limiting and backoff strategies
- Detailed logging and monitoring
- Graceful degradation when services are unavailable
"""

import logging
import time
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import yaml
import json
from dataclasses import dataclass
from enum import Enum
import backoff
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Enumeration of supported data sources"""
    GDELT = "gdelt"
    REDDIT = "reddit"
    YAHOO_FINANCE = "yahoo_finance"
    NEWS = "news"

class DataQualityLevel(Enum):
    """Data quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class DataQualityMetrics:
    """Data quality metrics for validation"""
    completeness: float  # Percentage of expected data present
    accuracy: float      # Percentage of data that passes validation
    consistency: float   # Percentage of data that follows expected patterns
    timeliness: float    # How recent the data is
    overall_quality: DataQualityLevel
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'completeness': self.completeness,
            'accuracy': self.accuracy,
            'consistency': self.consistency,
            'timeliness': self.timeliness,
            'overall_quality': self.overall_quality.value
        }

@dataclass
class CollectionResult:
    """Result of data collection operation"""
    success: bool
    data: Optional[pd.DataFrame]
    quality_metrics: Optional[DataQualityMetrics]
    error_message: Optional[str]
    source: DataSource
    timestamp: datetime
    records_collected: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'data_shape': self.data.shape if self.data is not None else None,
            'quality_metrics': self.quality_metrics.to_dict() if self.quality_metrics else None,
            'error_message': self.error_message,
            'source': self.source.value,
            'timestamp': self.timestamp.isoformat(),
            'records_collected': self.records_collected
        }

class DataCollectionError(Exception):
    """Custom exception for data collection errors"""
    pass

class RateLimitError(DataCollectionError):
    """Exception raised when rate limit is exceeded"""
    pass

class DataQualityError(DataCollectionError):
    """Exception raised when data quality is insufficient"""
    pass

class ImprovedDataCollector:
    """
    Enhanced data collector with robust error handling and validation
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the data collector with configuration"""
        self.config = self._load_config(config_path)
        self.session = self._create_session()
        self.quality_thresholds = self._get_quality_thresholds()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise DataCollectionError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise DataCollectionError(f"Error parsing configuration file: {e}")
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy"""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _get_quality_thresholds(self) -> Dict[str, float]:
        """Get data quality thresholds from configuration"""
        return {
            'completeness_min': 0.8,  # 80% of expected data
            'accuracy_min': 0.9,      # 90% of data passes validation
            'consistency_min': 0.85,  # 85% follows expected patterns
            'timeliness_hours': 24    # Data should be within 24 hours
        }
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, RateLimitError),
        max_tries=3,
        max_time=300
    )
    def collect_gdelt_data(self, start_date: str, end_date: str) -> CollectionResult:
        """
        Collect GDELT data with enhanced error handling and validation
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            CollectionResult with data and quality metrics
        """
        try:
            logger.info(f"Starting GDELT data collection from {start_date} to {end_date}")
            
            # Validate date format
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            if start_dt >= end_dt:
                raise DataCollectionError("Start date must be before end date")
            
            # Collect data with retry logic
            data = self._fetch_gdelt_data(start_date, end_date)
            
            # Validate data quality
            quality_metrics = self._assess_data_quality(data, DataSource.GDELT)
            
            # Check if quality meets minimum thresholds
            if quality_metrics.overall_quality == DataQualityLevel.CRITICAL:
                raise DataQualityError(f"Data quality is critical: {quality_metrics}")
            
            logger.info(f"GDELT data collection successful: {len(data)} records")
            
            return CollectionResult(
                success=True,
                data=data,
                quality_metrics=quality_metrics,
                error_message=None,
                source=DataSource.GDELT,
                timestamp=datetime.now(),
                records_collected=len(data)
            )
            
        except Exception as e:
            logger.error(f"GDELT data collection failed: {str(e)}")
            return CollectionResult(
                success=False,
                data=None,
                quality_metrics=None,
                error_message=str(e),
                source=DataSource.GDELT,
                timestamp=datetime.now(),
                records_collected=0
            )
    
    def _fetch_gdelt_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch GDELT data with proper error handling"""
        try:
            # Construct GDELT API URL
            base_url = self.config['data_sources']['gdelt']['export_url']
            params = {
                'query': 'domain:finance OR domain:business OR domain:economy',
                'startdate': start_date.replace('-', ''),
                'enddate': end_date.replace('-', ''),
                'format': 'json'
            }
            
            # Make request with timeout and retry logic
            response = self.session.get(
                base_url,
                params=params,
                timeout=self.config['data_sources']['gdelt']['timeout']
            )
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            if not data or 'results' not in data:
                raise DataCollectionError("No data returned from GDELT API")
            
            # Convert to DataFrame
            df = pd.DataFrame(data['results'])
            
            # Validate required columns
            required_columns = ['date', 'title', 'source', 'url']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise DataCollectionError(f"Missing required columns: {missing_columns}")
            
            return df
            
        except requests.exceptions.Timeout:
            raise DataCollectionError("GDELT API request timed out")
        except requests.exceptions.ConnectionError:
            raise DataCollectionError("Failed to connect to GDELT API")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                raise RateLimitError("GDELT API rate limit exceeded")
            raise DataCollectionError(f"GDELT API HTTP error: {e}")
        except json.JSONDecodeError:
            raise DataCollectionError("Invalid JSON response from GDELT API")
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, RateLimitError),
        max_tries=3,
        max_time=300
    )
    def collect_reddit_data(self, subreddits: List[str], limit: int = 100) -> CollectionResult:
        """
        Collect Reddit data with enhanced error handling and validation
        
        Args:
            subreddits: List of subreddit names to collect from
            limit: Maximum number of posts per subreddit
            
        Returns:
            CollectionResult with data and quality metrics
        """
        try:
            logger.info(f"Starting Reddit data collection from {len(subreddits)} subreddits")
            
            # Check if Reddit is enabled in config
            if not self.config['data_sources']['reddit']['enabled']:
                raise DataCollectionError("Reddit data collection is disabled in configuration")
            
            # Collect data from each subreddit
            all_data = []
            for subreddit in subreddits:
                try:
                    subreddit_data = self._fetch_reddit_data(subreddit, limit)
                    all_data.extend(subreddit_data)
                    
                    # Rate limiting delay
                    time.sleep(self.config['data_sources']['reddit']['rate_limit_delay'])
                    
                except Exception as e:
                    logger.warning(f"Failed to collect data from r/{subreddit}: {str(e)}")
                    continue
            
            if not all_data:
                raise DataCollectionError("No data collected from any subreddit")
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            # Validate data quality
            quality_metrics = self._assess_data_quality(df, DataSource.REDDIT)
            
            # Check if quality meets minimum thresholds
            if quality_metrics.overall_quality == DataQualityLevel.CRITICAL:
                raise DataQualityError(f"Data quality is critical: {quality_metrics}")
            
            logger.info(f"Reddit data collection successful: {len(df)} records")
            
            return CollectionResult(
                success=True,
                data=df,
                quality_metrics=quality_metrics,
                error_message=None,
                source=DataSource.REDDIT,
                timestamp=datetime.now(),
                records_collected=len(df)
            )
            
        except Exception as e:
            logger.error(f"Reddit data collection failed: {str(e)}")
            return CollectionResult(
                success=False,
                data=None,
                quality_metrics=None,
                error_message=str(e),
                source=DataSource.REDDIT,
                timestamp=datetime.now(),
                records_collected=0
            )
    
    def _fetch_reddit_data(self, subreddit: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch Reddit data for a specific subreddit"""
        try:
            # Construct Reddit API URL
            url = f"https://www.reddit.com/r/{subreddit}/hot.json"
            headers = {
                'User-Agent': self.config['data_sources']['reddit'].get('user_agent', 'BRI-DataCollector/1.0')
            }
            
            params = {
                'limit': limit,
                'raw_json': 1
            }
            
            # Make request
            response = self.session.get(
                url,
                headers=headers,
                params=params,
                timeout=self.config['data_sources']['reddit']['timeout']
            )
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            if 'data' not in data or 'children' not in data['data']:
                raise DataCollectionError(f"No data returned from r/{subreddit}")
            
            # Extract post data
            posts = []
            for child in data['data']['children']:
                post = child['data']
                posts.append({
                    'title': post.get('title', ''),
                    'selftext': post.get('selftext', ''),
                    'score': post.get('score', 0),
                    'num_comments': post.get('num_comments', 0),
                    'created_utc': post.get('created_utc', 0),
                    'subreddit': subreddit,
                    'url': post.get('url', ''),
                    'author': post.get('author', ''),
                    'is_self': post.get('is_self', False)
                })
            
            return posts
            
        except requests.exceptions.Timeout:
            raise DataCollectionError(f"Reddit API request timed out for r/{subreddit}")
        except requests.exceptions.ConnectionError:
            raise DataCollectionError(f"Failed to connect to Reddit API for r/{subreddit}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                raise RateLimitError(f"Reddit API rate limit exceeded for r/{subreddit}")
            raise DataCollectionError(f"Reddit API HTTP error for r/{subreddit}: {e}")
        except json.JSONDecodeError:
            raise DataCollectionError(f"Invalid JSON response from Reddit API for r/{subreddit}")
    
    def _assess_data_quality(self, data: pd.DataFrame, source: DataSource) -> DataQualityMetrics:
        """
        Assess data quality metrics for collected data
        
        Args:
            data: DataFrame with collected data
            source: Data source type
            
        Returns:
            DataQualityMetrics object with quality assessment
        """
        try:
            if data.empty:
                return DataQualityMetrics(
                    completeness=0.0,
                    accuracy=0.0,
                    consistency=0.0,
                    timeliness=0.0,
                    overall_quality=DataQualityLevel.CRITICAL
                )
            
            # Calculate completeness
            total_cells = data.shape[0] * data.shape[1]
            non_null_cells = data.count().sum()
            completeness = non_null_cells / total_cells if total_cells > 0 else 0.0
            
            # Calculate accuracy (data that passes validation)
            accuracy = self._calculate_accuracy(data, source)
            
            # Calculate consistency (data that follows expected patterns)
            consistency = self._calculate_consistency(data, source)
            
            # Calculate timeliness (how recent the data is)
            timeliness = self._calculate_timeliness(data, source)
            
            # Determine overall quality level
            overall_quality = self._determine_overall_quality(
                completeness, accuracy, consistency, timeliness
            )
            
            return DataQualityMetrics(
                completeness=completeness,
                accuracy=accuracy,
                consistency=consistency,
                timeliness=timeliness,
                overall_quality=overall_quality
            )
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {str(e)}")
            return DataQualityMetrics(
                completeness=0.0,
                accuracy=0.0,
                consistency=0.0,
                timeliness=0.0,
                overall_quality=DataQualityLevel.CRITICAL
            )
    
    def _calculate_accuracy(self, data: pd.DataFrame, source: DataSource) -> float:
        """Calculate data accuracy based on validation rules"""
        try:
            if data.empty:
                return 0.0
            
            # Define validation rules based on source
            validation_rules = {
                DataSource.GDELT: {
                    'required_columns': ['date', 'title', 'source'],
                    'numeric_columns': [],
                    'text_columns': ['title', 'source']
                },
                DataSource.REDDIT: {
                    'required_columns': ['title', 'subreddit'],
                    'numeric_columns': ['score', 'num_comments'],
                    'text_columns': ['title', 'subreddit']
                }
            }
            
            rules = validation_rules.get(source, {})
            if not rules:
                return 1.0  # No validation rules defined
            
            # Check required columns
            required_columns = rules.get('required_columns', [])
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                return 0.0
            
            # Check numeric columns
            numeric_columns = rules.get('numeric_columns', [])
            numeric_valid = 0
            for col in numeric_columns:
                if col in data.columns:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        numeric_valid += 1
            
            # Check text columns
            text_columns = rules.get('text_columns', [])
            text_valid = 0
            for col in text_columns:
                if col in data.columns:
                    if not data[col].isna().all():
                        text_valid += 1
            
            # Calculate overall accuracy
            total_checks = len(numeric_columns) + len(text_columns)
            if total_checks == 0:
                return 1.0
            
            accuracy = (numeric_valid + text_valid) / total_checks
            return accuracy
            
        except Exception as e:
            logger.error(f"Error calculating accuracy: {str(e)}")
            return 0.0
    
    def _calculate_consistency(self, data: pd.DataFrame, source: DataSource) -> float:
        """Calculate data consistency based on expected patterns"""
        try:
            if data.empty:
                return 0.0
            
            # Define consistency rules based on source
            consistency_rules = {
                DataSource.GDELT: {
                    'date_format': '%Y-%m-%d',
                    'min_title_length': 10,
                    'max_title_length': 500
                },
                DataSource.REDDIT: {
                    'min_title_length': 5,
                    'max_title_length': 300,
                    'score_range': (0, 10000),
                    'num_comments_range': (0, 1000)
                }
            }
            
            rules = consistency_rules.get(source, {})
            if not rules:
                return 1.0  # No consistency rules defined
            
            # Check title length consistency
            title_consistency = 1.0
            if 'title' in data.columns:
                title_lengths = data['title'].str.len()
                min_length = rules.get('min_title_length', 0)
                max_length = rules.get('max_title_length', 1000)
                
                valid_titles = ((title_lengths >= min_length) & 
                              (title_lengths <= max_length)).sum()
                total_titles = len(title_lengths)
                title_consistency = valid_titles / total_titles if total_titles > 0 else 0.0
            
            # Check numeric range consistency
            numeric_consistency = 1.0
            if source == DataSource.REDDIT:
                score_range = rules.get('score_range', (0, 10000))
                comments_range = rules.get('num_comments_range', (0, 1000))
                
                score_valid = 0
                comments_valid = 0
                
                if 'score' in data.columns:
                    score_valid = ((data['score'] >= score_range[0]) & 
                                 (data['score'] <= score_range[1])).sum()
                
                if 'num_comments' in data.columns:
                    comments_valid = ((data['num_comments'] >= comments_range[0]) & 
                                    (data['num_comments'] <= comments_range[1])).sum()
                
                total_records = len(data)
                if total_records > 0:
                    score_consistency = score_valid / total_records
                    comments_consistency = comments_valid / total_records
                    numeric_consistency = (score_consistency + comments_consistency) / 2
            
            # Calculate overall consistency
            overall_consistency = (title_consistency + numeric_consistency) / 2
            return overall_consistency
            
        except Exception as e:
            logger.error(f"Error calculating consistency: {str(e)}")
            return 0.0
    
    def _calculate_timeliness(self, data: pd.DataFrame, source: DataSource) -> float:
        """Calculate data timeliness based on how recent the data is"""
        try:
            if data.empty:
                return 0.0
            
            # Get current time
            current_time = datetime.now()
            
            # Define timeliness rules based on source
            timeliness_rules = {
                DataSource.GDELT: {
                    'date_column': 'date',
                    'max_age_hours': 24
                },
                DataSource.REDDIT: {
                    'date_column': 'created_utc',
                    'max_age_hours': 24
                }
            }
            
            rules = timeliness_rules.get(source, {})
            if not rules:
                return 1.0  # No timeliness rules defined
            
            date_column = rules.get('date_column')
            max_age_hours = rules.get('max_age_hours', 24)
            
            if date_column not in data.columns:
                return 0.0
            
            # Calculate age of each record
            if source == DataSource.GDELT:
                # GDELT dates are in YYYY-MM-DD format
                data['parsed_date'] = pd.to_datetime(data[date_column], errors='coerce')
            else:
                # Reddit timestamps are Unix timestamps
                data['parsed_date'] = pd.to_datetime(data[date_column], unit='s', errors='coerce')
            
            # Calculate age in hours
            data['age_hours'] = (current_time - data['parsed_date']).dt.total_seconds() / 3600
            
            # Count records within acceptable age
            recent_records = (data['age_hours'] <= max_age_hours).sum()
            total_records = len(data)
            
            timeliness = recent_records / total_records if total_records > 0 else 0.0
            return timeliness
            
        except Exception as e:
            logger.error(f"Error calculating timeliness: {str(e)}")
            return 0.0
    
    def _determine_overall_quality(self, completeness: float, accuracy: float, 
                                 consistency: float, timeliness: float) -> DataQualityLevel:
        """Determine overall data quality level based on individual metrics"""
        try:
            # Calculate weighted average
            weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights for all metrics
            overall_score = (completeness * weights[0] + 
                           accuracy * weights[1] + 
                           consistency * weights[2] + 
                           timeliness * weights[3])
            
            # Determine quality level based on score
            if overall_score >= 0.9:
                return DataQualityLevel.EXCELLENT
            elif overall_score >= 0.8:
                return DataQualityLevel.GOOD
            elif overall_score >= 0.7:
                return DataQualityLevel.FAIR
            elif overall_score >= 0.5:
                return DataQualityLevel.POOR
            else:
                return DataQualityLevel.CRITICAL
                
        except Exception as e:
            logger.error(f"Error determining overall quality: {str(e)}")
            return DataQualityLevel.CRITICAL
    
    def collect_all_data(self, start_date: str, end_date: str) -> Dict[str, CollectionResult]:
        """
        Collect data from all enabled sources
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary mapping source names to CollectionResult objects
        """
        results = {}
        
        # Collect GDELT data
        if self.config['data_sources']['gdelt']['enabled']:
            try:
                results['gdelt'] = self.collect_gdelt_data(start_date, end_date)
            except Exception as e:
                logger.error(f"GDELT data collection failed: {str(e)}")
                results['gdelt'] = CollectionResult(
                    success=False,
                    data=None,
                    quality_metrics=None,
                    error_message=str(e),
                    source=DataSource.GDELT,
                    timestamp=datetime.now(),
                    records_collected=0
                )
        
        # Collect Reddit data
        if self.config['data_sources']['reddit']['enabled']:
            try:
                subreddits = self.config['data_sources']['reddit']['subreddits']
                results['reddit'] = self.collect_reddit_data(subreddits)
            except Exception as e:
                logger.error(f"Reddit data collection failed: {str(e)}")
                results['reddit'] = CollectionResult(
                    success=False,
                    data=None,
                    quality_metrics=None,
                    error_message=str(e),
                    source=DataSource.REDDIT,
                    timestamp=datetime.now(),
                    records_collected=0
                )
        
        return results
    
    def get_collection_summary(self, results: Dict[str, CollectionResult]) -> Dict[str, Any]:
        """Get a summary of data collection results"""
        summary = {
            'total_sources': len(results),
            'successful_sources': sum(1 for r in results.values() if r.success),
            'total_records': sum(r.records_collected for r in results.values()),
            'quality_summary': {},
            'errors': []
        }
        
        for source, result in results.items():
            if result.success:
                summary['quality_summary'][source] = result.quality_metrics.to_dict()
            else:
                summary['errors'].append({
                    'source': source,
                    'error': result.error_message
                })
        
        return summary

# Example usage
if __name__ == "__main__":
    # Initialize the improved data collector
    collector = ImprovedDataCollector()
    
    # Collect data from all sources
    results = collector.collect_all_data("2024-01-01", "2024-01-02")
    
    # Get collection summary
    summary = collector.get_collection_summary(results)
    print(json.dumps(summary, indent=2))
