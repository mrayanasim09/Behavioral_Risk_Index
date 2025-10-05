"""
Data Quality Validation Module

This module provides comprehensive data validation at ingestion stage with:
- Data completeness checks
- Data accuracy validation
- Data consistency verification
- Data timeliness assessment
- Schema validation
- Outlier detection
- Data quality scoring
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import re
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class DataQualityScore(Enum):
    """Data quality score levels"""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"           # 80-89
    FAIR = "fair"           # 70-79
    POOR = "poor"           # 60-69
    CRITICAL = "critical"   # 0-59

@dataclass
class ValidationResult:
    """Result of a validation check"""
    is_valid: bool
    score: float
    level: ValidationLevel
    message: str
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'score': self.score,
            'level': self.level.value,
            'message': self.message,
            'details': self.details
        }

@dataclass
class DataQualityReport:
    """Comprehensive data quality report"""
    overall_score: float
    quality_level: DataQualityScore
    validation_results: List[ValidationResult]
    recommendations: List[str]
    timestamp: datetime
    data_shape: Tuple[int, int]
    missing_data_percentage: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_score': self.overall_score,
            'quality_level': self.quality_level.value,
            'validation_results': [result.to_dict() for result in self.validation_results],
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat(),
            'data_shape': self.data_shape,
            'missing_data_percentage': self.missing_data_percentage
        }

class DataValidator:
    """
    Comprehensive data validator with multiple validation strategies
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data validator with configuration"""
        self.config = config or self._get_default_config()
        self.validation_rules = self._load_validation_rules()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default validation configuration"""
        return {
            'completeness_threshold': 0.8,  # 80% of data should be present
            'accuracy_threshold': 0.9,      # 90% of data should be accurate
            'consistency_threshold': 0.85,  # 85% of data should be consistent
            'timeliness_hours': 24,         # Data should be within 24 hours
            'outlier_threshold': 0.05,      # 5% of data can be outliers
            'duplicate_threshold': 0.1,     # 10% of data can be duplicates
            'schema_strictness': 'medium'   # 'strict', 'medium', 'loose'
        }
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules for different data types"""
        return {
            'gdelt': {
                'required_columns': ['date', 'title', 'source', 'url'],
                'column_types': {
                    'date': 'datetime64[ns]',
                    'title': 'object',
                    'source': 'object',
                    'url': 'object'
                },
                'text_validation': {
                    'title': {
                        'min_length': 10,
                        'max_length': 500,
                        'pattern': r'^[a-zA-Z0-9\s\.,!?\-\(\)]+$'
                    }
                },
                'numeric_validation': {},
                'date_validation': {
                    'date': {
                        'min_date': '2020-01-01',
                        'max_date': '2030-12-31'
                    }
                }
            },
            'reddit': {
                'required_columns': ['title', 'subreddit', 'score', 'created_utc'],
                'column_types': {
                    'title': 'object',
                    'subreddit': 'object',
                    'score': 'int64',
                    'created_utc': 'float64'
                },
                'text_validation': {
                    'title': {
                        'min_length': 5,
                        'max_length': 300,
                        'pattern': r'^[a-zA-Z0-9\s\.,!?\-\(\)\[\]]+$'
                    },
                    'subreddit': {
                        'min_length': 3,
                        'max_length': 21,
                        'pattern': r'^[a-zA-Z0-9_]+$'
                    }
                },
                'numeric_validation': {
                    'score': {
                        'min_value': -1000,
                        'max_value': 10000
                    },
                    'num_comments': {
                        'min_value': 0,
                        'max_value': 1000
                    }
                },
                'date_validation': {
                    'created_utc': {
                        'min_timestamp': 1262304000,  # 2010-01-01
                        'max_timestamp': 2000000000   # 2033-05-18
                    }
                }
            },
            'yahoo_finance': {
                'required_columns': ['Date', 'Close', 'Volume'],
                'column_types': {
                    'Date': 'datetime64[ns]',
                    'Close': 'float64',
                    'Volume': 'int64'
                },
                'numeric_validation': {
                    'Close': {
                        'min_value': 0.01,
                        'max_value': 10000
                    },
                    'Volume': {
                        'min_value': 0,
                        'max_value': 1000000000
                    }
                },
                'date_validation': {
                    'Date': {
                        'min_date': '1990-01-01',
                        'max_date': '2030-12-31'
                    }
                }
            }
        }
    
    def validate_data(self, data: pd.DataFrame, data_type: str) -> DataQualityReport:
        """
        Perform comprehensive data validation
        
        Args:
            data: DataFrame to validate
            data_type: Type of data ('gdelt', 'reddit', 'yahoo_finance')
            
        Returns:
            DataQualityReport with validation results
        """
        logger.info(f"Starting data validation for {data_type} data")
        
        validation_results = []
        recommendations = []
        
        # Basic data checks
        validation_results.extend(self._validate_basic_structure(data, data_type))
        validation_results.extend(self._validate_completeness(data, data_type))
        validation_results.extend(self._validate_accuracy(data, data_type))
        validation_results.extend(self._validate_consistency(data, data_type))
        validation_results.extend(self._validate_timeliness(data, data_type))
        validation_results.extend(self._validate_schema(data, data_type))
        validation_results.extend(self._validate_outliers(data, data_type))
        validation_results.extend(self._validate_duplicates(data, data_type))
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(validation_results)
        quality_level = self._determine_quality_level(overall_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validation_results, data_type)
        
        # Calculate missing data percentage
        missing_data_percentage = self._calculate_missing_data_percentage(data)
        
        report = DataQualityReport(
            overall_score=overall_score,
            quality_level=quality_level,
            validation_results=validation_results,
            recommendations=recommendations,
            timestamp=datetime.now(),
            data_shape=data.shape,
            missing_data_percentage=missing_data_percentage
        )
        
        logger.info(f"Data validation completed. Overall score: {overall_score:.2f}")
        return report
    
    def _validate_basic_structure(self, data: pd.DataFrame, data_type: str) -> List[ValidationResult]:
        """Validate basic data structure"""
        results = []
        
        # Check if data is empty
        if data.empty:
            results.append(ValidationResult(
                is_valid=False,
                score=0.0,
                level=ValidationLevel.CRITICAL,
                message="Data is empty",
                details={'rows': 0, 'columns': 0}
            ))
            return results
        
        # Check minimum number of rows
        min_rows = self.config.get('min_rows', 10)
        if len(data) < min_rows:
            results.append(ValidationResult(
                is_valid=False,
                score=0.5,
                level=ValidationLevel.HIGH,
                message=f"Data has only {len(data)} rows, minimum required: {min_rows}",
                details={'actual_rows': len(data), 'min_required': min_rows}
            ))
        else:
            results.append(ValidationResult(
                is_valid=True,
                score=1.0,
                level=ValidationLevel.INFO,
                message=f"Data has sufficient rows: {len(data)}",
                details={'rows': len(data)}
            ))
        
        # Check minimum number of columns
        min_cols = self.config.get('min_columns', 3)
        if len(data.columns) < min_cols:
            results.append(ValidationResult(
                is_valid=False,
                score=0.5,
                level=ValidationLevel.HIGH,
                message=f"Data has only {len(data.columns)} columns, minimum required: {min_cols}",
                details={'actual_columns': len(data.columns), 'min_required': min_cols}
            ))
        else:
            results.append(ValidationResult(
                is_valid=True,
                score=1.0,
                level=ValidationLevel.INFO,
                message=f"Data has sufficient columns: {len(data.columns)}",
                details={'columns': len(data.columns)}
            ))
        
        return results
    
    def _validate_completeness(self, data: pd.DataFrame, data_type: str) -> List[ValidationResult]:
        """Validate data completeness"""
        results = []
        
        # Calculate completeness score
        total_cells = data.shape[0] * data.shape[1]
        non_null_cells = data.count().sum()
        completeness_score = non_null_cells / total_cells if total_cells > 0 else 0.0
        
        # Check against threshold
        threshold = self.config['completeness_threshold']
        if completeness_score >= threshold:
            results.append(ValidationResult(
                is_valid=True,
                score=completeness_score,
                level=ValidationLevel.INFO,
                message=f"Data completeness is good: {completeness_score:.2%}",
                details={'completeness_score': completeness_score, 'threshold': threshold}
            ))
        else:
            results.append(ValidationResult(
                is_valid=False,
                score=completeness_score,
                level=ValidationLevel.HIGH,
                message=f"Data completeness is below threshold: {completeness_score:.2%} < {threshold:.2%}",
                details={'completeness_score': completeness_score, 'threshold': threshold}
            ))
        
        # Check individual column completeness
        for column in data.columns:
            column_completeness = data[column].count() / len(data)
            if column_completeness < 0.5:  # Less than 50% complete
                results.append(ValidationResult(
                    is_valid=False,
                    score=column_completeness,
                    level=ValidationLevel.MEDIUM,
                    message=f"Column '{column}' has low completeness: {column_completeness:.2%}",
                    details={'column': column, 'completeness': column_completeness}
                ))
        
        return results
    
    def _validate_accuracy(self, data: pd.DataFrame, data_type: str) -> List[ValidationResult]:
        """Validate data accuracy"""
        results = []
        
        # Get validation rules for this data type
        rules = self.validation_rules.get(data_type, {})
        text_validation = rules.get('text_validation', {})
        numeric_validation = rules.get('numeric_validation', {})
        
        # Validate text columns
        for column, rules in text_validation.items():
            if column in data.columns:
                accuracy_score = self._validate_text_column(data[column], rules)
                results.append(ValidationResult(
                    is_valid=accuracy_score >= 0.9,
                    score=accuracy_score,
                    level=ValidationLevel.MEDIUM if accuracy_score < 0.9 else ValidationLevel.INFO,
                    message=f"Text column '{column}' accuracy: {accuracy_score:.2%}",
                    details={'column': column, 'accuracy': accuracy_score}
                ))
        
        # Validate numeric columns
        for column, rules in numeric_validation.items():
            if column in data.columns:
                accuracy_score = self._validate_numeric_column(data[column], rules)
                results.append(ValidationResult(
                    is_valid=accuracy_score >= 0.9,
                    score=accuracy_score,
                    level=ValidationLevel.MEDIUM if accuracy_score < 0.9 else ValidationLevel.INFO,
                    message=f"Numeric column '{column}' accuracy: {accuracy_score:.2%}",
                    details={'column': column, 'accuracy': accuracy_score}
                ))
        
        return results
    
    def _validate_text_column(self, series: pd.Series, rules: Dict[str, Any]) -> float:
        """Validate a text column against rules"""
        if series.empty:
            return 0.0
        
        valid_count = 0
        total_count = series.count()
        
        if total_count == 0:
            return 0.0
        
        for value in series.dropna():
            is_valid = True
            
            # Check length constraints
            if 'min_length' in rules and len(str(value)) < rules['min_length']:
                is_valid = False
            if 'max_length' in rules and len(str(value)) > rules['max_length']:
                is_valid = False
            
            # Check pattern
            if 'pattern' in rules and not re.match(rules['pattern'], str(value)):
                is_valid = False
            
            if is_valid:
                valid_count += 1
        
        return valid_count / total_count
    
    def _validate_numeric_column(self, series: pd.Series, rules: Dict[str, Any]) -> float:
        """Validate a numeric column against rules"""
        if series.empty:
            return 0.0
        
        valid_count = 0
        total_count = series.count()
        
        if total_count == 0:
            return 0.0
        
        for value in series.dropna():
            is_valid = True
            
            # Check value range
            if 'min_value' in rules and value < rules['min_value']:
                is_valid = False
            if 'max_value' in rules and value > rules['max_value']:
                is_valid = False
            
            if is_valid:
                valid_count += 1
        
        return valid_count / total_count
    
    def _validate_consistency(self, data: pd.DataFrame, data_type: str) -> List[ValidationResult]:
        """Validate data consistency"""
        results = []
        
        # Check for consistent data types
        for column in data.columns:
            if data[column].dtype == 'object':
                # Check if all values are strings
                string_count = data[column].apply(lambda x: isinstance(x, str)).sum()
                consistency_score = string_count / len(data)
                
                results.append(ValidationResult(
                    is_valid=consistency_score >= 0.95,
                    score=consistency_score,
                    level=ValidationLevel.MEDIUM if consistency_score < 0.95 else ValidationLevel.INFO,
                    message=f"Column '{column}' type consistency: {consistency_score:.2%}",
                    details={'column': column, 'consistency': consistency_score}
                ))
        
        # Check for consistent date formats
        date_columns = data.select_dtypes(include=['datetime64[ns]']).columns
        for column in date_columns:
            if not data[column].empty:
                # Check if all dates are valid
                valid_dates = data[column].notna().sum()
                total_dates = len(data[column])
                consistency_score = valid_dates / total_dates
                
                results.append(ValidationResult(
                    is_valid=consistency_score >= 0.95,
                    score=consistency_score,
                    level=ValidationLevel.MEDIUM if consistency_score < 0.95 else ValidationLevel.INFO,
                    message=f"Date column '{column}' consistency: {consistency_score:.2%}",
                    details={'column': column, 'consistency': consistency_score}
                ))
        
        return results
    
    def _validate_timeliness(self, data: pd.DataFrame, data_type: str) -> List[ValidationResult]:
        """Validate data timeliness"""
        results = []
        
        # Get timeliness rules
        rules = self.validation_rules.get(data_type, {})
        date_validation = rules.get('date_validation', {})
        
        for column, rules in date_validation.items():
            if column in data.columns:
                timeliness_score = self._validate_timeliness_column(data[column], rules)
                results.append(ValidationResult(
                    is_valid=timeliness_score >= 0.8,
                    score=timeliness_score,
                    level=ValidationLevel.MEDIUM if timeliness_score < 0.8 else ValidationLevel.INFO,
                    message=f"Column '{column}' timeliness: {timeliness_score:.2%}",
                    details={'column': column, 'timeliness': timeliness_score}
                ))
        
        return results
    
    def _validate_timeliness_column(self, series: pd.Series, rules: Dict[str, Any]) -> float:
        """Validate timeliness of a date column"""
        if series.empty:
            return 0.0
        
        valid_count = 0
        total_count = series.count()
        
        if total_count == 0:
            return 0.0
        
        for value in series.dropna():
            is_valid = True
            
            # Check date range
            if 'min_date' in rules:
                min_date = pd.to_datetime(rules['min_date'])
                if pd.to_datetime(value) < min_date:
                    is_valid = False
            
            if 'max_date' in rules:
                max_date = pd.to_datetime(rules['max_date'])
                if pd.to_datetime(value) > max_date:
                    is_valid = False
            
            # Check timestamp range
            if 'min_timestamp' in rules and value < rules['min_timestamp']:
                is_valid = False
            if 'max_timestamp' in rules and value > rules['max_timestamp']:
                is_valid = False
            
            if is_valid:
                valid_count += 1
        
        return valid_count / total_count
    
    def _validate_schema(self, data: pd.DataFrame, data_type: str) -> List[ValidationResult]:
        """Validate data schema"""
        results = []
        
        # Get schema rules
        rules = self.validation_rules.get(data_type, {})
        required_columns = rules.get('required_columns', [])
        column_types = rules.get('column_types', {})
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            results.append(ValidationResult(
                is_valid=False,
                score=0.0,
                level=ValidationLevel.CRITICAL,
                message=f"Missing required columns: {missing_columns}",
                details={'missing_columns': missing_columns, 'required_columns': required_columns}
            ))
        else:
            results.append(ValidationResult(
                is_valid=True,
                score=1.0,
                level=ValidationLevel.INFO,
                message="All required columns present",
                details={'required_columns': required_columns}
            ))
        
        # Check column types
        for column, expected_type in column_types.items():
            if column in data.columns:
                actual_type = str(data[column].dtype)
                if actual_type == expected_type:
                    results.append(ValidationResult(
                        is_valid=True,
                        score=1.0,
                        level=ValidationLevel.INFO,
                        message=f"Column '{column}' has correct type: {actual_type}",
                        details={'column': column, 'type': actual_type}
                    ))
                else:
                    results.append(ValidationResult(
                        is_valid=False,
                        score=0.5,
                        level=ValidationLevel.MEDIUM,
                        message=f"Column '{column}' has wrong type: {actual_type} (expected: {expected_type})",
                        details={'column': column, 'actual_type': actual_type, 'expected_type': expected_type}
                    ))
        
        return results
    
    def _validate_outliers(self, data: pd.DataFrame, data_type: str) -> List[ValidationResult]:
        """Validate for outliers in numeric columns"""
        results = []
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if data[column].count() < 10:  # Need at least 10 values for outlier detection
                continue
            
            # Use Isolation Forest for outlier detection
            outlier_detector = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = outlier_detector.fit_predict(data[[column]].dropna().values.reshape(-1, 1))
            
            outlier_count = (outlier_labels == -1).sum()
            total_count = len(outlier_labels)
            outlier_percentage = outlier_count / total_count
            
            threshold = self.config['outlier_threshold']
            if outlier_percentage <= threshold:
                results.append(ValidationResult(
                    is_valid=True,
                    score=1.0 - outlier_percentage,
                    level=ValidationLevel.INFO,
                    message=f"Column '{column}' has acceptable outlier percentage: {outlier_percentage:.2%}",
                    details={'column': column, 'outlier_percentage': outlier_percentage, 'threshold': threshold}
                ))
            else:
                results.append(ValidationResult(
                    is_valid=False,
                    score=1.0 - outlier_percentage,
                    level=ValidationLevel.MEDIUM,
                    message=f"Column '{column}' has high outlier percentage: {outlier_percentage:.2%} > {threshold:.2%}",
                    details={'column': column, 'outlier_percentage': outlier_percentage, 'threshold': threshold}
                ))
        
        return results
    
    def _validate_duplicates(self, data: pd.DataFrame, data_type: str) -> List[ValidationResult]:
        """Validate for duplicate records"""
        results = []
        
        # Check for exact duplicates
        duplicate_count = data.duplicated().sum()
        total_count = len(data)
        duplicate_percentage = duplicate_count / total_count if total_count > 0 else 0.0
        
        threshold = self.config['duplicate_threshold']
        if duplicate_percentage <= threshold:
            results.append(ValidationResult(
                is_valid=True,
                score=1.0 - duplicate_percentage,
                level=ValidationLevel.INFO,
                message=f"Duplicate percentage is acceptable: {duplicate_percentage:.2%}",
                details={'duplicate_percentage': duplicate_percentage, 'threshold': threshold}
            ))
        else:
            results.append(ValidationResult(
                is_valid=False,
                score=1.0 - duplicate_percentage,
                level=ValidationLevel.MEDIUM,
                message=f"High duplicate percentage: {duplicate_percentage:.2%} > {threshold:.2%}",
                details={'duplicate_percentage': duplicate_percentage, 'threshold': threshold}
            ))
        
        return results
    
    def _calculate_overall_score(self, validation_results: List[ValidationResult]) -> float:
        """Calculate overall validation score"""
        if not validation_results:
            return 0.0
        
        # Weight different validation levels
        weights = {
            ValidationLevel.CRITICAL: 1.0,
            ValidationLevel.HIGH: 0.8,
            ValidationLevel.MEDIUM: 0.6,
            ValidationLevel.LOW: 0.4,
            ValidationLevel.INFO: 0.2
        }
        
        weighted_scores = []
        total_weight = 0
        
        for result in validation_results:
            weight = weights.get(result.level, 0.5)
            weighted_scores.append(result.score * weight)
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return sum(weighted_scores) / total_weight
    
    def _determine_quality_level(self, score: float) -> DataQualityScore:
        """Determine data quality level based on score"""
        if score >= 0.9:
            return DataQualityScore.EXCELLENT
        elif score >= 0.8:
            return DataQualityScore.GOOD
        elif score >= 0.7:
            return DataQualityScore.FAIR
        elif score >= 0.6:
            return DataQualityScore.POOR
        else:
            return DataQualityScore.CRITICAL
    
    def _generate_recommendations(self, validation_results: List[ValidationResult], 
                                data_type: str) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Analyze validation results
        critical_issues = [r for r in validation_results if r.level == ValidationLevel.CRITICAL]
        high_issues = [r for r in validation_results if r.level == ValidationLevel.HIGH]
        medium_issues = [r for r in validation_results if r.level == ValidationLevel.MEDIUM]
        
        if critical_issues:
            recommendations.append("CRITICAL: Address critical data quality issues immediately")
        
        if high_issues:
            recommendations.append("HIGH: Fix high-priority data quality issues")
        
        if medium_issues:
            recommendations.append("MEDIUM: Consider addressing medium-priority issues")
        
        # Specific recommendations based on data type
        if data_type == 'reddit':
            recommendations.append("Consider implementing Reddit API rate limiting")
            recommendations.append("Validate subreddit names and post formats")
        
        elif data_type == 'gdelt':
            recommendations.append("Verify GDELT API response format")
            recommendations.append("Check date format consistency")
        
        elif data_type == 'yahoo_finance':
            recommendations.append("Validate financial data ranges")
            recommendations.append("Check for missing trading days")
        
        return recommendations
    
    def _calculate_missing_data_percentage(self, data: pd.DataFrame) -> float:
        """Calculate percentage of missing data"""
        if data.empty:
            return 100.0
        
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        return (missing_cells / total_cells) * 100

# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'title': ['Sample post 1', 'Sample post 2', 'Sample post 3'],
        'subreddit': ['investing', 'stocks', 'investing'],
        'score': [100, 200, 150],
        'created_utc': [1640995200, 1640995300, 1640995400]
    })
    
    # Initialize validator
    validator = DataValidator()
    
    # Validate data
    report = validator.validate_data(sample_data, 'reddit')
    
    # Print results
    print(f"Overall Score: {report.overall_score:.2f}")
    print(f"Quality Level: {report.quality_level.value}")
    print(f"Missing Data: {report.missing_data_percentage:.2f}%")
    print("\nRecommendations:")
    for rec in report.recommendations:
        print(f"- {rec}")
