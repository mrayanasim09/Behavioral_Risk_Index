"""
Unit Tests for Data Validation Module

This module provides comprehensive unit tests for the data validation
functionality, including data quality assessment, validation rules,
and error handling.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_validation import (
    DataValidator,
    ValidationResult,
    DataQualityReport,
    DataQualityScore,
    ValidationLevel
)

class TestDataValidator(unittest.TestCase):
    """Test DataValidator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = DataValidator()
    
    def test_validate_data_gdelt(self):
        """Test GDELT data validation"""
        # Create test GDELT data
        gdelt_data = pd.DataFrame({
            'GLOBALEVENTID': range(100),
            'date': pd.date_range('2024-01-01', periods=100),
            'Actor1Name': [f'Actor{i}' for i in range(100)],
            'Actor2Name': [f'Actor{i+100}' for i in range(100)],
            'EventCode': [f'EVENT{i}' for i in range(100)],
            'GoldsteinScale': np.random.uniform(-10, 10, 100),
            'AvgTone': np.random.uniform(-100, 100, 100),
            'NumMentions': np.random.randint(1, 100, 100),
            'NumSources': np.random.randint(1, 50, 100),
            'NumArticles': np.random.randint(1, 200, 100),
            'SOURCEURL': [f'http://example.com/{i}' for i in range(100)]
        })
        
        result = self.validator.validate_data(gdelt_data, 'gdelt')
        
        self.assertIsInstance(result, DataQualityReport)
        self.assertIn('overall_score', result.to_dict())
        self.assertIn('quality_level', result.to_dict())
        self.assertIn('validation_results', result.to_dict())
        self.assertIn('recommendations', result.to_dict())
    
    def test_validate_data_reddit(self):
        """Test Reddit data validation"""
        # Create test Reddit data
        reddit_data = pd.DataFrame({
            'title': [f'Post {i}' for i in range(50)],
            'subreddit': ['investing'] * 25 + ['stocks'] * 25,
            'score': np.random.randint(0, 1000, 50),
            'created_utc': np.random.uniform(1600000000, 1700000000, 50),
            'num_comments': np.random.randint(0, 100, 50),
            'author': [f'user{i}' for i in range(50)],
            'is_self': [True] * 25 + [False] * 25,
            'upvote_ratio': np.random.uniform(0, 1, 50)
        })
        
        result = self.validator.validate_data(reddit_data, 'reddit')
        
        self.assertIsInstance(result, DataQualityReport)
        self.assertGreaterEqual(result.overall_score, 0)
        self.assertLessEqual(result.overall_score, 1)
    
    def test_validate_data_yahoo_finance(self):
        """Test Yahoo Finance data validation"""
        # Create test Yahoo Finance data
        yahoo_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.randint(1000000, 10000000, 100),
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(100, 200, 100),
            'Low': np.random.uniform(100, 200, 100)
        })
        
        result = self.validator.validate_data(yahoo_data, 'yahoo_finance')
        
        self.assertIsInstance(result, DataQualityReport)
        self.assertGreaterEqual(result.overall_score, 0)
        self.assertLessEqual(result.overall_score, 1)
    
    def test_validate_data_empty(self):
        """Test validation with empty data"""
        empty_data = pd.DataFrame()
        result = self.validator.validate_data(empty_data, 'gdelt')
        
        self.assertIsInstance(result, DataQualityReport)
        self.assertEqual(result.overall_score, 0)
        self.assertEqual(result.quality_level, DataQualityScore.CRITICAL)
    
    def test_validate_basic_structure(self):
        """Test basic structure validation"""
        # Test with sufficient data
        good_data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        results = self.validator._validate_basic_structure(good_data, 'test')
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Check that all results are ValidationResult objects
        for result in results:
            self.assertIsInstance(result, ValidationResult)
    
    def test_validate_completeness(self):
        """Test completeness validation"""
        # Test with complete data
        complete_data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        results = self.validator._validate_completeness(complete_data, 'test')
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Test with incomplete data
        incomplete_data = pd.DataFrame({
            'col1': [1, 2, None, 4, 5],
            'col2': ['a', 'b', 'c', None, 'e'],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        results = self.validator._validate_completeness(incomplete_data, 'test')
        
        # Should have results for overall completeness and individual columns
        self.assertGreater(len(results), 1)
    
    def test_validate_accuracy(self):
        """Test accuracy validation"""
        # Test with accurate data
        accurate_data = pd.DataFrame({
            'title': ['Valid Title 1', 'Valid Title 2', 'Valid Title 3'],
            'score': [100, 200, 300],
            'num_comments': [10, 20, 30]
        })
        
        results = self.validator._validate_accuracy(accurate_data, 'reddit')
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Test with inaccurate data
        inaccurate_data = pd.DataFrame({
            'title': ['A', 'B', 'C'],  # Too short
            'score': [-1000, 200, 300],  # Out of range
            'num_comments': [10, 20, 30]
        })
        
        results = self.validator._validate_accuracy(inaccurate_data, 'reddit')
        
        # Should have results for each column
        self.assertGreater(len(results), 0)
    
    def test_validate_consistency(self):
        """Test consistency validation"""
        # Test with consistent data
        consistent_data = pd.DataFrame({
            'col1': ['a', 'b', 'c', 'd', 'e'],
            'col2': [1, 2, 3, 4, 5],
            'col3': pd.date_range('2024-01-01', periods=5)
        })
        
        results = self.validator._validate_consistency(consistent_data, 'test')
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
    
    def test_validate_timeliness(self):
        """Test timeliness validation"""
        # Test with recent data
        recent_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'created_utc': np.random.uniform(1700000000, 1701000000, 10)  # Recent timestamps
        })
        
        results = self.validator._validate_timeliness(recent_data, 'reddit')
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
    
    def test_validate_schema(self):
        """Test schema validation"""
        # Test with correct schema
        correct_schema_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'title': [f'Title {i}' for i in range(10)],
            'subreddit': ['investing'] * 10,
            'score': np.random.randint(0, 1000, 10),
            'created_utc': np.random.uniform(1600000000, 1700000000, 10)
        })
        
        results = self.validator._validate_schema(correct_schema_data, 'reddit')
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Test with missing required columns
        missing_columns_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'title': [f'Title {i}' for i in range(10)]
            # Missing required columns: subreddit, score, created_utc
        })
        
        results = self.validator._validate_schema(missing_columns_data, 'reddit')
        
        # Should have critical error for missing columns
        critical_errors = [r for r in results if r.level == ValidationLevel.CRITICAL]
        self.assertGreater(len(critical_errors), 0)
    
    def test_validate_outliers(self):
        """Test outlier validation"""
        # Test with normal data
        normal_data = pd.DataFrame({
            'score': np.random.normal(100, 20, 100),
            'num_comments': np.random.normal(50, 10, 100)
        })
        
        results = self.validator._validate_outliers(normal_data, 'reddit')
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Test with outlier data
        outlier_data = pd.DataFrame({
            'score': [100] * 90 + [10000] * 10,  # 10% outliers
            'num_comments': [50] * 90 + [1000] * 10
        })
        
        results = self.validator._validate_outliers(outlier_data, 'reddit')
        
        # Should detect outliers
        self.assertGreater(len(results), 0)
    
    def test_validate_duplicates(self):
        """Test duplicate validation"""
        # Test with no duplicates
        no_duplicates_data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e']
        })
        
        results = self.validator._validate_duplicates(no_duplicates_data, 'test')
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        
        # Test with duplicates
        duplicates_data = pd.DataFrame({
            'col1': [1, 2, 3, 1, 2],
            'col2': ['a', 'b', 'c', 'a', 'b']
        })
        
        results = self.validator._validate_duplicates(duplicates_data, 'test')
        
        # Should detect duplicates
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], ValidationResult)
    
    def test_calculate_overall_score(self):
        """Test overall score calculation"""
        # Test with mixed results
        results = [
            ValidationResult(True, 0.9, ValidationLevel.INFO, "Good"),
            ValidationResult(False, 0.5, ValidationLevel.HIGH, "Bad"),
            ValidationResult(True, 0.8, ValidationLevel.MEDIUM, "OK")
        ]
        
        score = self.validator._calculate_overall_score(results)
        
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        self.assertIsInstance(score, float)
    
    def test_determine_quality_level(self):
        """Test quality level determination"""
        # Test different score ranges
        self.assertEqual(
            self.validator._determine_quality_level(0.95),
            DataQualityScore.EXCELLENT
        )
        self.assertEqual(
            self.validator._determine_quality_level(0.85),
            DataQualityScore.GOOD
        )
        self.assertEqual(
            self.validator._determine_quality_level(0.75),
            DataQualityScore.FAIR
        )
        self.assertEqual(
            self.validator._determine_quality_level(0.65),
            DataQualityScore.POOR
        )
        self.assertEqual(
            self.validator._determine_quality_level(0.45),
            DataQualityScore.CRITICAL
        )
    
    def test_generate_recommendations(self):
        """Test recommendation generation"""
        # Test with mixed results
        results = [
            ValidationResult(False, 0.3, ValidationLevel.CRITICAL, "Critical issue"),
            ValidationResult(False, 0.6, ValidationLevel.HIGH, "High priority issue"),
            ValidationResult(False, 0.7, ValidationLevel.MEDIUM, "Medium priority issue")
        ]
        
        recommendations = self.validator._generate_recommendations(results, 'reddit')
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Should have recommendations for different levels
        self.assertTrue(any('CRITICAL' in rec for rec in recommendations))
        self.assertTrue(any('HIGH' in rec for rec in recommendations))
        self.assertTrue(any('MEDIUM' in rec for rec in recommendations))
    
    def test_calculate_missing_data_percentage(self):
        """Test missing data percentage calculation"""
        # Test with no missing data
        complete_data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e']
        })
        
        percentage = self.validator._calculate_missing_data_percentage(complete_data)
        self.assertEqual(percentage, 0.0)
        
        # Test with some missing data
        incomplete_data = pd.DataFrame({
            'col1': [1, 2, None, 4, 5],
            'col2': ['a', 'b', 'c', None, 'e']
        })
        
        percentage = self.validator._calculate_missing_data_percentage(incomplete_data)
        self.assertGreater(percentage, 0)
        self.assertLess(percentage, 100)
        
        # Test with empty data
        empty_data = pd.DataFrame()
        percentage = self.validator._calculate_missing_data_percentage(empty_data)
        self.assertEqual(percentage, 100.0)

class TestValidationResult(unittest.TestCase):
    """Test ValidationResult class"""
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation"""
        result = ValidationResult(
            is_valid=True,
            score=0.8,
            level=ValidationLevel.INFO,
            message="Test message",
            details={'key': 'value'}
        )
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.score, 0.8)
        self.assertEqual(result.level, ValidationLevel.INFO)
        self.assertEqual(result.message, "Test message")
        self.assertEqual(result.details, {'key': 'value'})
    
    def test_validation_result_to_dict(self):
        """Test ValidationResult to_dict method"""
        result = ValidationResult(
            is_valid=True,
            score=0.8,
            level=ValidationLevel.INFO,
            message="Test message",
            details={'key': 'value'}
        )
        
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertIn('is_valid', result_dict)
        self.assertIn('score', result_dict)
        self.assertIn('level', result_dict)
        self.assertIn('message', result_dict)
        self.assertIn('details', result_dict)
        
        self.assertTrue(result_dict['is_valid'])
        self.assertEqual(result_dict['score'], 0.8)
        self.assertEqual(result_dict['level'], 'info')
        self.assertEqual(result_dict['message'], "Test message")
        self.assertEqual(result_dict['details'], {'key': 'value'})

class TestDataQualityReport(unittest.TestCase):
    """Test DataQualityReport class"""
    
    def test_data_quality_report_creation(self):
        """Test DataQualityReport creation"""
        validation_results = [
            ValidationResult(True, 0.8, ValidationLevel.INFO, "Good"),
            ValidationResult(False, 0.6, ValidationLevel.MEDIUM, "OK")
        ]
        
        report = DataQualityReport(
            overall_score=0.7,
            quality_level=DataQualityScore.FAIR,
            validation_results=validation_results,
            recommendations=["Fix issue 1", "Improve data quality"],
            timestamp=datetime.now(),
            data_shape=(100, 5),
            missing_data_percentage=10.0
        )
        
        self.assertEqual(report.overall_score, 0.7)
        self.assertEqual(report.quality_level, DataQualityScore.FAIR)
        self.assertEqual(len(report.validation_results), 2)
        self.assertEqual(len(report.recommendations), 2)
        self.assertEqual(report.data_shape, (100, 5))
        self.assertEqual(report.missing_data_percentage, 10.0)
    
    def test_data_quality_report_to_dict(self):
        """Test DataQualityReport to_dict method"""
        validation_results = [
            ValidationResult(True, 0.8, ValidationLevel.INFO, "Good")
        ]
        
        report = DataQualityReport(
            overall_score=0.8,
            quality_level=DataQualityScore.GOOD,
            validation_results=validation_results,
            recommendations=["Good data"],
            timestamp=datetime.now(),
            data_shape=(100, 5),
            missing_data_percentage=5.0
        )
        
        report_dict = report.to_dict()
        
        self.assertIsInstance(report_dict, dict)
        self.assertIn('overall_score', report_dict)
        self.assertIn('quality_level', report_dict)
        self.assertIn('validation_results', report_dict)
        self.assertIn('recommendations', report_dict)
        self.assertIn('timestamp', report_dict)
        self.assertIn('data_shape', report_dict)
        self.assertIn('missing_data_percentage', report_dict)
        
        self.assertEqual(report_dict['overall_score'], 0.8)
        self.assertEqual(report_dict['quality_level'], 'good')
        self.assertEqual(len(report_dict['validation_results']), 1)
        self.assertEqual(len(report_dict['recommendations']), 1)
        self.assertEqual(report_dict['data_shape'], (100, 5))
        self.assertEqual(report_dict['missing_data_percentage'], 5.0)

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
