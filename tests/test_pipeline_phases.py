"""
Comprehensive Unit Tests for Pipeline Phases

This module provides comprehensive unit tests for all pipeline phases,
including data collection, preprocessing, feature engineering, BRI calculation,
validation, and visualization components.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline_phases import (
    Phase1DataCollection,
    Phase2DataPreprocessing,
    Phase3FeatureEngineering,
    Phase4BRICalculation,
    Phase5AnalysisValidation,
    Phase6Visualization,
    Phase7FinalDeliverables
)

class TestPhase1DataCollection(unittest.TestCase):
    """Test Phase 1: Data Collection"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_data_collector = Mock()
        self.mock_gdelt_processor = Mock()
        self.phase1 = Phase1DataCollection(self.mock_data_collector, self.mock_gdelt_processor)
    
    def test_collect_market_data(self):
        """Test market data collection"""
        # Mock market data
        mock_market_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'close': np.random.randn(10) * 100 + 1000,
            'volume': np.random.randint(1000000, 10000000, 10)
        })
        
        self.mock_data_collector.collect_market_data.return_value = mock_market_data
        
        result = self.phase1.collect_market_data('2024-01-01', '2024-01-10')
        
        self.assertEqual(len(result), 10)
        self.assertIn('date', result.columns)
        self.assertIn('close', result.columns)
        self.assertIn('volume', result.columns)
        self.mock_data_collector.collect_market_data.assert_called_once_with('2024-01-01', '2024-01-10')
    
    def test_process_gdelt_data(self):
        """Test GDELT data processing"""
        # Mock GDELT data
        mock_gdelt_data = pd.DataFrame({
            'GLOBALEVENTID': range(100),
            'date': pd.date_range('2024-01-01', periods=100),
            'GoldsteinScale': np.random.uniform(-10, 10, 100),
            'AvgTone': np.random.uniform(-100, 100, 100)
        })
        
        self.mock_gdelt_processor.process_export_file.return_value = mock_gdelt_data
        
        result = self.phase1.process_gdelt_data('test_file.csv')
        
        self.assertEqual(len(result), 100)
        self.assertIn('GLOBALEVENTID', result.columns)
        self.assertIn('date', result.columns)
        self.mock_gdelt_processor.process_export_file.assert_called_once_with('test_file.csv')
    
    def test_collect_reddit_data(self):
        """Test Reddit data collection"""
        # Mock Reddit data
        mock_reddit_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=50),
            'title': [f'Post {i}' for i in range(50)],
            'score': np.random.randint(0, 1000, 50),
            'num_comments': np.random.randint(0, 100, 50)
        })
        
        self.mock_data_collector.collect_reddit_data.return_value = mock_reddit_data
        
        result = self.phase1.collect_reddit_data('2024-01-01', '2024-01-10')
        
        self.assertEqual(len(result), 50)
        self.assertIn('date', result.columns)
        self.assertIn('title', result.columns)
        self.mock_data_collector.collect_reddit_data.assert_called_once_with('2024-01-01', '2024-01-10')

class TestPhase2DataPreprocessing(unittest.TestCase):
    """Test Phase 2: Data Preprocessing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_text_preprocessor = Mock()
        self.phase2 = Phase2DataPreprocessing(self.mock_text_preprocessor)
    
    def test_clean_gdelt_data(self):
        """Test GDELT data cleaning"""
        # Create test GDELT data
        gdelt_data = pd.DataFrame({
            'GLOBALEVENTID': range(10),
            'date': pd.date_range('2024-01-01', periods=10),
            'GoldsteinScale': np.random.uniform(-10, 10, 10),
            'AvgTone': np.random.uniform(-100, 100, 10),
            'NumMentions': np.random.randint(1, 100, 10),
            'NumSources': np.random.randint(1, 50, 10),
            'NumArticles': np.random.randint(1, 200, 10)
        })
        
        result = self.phase2.clean_gdelt_data(gdelt_data)
        
        self.assertEqual(len(result), 10)
        self.assertIn('date', result.columns)
        self.assertIn('avg_goldstein_tone', result.columns)
        self.assertIn('avg_tone', result.columns)
        self.assertIn('event_count', result.columns)
        
        # Check Goldstein normalization
        self.assertTrue(result['avg_goldstein_tone'].min() >= 0)
        self.assertTrue(result['avg_goldstein_tone'].max() <= 1)
    
    def test_clean_gdelt_data_empty(self):
        """Test GDELT data cleaning with empty data"""
        empty_data = pd.DataFrame()
        result = self.phase2.clean_gdelt_data(empty_data)
        self.assertTrue(result.empty)
    
    def test_clean_reddit_text(self):
        """Test Reddit text cleaning"""
        # Create test Reddit data
        reddit_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'combined_text': ['This is a test post', 'Another test post', 'Third test post', '', 'Short'],
            'subreddit': ['investing', 'stocks', 'investing', 'stocks', 'investing'],
            'score': [100, 200, 150, 50, 5],
            'num_comments': [10, 20, 15, 5, 1],
            'url': ['http://example.com'] * 5
        })
        
        # Mock text preprocessor
        self.mock_text_preprocessor.preprocess_text.side_effect = lambda x: x.lower() if len(x) > 10 else None
        
        result = self.phase2.clean_reddit_text(reddit_data)
        
        # Should filter out empty and short texts
        self.assertEqual(len(result), 3)  # Only 3 posts with length > 10
        self.assertIn('text', result.columns)
        self.assertIn('subreddit', result.columns)
        self.assertIn('score', result.columns)
    
    def test_perform_sentiment_analysis(self):
        """Test sentiment analysis"""
        # Create test data
        reddit_clean = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'text': ['This is positive', 'This is negative', 'This is neutral'],
            'subreddit': ['investing', 'stocks', 'investing'],
            'score': [100, 200, 150],
            'num_comments': [10, 20, 15]
        })
        
        gdelt_clean = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'avg_tone': [50, -30, 0],
            'total_mentions': [100, 200, 150]
        })
        
        # Mock sentiment pipeline
        with patch('pipeline_phases.pipeline') as mock_pipeline:
            mock_pipeline.return_value = Mock()
            mock_pipeline.return_value.side_effect = [
                [{'label': 'POSITIVE', 'score': 0.8}],
                [{'label': 'NEGATIVE', 'score': 0.7}],
                [{'label': 'POSITIVE', 'score': 0.5}]
            ]
            
            result = self.phase2.perform_sentiment_analysis(reddit_clean, gdelt_clean)
            
            self.assertEqual(len(result), 6)  # 3 Reddit + 3 GDELT
            self.assertIn('sentiment', result.columns)
            self.assertIn('confidence', result.columns)
            self.assertIn('source', result.columns)

class TestPhase3FeatureEngineering(unittest.TestCase):
    """Test Phase 3: Feature Engineering"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.phase3 = Phase3FeatureEngineering()
    
    def test_create_behavioral_features(self):
        """Test behavioral feature creation"""
        # Create test sentiment data
        sentiment_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'sentiment': np.random.uniform(-1, 1, 10),
            'confidence': np.random.uniform(0, 1, 10),
            'engagement': np.random.randint(0, 1000, 10),
            'source': ['reddit'] * 10,
            'subreddit': ['investing'] * 10
        })
        
        gdelt_clean = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'avg_tone': np.random.uniform(-100, 100, 10),
            'event_count': np.random.randint(1, 100, 10)
        })
        
        reddit_clean = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'score': np.random.randint(0, 1000, 10),
            'num_comments': np.random.randint(0, 100, 10)
        })
        
        result = self.phase3.create_behavioral_features(sentiment_data, gdelt_clean, reddit_clean)
        
        self.assertEqual(len(result), 10)
        self.assertIn('sentiment_volatility', result.columns)
        self.assertIn('news_tone', result.columns)
        self.assertIn('media_herding', result.columns)
        self.assertIn('polarity_skew', result.columns)
        self.assertIn('event_density', result.columns)
        self.assertIn('engagement_index', result.columns)
        self.assertIn('fear_index', result.columns)
        self.assertIn('overconfidence_index', result.columns)
    
    def test_calculate_behavioral_features(self):
        """Test specific behavioral feature calculations"""
        # Create test data
        daily_sentiment = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'sentiment_mean': [0.5, -0.3, 0.1, 0.8, -0.2],
            'sentiment_std': [0.2, 0.4, 0.1, 0.3, 0.5],
            'sentiment_count': [10, 15, 8, 12, 20],
            'confidence_mean': [0.8, 0.6, 0.9, 0.7, 0.5],
            'engagement_sum': [100, 200, 150, 300, 250]
        })
        
        gdelt_clean = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'avg_tone': [50, -30, 10, 80, -20],
            'event_count': [10, 15, 8, 12, 20]
        })
        
        reddit_clean = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'score': [100, 200, 150, 300, 250],
            'num_comments': [10, 20, 15, 30, 25]
        })
        
        result = self.phase3._calculate_behavioral_features(daily_sentiment, gdelt_clean, reddit_clean)
        
        # Check that all features are calculated
        expected_features = [
            'sentiment_volatility', 'news_tone', 'media_herding',
            'polarity_skew', 'event_density', 'engagement_index',
            'fear_index', 'overconfidence_index'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, result.columns)
            self.assertFalse(result[feature].isna().all())  # Should not be all NaN

class TestPhase4BRICalculation(unittest.TestCase):
    """Test Phase 4: BRI Calculation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.phase4 = Phase4BRICalculation()
    
    def test_calculate_bri(self):
        """Test BRI calculation"""
        # Create test behavioral features
        behavioral_features = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'sentiment_volatility': np.random.uniform(0, 1, 10),
            'news_tone': np.random.uniform(0, 1, 10),
            'media_herding': np.random.uniform(0, 1, 10),
            'polarity_skew': np.random.uniform(0, 1, 10),
            'event_density': np.random.uniform(0, 1, 10),
            'engagement_index': np.random.uniform(0, 1, 10),
            'fear_index': np.random.uniform(0, 1, 10),
            'overconfidence_index': np.random.uniform(0, 1, 10)
        })
        
        result = self.phase4.calculate_bri(behavioral_features)
        
        self.assertEqual(len(result), 10)
        self.assertIn('date', result.columns)
        self.assertIn('bri', result.columns)
        self.assertIn('bri_normalized', result.columns)
        
        # Check BRI normalization
        self.assertTrue(result['bri_normalized'].min() >= 0)
        self.assertTrue(result['bri_normalized'].max() <= 100)
    
    def test_calculate_bri_no_features(self):
        """Test BRI calculation with no features"""
        behavioral_features = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5)
        })
        
        result = self.phase4.calculate_bri(behavioral_features)
        self.assertTrue(result.empty)
    
    def test_normalize_features(self):
        """Test feature normalization"""
        features = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        result = self.phase4._normalize_features(features)
        
        # Check normalization
        self.assertTrue(result['feature1'].min() >= 0)
        self.assertTrue(result['feature1'].max() <= 1)
        self.assertTrue(result['feature2'].min() >= 0)
        self.assertTrue(result['feature2'].max() <= 1)
    
    def test_get_feature_weights(self):
        """Test feature weight calculation"""
        features = ['sentiment_volatility', 'news_tone', 'media_herding']
        weights = self.phase4._get_feature_weights(features)
        
        self.assertEqual(len(weights), 3)
        self.assertIn('sentiment_volatility', weights)
        self.assertIn('news_tone', weights)
        self.assertIn('media_herding', weights)
        
        # Check that weights sum to approximately 1
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=1)
    
    def test_calculate_weighted_bri(self):
        """Test weighted BRI calculation"""
        normalized_features = pd.DataFrame({
            'feature1': [0.2, 0.4, 0.6, 0.8, 1.0],
            'feature2': [0.1, 0.3, 0.5, 0.7, 0.9]
        })
        
        weights = {'feature1': 0.6, 'feature2': 0.4}
        
        result = self.phase4._calculate_weighted_bri(normalized_features, weights)
        
        self.assertEqual(len(result), 5)
        self.assertTrue(result.min() >= 0)
        self.assertTrue(result.max() <= 1)
    
    def test_normalize_bri_to_0_100(self):
        """Test BRI normalization to 0-100 scale"""
        bri_scores = pd.Series([-2, -1, 0, 1, 2])
        result = self.phase4._normalize_bri_to_0_100(bri_scores)
        
        self.assertTrue(result.min() >= 0)
        self.assertTrue(result.max() <= 100)
        self.assertEqual(len(result), 5)

class TestPhase5AnalysisValidation(unittest.TestCase):
    """Test Phase 5: Analysis & Validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_data_collector = Mock()
        self.phase5 = Phase5AnalysisValidation(self.mock_data_collector)
    
    def test_collect_vix_data(self):
        """Test VIX data collection"""
        mock_vix_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'vix': np.random.uniform(10, 50, 10)
        })
        
        self.mock_data_collector.collect_vix_data.return_value = mock_vix_data
        
        result = self.phase5.collect_vix_data('2024-01-01', '2024-01-10')
        
        self.assertEqual(len(result), 10)
        self.assertIn('date', result.columns)
        self.assertIn('vix', result.columns)
        self.mock_data_collector.collect_vix_data.assert_called_once_with('2024-01-01', '2024-01-10')
    
    def test_run_validation_analysis(self):
        """Test validation analysis"""
        bri_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'bri': np.random.uniform(0, 100, 10)
        })
        
        vix_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'vix': np.random.uniform(10, 50, 10)
        })
        
        market_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'close': np.random.uniform(1000, 2000, 10),
            'volume': np.random.randint(1000000, 10000000, 10)
        })
        
        result = self.phase5.run_validation_analysis(bri_data, vix_data, market_data)
        
        self.assertIn('correlations', result)
        self.assertIn('lag_analysis', result)
        self.assertIn('performance_metrics', result)
        self.assertIn('data_points', result)
        
        # Check correlations
        correlations = result['correlations']
        self.assertIn('bri_vix', correlations)
        self.assertIn('bri_market', correlations)
    
    def test_run_economic_backtesting(self):
        """Test economic backtesting"""
        bri_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'bri': np.random.uniform(0, 100, 100)
        })
        
        market_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'close': np.random.uniform(1000, 2000, 100),
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        
        result = self.phase5.run_economic_backtesting(bri_data, market_data)
        
        self.assertIn('events_analyzed', result)
        self.assertIn('event_results', result)
        self.assertIsInstance(result['event_results'], list)
    
    def test_merge_validation_data(self):
        """Test validation data merging"""
        bri_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'bri': [50, 60, 70, 80, 90]
        })
        
        vix_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'vix': [20, 25, 30, 35, 40]
        })
        
        market_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'close': [1000, 1100, 1200, 1300, 1400],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        result = self.phase5._merge_validation_data(bri_data, vix_data, market_data)
        
        self.assertEqual(len(result), 5)
        self.assertIn('bri', result.columns)
        self.assertIn('vix', result.columns)
        self.assertIn('close', result.columns)
        self.assertIn('volume', result.columns)
    
    def test_calculate_correlations(self):
        """Test correlation calculation"""
        data = pd.DataFrame({
            'bri': [50, 60, 70, 80, 90],
            'vix': [20, 25, 30, 35, 40],
            'close': [1000, 1100, 1200, 1300, 1400]
        })
        
        result = self.phase5._calculate_correlations(data)
        
        self.assertIn('bri_vix', result)
        self.assertIn('bri_market', result)
        self.assertIsInstance(result['bri_vix'], float)
        self.assertIsInstance(result['bri_market'], float)
    
    def test_calculate_lag_analysis(self):
        """Test lag analysis calculation"""
        data = pd.DataFrame({
            'bri': np.random.uniform(0, 100, 20),
            'vix': np.random.uniform(10, 50, 20)
        })
        
        result = self.phase5._calculate_lag_analysis(data)
        
        self.assertIn('lag_correlations', result)
        self.assertIn('best_lag', result)
        self.assertIn('best_correlation', result)
        
        # Check lag correlations
        lag_correlations = result['lag_correlations']
        self.assertIsInstance(lag_correlations, dict)
        self.assertIn(0, lag_correlations)  # Should have lag 0
    
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation"""
        data = pd.DataFrame({
            'bri': [50, 60, 70, 80, 90],
            'vix': [20, 25, 30, 35, 40]
        })
        
        result = self.phase5._calculate_performance_metrics(data)
        
        self.assertIn('bri_mean', result)
        self.assertIn('bri_std', result)
        self.assertIn('bri_min', result)
        self.assertIn('bri_max', result)
        self.assertIn('vix_mean', result)
        self.assertIn('vix_std', result)
        
        # Check that metrics are reasonable
        self.assertGreater(result['bri_mean'], 0)
        self.assertGreater(result['bri_std'], 0)
        self.assertGreater(result['vix_mean'], 0)
        self.assertGreater(result['vix_std'], 0)

class TestPhase6Visualization(unittest.TestCase):
    """Test Phase 6: Visualization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.phase6 = Phase6Visualization()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_create_visualizations(self):
        """Test visualization creation"""
        bri_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'bri': np.random.uniform(0, 100, 10)
        })
        
        vix_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'vix': np.random.uniform(10, 50, 10)
        })
        
        market_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'close': np.random.uniform(1000, 2000, 10),
            'volume': np.random.randint(1000000, 10000000, 10)
        })
        
        # Mock matplotlib to avoid actual plot generation
        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.plot'), \
             patch('matplotlib.pyplot.scatter'), \
             patch('matplotlib.pyplot.subplots'), \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'):
            
            self.phase6.create_visualizations(bri_data, vix_data, market_data, self.temp_dir)
            
            # Check that charts directory was created
            charts_dir = os.path.join(self.temp_dir, 'charts')
            self.assertTrue(os.path.exists(charts_dir))
    
    def test_create_bri_timeseries_chart(self):
        """Test BRI time series chart creation"""
        bri_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'bri': [50, 60, 70, 80, 90]
        })
        
        charts_dir = os.path.join(self.temp_dir, 'charts')
        os.makedirs(charts_dir, exist_ok=True)
        
        # Mock matplotlib
        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.plot'), \
             patch('matplotlib.pyplot.title'), \
             patch('matplotlib.pyplot.xlabel'), \
             patch('matplotlib.pyplot.ylabel'), \
             patch('matplotlib.pyplot.grid'), \
             patch('matplotlib.pyplot.xticks'), \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'):
            
            self.phase6._create_bri_timeseries_chart(bri_data, charts_dir)
            
            # Check that the method completed without error
            self.assertTrue(True)
    
    def test_create_correlation_chart(self):
        """Test correlation chart creation"""
        bri_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'bri': [50, 60, 70, 80, 90]
        })
        
        vix_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'vix': [20, 25, 30, 35, 40]
        })
        
        charts_dir = os.path.join(self.temp_dir, 'charts')
        os.makedirs(charts_dir, exist_ok=True)
        
        # Mock matplotlib
        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.scatter'), \
             patch('matplotlib.pyplot.xlabel'), \
             patch('matplotlib.pyplot.ylabel'), \
             patch('matplotlib.pyplot.title'), \
             patch('matplotlib.pyplot.text'), \
             patch('matplotlib.pyplot.grid'), \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'):
            
            self.phase6._create_correlation_chart(bri_data, vix_data, charts_dir)
            
            # Check that the method completed without error
            self.assertTrue(True)

class TestPhase7FinalDeliverables(unittest.TestCase):
    """Test Phase 7: Final Deliverables"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.phase7 = Phase7FinalDeliverables()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_all_deliverables(self):
        """Test saving all deliverables"""
        deliverables = {
            'bri_timeseries': pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=5),
                'bri': [50, 60, 70, 80, 90]
            }),
            'validation_results': {'correlation': 0.5, 'data_points': 100},
            'backtest_results': {'events_analyzed': 5, 'event_results': []}
        }
        
        self.phase7.save_all_deliverables(self.temp_dir, deliverables)
        
        # Check that files were created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'bri_timeseries.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'validation_results.json')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'backtest_results.json')))
    
    def test_generate_final_report(self):
        """Test final report generation"""
        validation_results = {
            'data_points': 100,
            'correlations': {'bri_vix': 0.5, 'bri_market': 0.3}
        }
        
        backtest_results = {
            'events_analyzed': 5,
            'event_results': [
                {'event': 'Test Event', 'date': '2024-01-01', 'type': 'crisis', 'bri_change': 10}
            ]
        }
        
        self.phase7.generate_final_report(self.temp_dir, validation_results, backtest_results)
        
        # Check that report was created
        report_path = os.path.join(self.temp_dir, 'final_report.md')
        self.assertTrue(os.path.exists(report_path))
        
        # Check report content
        with open(report_path, 'r') as f:
            content = f.read()
            self.assertIn('Behavioral Risk Index (BRI) Final Report', content)
            self.assertIn('Validation Results', content)
            self.assertIn('Backtest Results', content)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_complete_pipeline_integration(self):
        """Test complete pipeline integration"""
        # This would be a more comprehensive integration test
        # that tests the entire pipeline flow
        pass

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
