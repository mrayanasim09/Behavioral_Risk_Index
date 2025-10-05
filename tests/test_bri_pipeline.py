"""
Unit tests for BRI pipeline components.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocess import TextPreprocessor
from vectorize import Vectorizer
from bri import BRICalculator
from validation import ValidationEngine
from utils import calculate_returns, calculate_realized_volatility, safe_divide

class TestTextPreprocessing:
    """Test text preprocessing functionality."""
    
    def setup_method(self):
        """Setup test data."""
        self.preprocessor = TextPreprocessor()
        self.sample_texts = [
            "Market volatility spikes as investors react to economic uncertainty",
            "Fed signals potential rate hikes amid inflation concerns",
            "Stocks tumble on recession fears and geopolitical tensions"
        ]
    
    def test_clean_text(self):
        """Test text cleaning."""
        dirty_text = "Market volatility spikes! @investor #stocks https://example.com"
        cleaned = self.preprocessor.clean_text(dirty_text)
        assert "@investor" not in cleaned
        assert "#stocks" not in cleaned
        assert "https://example.com" not in cleaned
        assert "!" not in cleaned
    
    def test_tokenize_text(self):
        """Test text tokenization."""
        text = "Market volatility spikes as investors react"
        tokens = self.preprocessor.tokenize_text(text)
        assert isinstance(tokens, list)
        assert "market" in tokens
        assert "volatility" in tokens
    
    def test_remove_stopwords(self):
        """Test stopword removal."""
        tokens = ["market", "the", "volatility", "and", "investors"]
        filtered = self.preprocessor.remove_stopwords(tokens)
        assert "the" not in filtered
        assert "and" not in filtered
        assert "market" in filtered  # Finance term should be kept
    
    def test_preprocess_text(self):
        """Test complete preprocessing pipeline."""
        text = "Market volatility spikes! @investor #stocks"
        processed = self.preprocessor.preprocess_text(text)
        assert isinstance(processed, str)
        assert len(processed) > 0
        assert "@investor" not in processed
        assert "#stocks" not in processed
    
    def test_preprocess_df(self):
        """Test DataFrame preprocessing."""
        df = pd.DataFrame({
            'text': self.sample_texts,
            'date': pd.date_range('2023-01-01', periods=3)
        })
        
        processed_df = self.preprocessor.preprocess_df(df, 'text')
        
        assert 'processed_text' in processed_df.columns
        assert len(processed_df) == len(df)
        assert all(isinstance(text, str) for text in processed_df['processed_text'])

class TestVectorization:
    """Test vectorization functionality."""
    
    def setup_method(self):
        """Setup test data."""
        self.vectorizer = Vectorizer()
        self.sample_texts = [
            "market volatility spikes investors react economic uncertainty",
            "fed signals potential rate hikes inflation concerns",
            "stocks tumble recession fears geopolitical tensions"
        ]
    
    def test_build_tfidf(self):
        """Test TF-IDF vectorizer building."""
        vectorizer = self.vectorizer.build_tfidf(self.sample_texts)
        assert vectorizer is not None
        assert hasattr(vectorizer, 'vocabulary_')
        assert len(vectorizer.vocabulary_) > 0
    
    def test_transform_tfidf(self):
        """Test TF-IDF transformation."""
        self.vectorizer.build_tfidf(self.sample_texts)
        vectors = self.vectorizer.transform_tfidf(self.sample_texts)
        
        assert vectors.shape[0] == len(self.sample_texts)
        assert vectors.shape[1] > 0
        assert isinstance(vectors, np.ndarray)
    
    def test_embed_documents(self):
        """Test document embedding."""
        embeddings = self.vectorizer.embed_documents(self.sample_texts)
        
        assert embeddings.shape[0] == len(self.sample_texts)
        assert embeddings.shape[1] > 0
        assert isinstance(embeddings, np.ndarray)

class TestBRICalculation:
    """Test BRI calculation functionality."""
    
    def setup_method(self):
        """Setup test data."""
        self.bri_calculator = BRICalculator()
        
        # Create test daily corpus
        self.daily_corpus = {
            pd.Timestamp('2023-01-01'): [
                "market volatility spikes investors react economic uncertainty",
                "market volatility spikes investors react economic uncertainty",  # Duplicate
                "fed signals potential rate hikes inflation concerns"
            ],
            pd.Timestamp('2023-01-02'): [
                "stocks tumble recession fears geopolitical tensions",
                "tech stocks rally strong earnings ai optimism",
                "energy sector gains oil price recovery"
            ]
        }
    
    def test_compute_bri_entropy(self):
        """Test entropy-based BRI calculation."""
        bri_df = self.bri_calculator.compute_bri_entropy(self.daily_corpus)
        
        assert isinstance(bri_df, pd.DataFrame)
        assert len(bri_df) == 2
        assert 'date' in bri_df.columns
        assert 'BRI_t' in bri_df.columns
        assert 'H_t' in bri_df.columns
        assert 'H_norm' in bri_df.columns
        
        # Check that BRI values are in expected range
        assert all(0 <= bri <= 100 for bri in bri_df['BRI_t'])
        
        # First day should have higher BRI due to duplicate text
        assert bri_df.iloc[0]['BRI_t'] > bri_df.iloc[1]['BRI_t']
    
    def test_compute_bri_clusters(self):
        """Test cluster-based BRI calculation."""
        # Create test vectors
        daily_vectors = {
            pd.Timestamp('2023-01-01'): np.random.rand(3, 10),
            pd.Timestamp('2023-01-02'): np.random.rand(3, 10)
        }
        
        bri_df = self.bri_calculator.compute_bri_clusters(daily_vectors)
        
        assert isinstance(bri_df, pd.DataFrame)
        assert len(bri_df) == 2
        assert 'date' in bri_df.columns
        assert 'BRI_t' in bri_df.columns
        assert 'HHI' in bri_df.columns
        
        # Check that BRI values are in expected range
        assert all(0 <= bri <= 100 for bri in bri_df['BRI_t'])
    
    def test_normalize_bri(self):
        """Test BRI normalization."""
        bri_df = pd.DataFrame({
            'BRI_t': [10, 20, 30, 40, 50]
        })
        
        normalized_df = self.bri_calculator.normalize_bri(bri_df, method='zscore')
        
        assert 'BRI_normalized' in normalized_df.columns
        assert normalized_df['BRI_normalized'].mean() == pytest.approx(0, abs=1e-10)
        assert normalized_df['BRI_normalized'].std() == pytest.approx(1, abs=1e-10)

class TestValidation:
    """Test validation functionality."""
    
    def setup_method(self):
        """Setup test data."""
        self.validation_engine = ValidationEngine()
        
        # Create test data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        self.bri_data = pd.DataFrame({
            'date': dates,
            'BRI_t': np.random.randn(100).cumsum() + 50
        })
        
        self.market_data = pd.DataFrame({
            'date': dates,
            '^VIX_Close': np.random.randn(100).cumsum() + 20,
            '^GSPC_Close': np.random.randn(100).cumsum() + 4000,
            'returns': np.random.randn(100) * 0.02,
            'realized_vol': np.abs(np.random.randn(100)) * 0.02
        })
    
    def test_prepare_validation_data(self):
        """Test validation data preparation."""
        validation_data = self.validation_engine.prepare_validation_data(
            self.bri_data, self.market_data
        )
        
        assert isinstance(validation_data, pd.DataFrame)
        assert len(validation_data) > 0
        assert 'BRI_t' in validation_data.columns
        assert '^VIX_Close' in validation_data.columns
        assert 'BRI_lag1' in validation_data.columns
    
    def test_compute_correlations(self):
        """Test correlation computation."""
        validation_data = self.validation_engine.prepare_validation_data(
            self.bri_data, self.market_data
        )
        
        corr_matrix = self.validation_engine.compute_correlations(validation_data)
        
        assert isinstance(corr_matrix, pd.DataFrame)
        assert not corr_matrix.empty
        assert corr_matrix.shape[0] == corr_matrix.shape[1]  # Square matrix
    
    def test_run_ols_regression(self):
        """Test OLS regression."""
        validation_data = self.validation_engine.prepare_validation_data(
            self.bri_data, self.market_data
        )
        
        results = self.validation_engine.run_ols_regression(
            validation_data, 'realized_vol', ['^VIX_Close', 'BRI_lag1']
        )
        
        assert isinstance(results, dict)
        assert 'r_squared' in results
        assert 'coefficients' in results
        assert 'p_values' in results

class TestUtils:
    """Test utility functions."""
    
    def test_safe_divide(self):
        """Test safe division function."""
        assert safe_divide(10, 2) == 5
        assert safe_divide(10, 0) == 0
        assert safe_divide(10, 0, default=1) == 1
    
    def test_calculate_returns(self):
        """Test returns calculation."""
        prices = pd.Series([100, 105, 110, 108, 112])
        returns = calculate_returns(prices)
        
        assert len(returns) == len(prices)
        assert pd.isna(returns.iloc[0])  # First value should be NaN
        assert not pd.isna(returns.iloc[1:]).any()  # Rest should be valid
    
    def test_calculate_realized_volatility(self):
        """Test realized volatility calculation."""
        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02])
        rv = calculate_realized_volatility(returns, window=3)
        
        assert len(rv) == len(returns)
        assert pd.isna(rv.iloc[:2]).all()  # First two should be NaN
        assert not pd.isna(rv.iloc[2:]).any()  # Rest should be valid

class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline execution."""
        # This would be a more comprehensive test
        # For now, just test that components can be imported and initialized
        from data_collect import DataCollector
        from preprocess import TextPreprocessor
        from vectorize import Vectorizer
        from bri import BRICalculator
        from validation import ValidationEngine
        
        # Test initialization
        data_collector = DataCollector()
        preprocessor = TextPreprocessor()
        vectorizer = Vectorizer()
        bri_calculator = BRICalculator()
        validation_engine = ValidationEngine()
        
        assert data_collector is not None
        assert preprocessor is not None
        assert vectorizer is not None
        assert bri_calculator is not None
        assert validation_engine is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
