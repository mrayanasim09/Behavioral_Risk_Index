"""
Text preprocessing module for BRI pipeline.
Handles cleaning, tokenization, and normalization of text data.
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

class TextPreprocessor:
    """Text preprocessing class for BRI pipeline."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Finance-specific stop words to keep
        self.finance_terms = {
            'market', 'markets', 'stock', 'stocks', 'price', 'prices', 'trading', 'trade',
            'investor', 'investors', 'investment', 'investments', 'financial', 'finance',
            'economy', 'economic', 'recession', 'inflation', 'fed', 'federal', 'rate', 'rates',
            'earnings', 'revenue', 'profit', 'loss', 'gain', 'gains', 'volatility', 'volatile',
            'bull', 'bear', 'bullish', 'bearish', 'rally', 'crash', 'correction', 'bubble',
            'bond', 'bonds', 'yield', 'yields', 'treasury', 'dollar', 'currency', 'currencies',
            'sector', 'sectors', 'industry', 'industries', 'company', 'companies', 'corporate',
            'bank', 'banks', 'banking', 'credit', 'debt', 'equity', 'equities', 'derivatives',
            'hedge', 'hedging', 'portfolio', 'portfolios', 'asset', 'assets', 'liability',
            'liabilities', 'balance', 'sheet', 'income', 'statement', 'cash', 'flow', 'flows',
            'dividend', 'dividends', 'share', 'shares', 'shareholder', 'shareholders',
            'ipo', 'merger', 'acquisition', 'acquisitions', 'takeover', 'takeovers',
            'analyst', 'analysts', 'forecast', 'forecasts', 'prediction', 'predictions',
            'risk', 'risks', 'risky', 'safe', 'safety', 'secure', 'security', 'securities',
            'fund', 'funds', 'funding', 'capital', 'liquidity', 'liquid', 'illiquid',
            'leverage', 'leveraged', 'margin', 'margins', 'option', 'options', 'futures',
            'commodity', 'commodities', 'gold', 'silver', 'oil', 'gas', 'energy', 'utilities',
            'technology', 'tech', 'biotech', 'pharma', 'healthcare', 'retail', 'consumer',
            'industrial', 'materials', 'real', 'estate', 'reit', 'reits', 'infrastructure'
        }
        
        # Remove finance terms from stop words
        self.stop_words = self.stop_words - self.finance_terms
        
        # Initialize spaCy model (if available)
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except OSError:
            self.logger.warning("spaCy model not found, using NLTK only")
            self.nlp = None
            self.use_spacy = False
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove user mentions and hashtags (for social media)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if not text:
            return []
        
        if self.use_spacy:
            # Use spaCy for better tokenization
            doc = self.nlp(text)
            tokens = [token.text for token in doc if not token.is_space and not token.is_punct]
        else:
            # Use NLTK tokenization
            tokens = word_tokenize(text)
            # Remove punctuation
            tokens = [token for token in tokens if token not in string.punctuation]
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stop words while keeping finance-specific terms."""
        return [token for token in tokens if token not in self.stop_words and len(token) > 2]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens to their root forms."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_text(self, text: str, lemmatize: bool = True) -> str:
        """Complete text preprocessing pipeline."""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize_text(cleaned_text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Lemmatize if requested
        if lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        # Join tokens back into text
        return ' '.join(tokens)
    
    def preprocess_df(self, df: pd.DataFrame, text_col: str = 'text', 
                     output_col: str = 'processed_text', lemmatize: bool = True) -> pd.DataFrame:
        """Preprocess text column in DataFrame."""
        self.logger.info(f"Preprocessing {len(df)} texts from column '{text_col}'")
        
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Apply preprocessing
        df_processed[output_col] = df_processed[text_col].apply(
            lambda x: self.preprocess_text(x, lemmatize=lemmatize)
        )
        
        # Remove empty processed texts
        df_processed = df_processed[df_processed[output_col].str.len() > 0]
        
        self.logger.info(f"Preprocessing completed. {len(df_processed)} texts remaining.")
        
        return df_processed
    
    def extract_ngrams(self, text: str, n: int = 2) -> List[str]:
        """Extract n-grams from text."""
        tokens = self.tokenize_text(text)
        tokens = self.remove_stopwords(tokens)
        
        if len(tokens) < n:
            return []
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    def get_vocabulary_stats(self, texts: List[str]) -> Dict:
        """Get vocabulary statistics from a list of texts."""
        all_tokens = []
        for text in texts:
            tokens = self.tokenize_text(text)
            tokens = self.remove_stopwords(tokens)
            all_tokens.extend(tokens)
        
        # Count token frequencies
        token_counts = pd.Series(all_tokens).value_counts()
        
        return {
            'total_tokens': len(all_tokens),
            'unique_tokens': len(token_counts),
            'vocabulary_size': len(token_counts),
            'most_common': token_counts.head(20).to_dict(),
            'avg_tokens_per_text': len(all_tokens) / len(texts) if texts else 0
        }
    
    def filter_by_length(self, df: pd.DataFrame, text_col: str, 
                        min_length: int = 10, max_length: int = 1000) -> pd.DataFrame:
        """Filter texts by length."""
        mask = (df[text_col].str.len() >= min_length) & (df[text_col].str.len() <= max_length)
        return df[mask]
    
    def remove_duplicates(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Remove duplicate texts."""
        return df.drop_duplicates(subset=[text_col])
    
    def create_daily_corpus(self, df: pd.DataFrame, date_col: str = 'date', 
                           text_col: str = 'processed_text') -> Dict[str, List[str]]:
        """Create daily corpus from processed texts."""
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        daily_corpus = {}
        for date, group in df.groupby(df[date_col].dt.date):
            texts = group[text_col].tolist()
            daily_corpus[date] = texts
        
        return daily_corpus
    
    def preprocess_news_data(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess news data specifically."""
        self.logger.info("Preprocessing news data")
        
        # Combine headline and text if available
        if 'text' in news_df.columns:
            news_df['combined_text'] = news_df['headline'].fillna('') + ' ' + news_df['text'].fillna('')
        else:
            news_df['combined_text'] = news_df['headline']
        
        # Preprocess combined text
        news_df = self.preprocess_df(news_df, 'combined_text', 'processed_text')
        
        # Filter by length
        news_df = self.filter_by_length(news_df, 'processed_text', min_length=20)
        
        # Remove duplicates
        news_df = self.remove_duplicates(news_df, 'processed_text')
        
        return news_df
    
    def preprocess_social_data(self, social_df: pd.DataFrame, text_col: str = 'combined_text') -> pd.DataFrame:
        """Preprocess social media data specifically."""
        self.logger.info("Preprocessing social media data")
        
        # Preprocess text
        social_df = self.preprocess_df(social_df, text_col, 'processed_text')
        
        # Filter by length (social media can be shorter)
        social_df = self.filter_by_length(social_df, 'processed_text', min_length=10)
        
        # Remove duplicates
        social_df = self.remove_duplicates(social_df, 'processed_text')
        
        return social_df
