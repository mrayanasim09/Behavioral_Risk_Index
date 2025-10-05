"""
Behavioral Risk Index (BRI) calculation module.
Implements entropy-based and cluster-based BRI methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import logging

from utils import safe_divide

class BRICalculator:
    """Main BRI calculation class."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # BRI parameters
        self.entropy_params = self.config.get('entropy', {})
        self.cluster_params = self.config.get('cluster', {})
        
        # Default parameters
        self.min_vocab_size = self.entropy_params.get('min_vocab_size', 10)
        self.max_vocab_size = self.entropy_params.get('max_vocab_size', 50000)
        self.n_clusters = self.cluster_params.get('n_clusters', 10)
        self.random_state = self.cluster_params.get('random_state', 42)
    
    def compute_bri_entropy(self, daily_corpus: Dict[str, List[str]], 
                           vocab_size: Optional[int] = None) -> pd.DataFrame:
        """
        Compute BRI using entropy-based method.
        
        For each day t:
        1. Compute word frequency distribution p_w,t over vocabulary V
        2. Calculate Shannon entropy: H_t = -sum(p_w * log(p_w))
        3. Normalize: H_norm = H_t / log(|V_t|)
        4. Behavioral concentration: C_t = 1 - H_norm
        5. Scale to 0-100: BRI_t = 100 * C_t
        """
        self.logger.info("Computing BRI using entropy method")
        
        bri_data = []
        
        for date, texts in daily_corpus.items():
            if not texts:
                bri_data.append({
                    'date': date,
                    'raw_count': 0,
                    'vocab_size': 0,
                    'H_t': 0.0,
                    'H_norm': 0.0,
                    'BRI_t': 0.0
                })
                continue
            
            # Combine all texts for the day
            combined_text = ' '.join(texts)
            
            # Tokenize and count words
            words = combined_text.lower().split()
            word_counts = pd.Series(words).value_counts()
            
            # Filter by vocabulary size if specified
            if vocab_size and len(word_counts) > vocab_size:
                word_counts = word_counts.head(vocab_size)
            
            # Calculate probabilities
            total_words = word_counts.sum()
            if total_words == 0:
                bri_data.append({
                    'date': date,
                    'raw_count': 0,
                    'vocab_size': 0,
                    'H_t': 0.0,
                    'H_norm': 0.0,
                    'BRI_t': 0.0
                })
                continue
            
            p_w = word_counts / total_words
            
            # Calculate Shannon entropy
            H_t = -np.sum(p_w * np.log(p_w + 1e-10))  # Add small epsilon to avoid log(0)
            
            # Normalize by vocabulary size
            V_t = len(word_counts)
            H_norm = safe_divide(H_t, np.log(V_t), 0.0)
            
            # Calculate behavioral concentration
            C_t = 1.0 - H_norm
            
            # Scale to 0-100
            BRI_t = 100.0 * C_t
            
            bri_data.append({
                'date': date,
                'raw_count': total_words,
                'vocab_size': V_t,
                'H_t': H_t,
                'H_norm': H_norm,
                'BRI_t': BRI_t
            })
        
        bri_df = pd.DataFrame(bri_data)
        bri_df['date'] = pd.to_datetime(bri_df['date'])
        bri_df = bri_df.sort_values('date')
        
        self.logger.info(f"Computed entropy BRI for {len(bri_df)} days")
        
        return bri_df
    
    def compute_bri_clusters(self, daily_vectors: Dict[str, np.ndarray], 
                           n_clusters: Optional[int] = None) -> pd.DataFrame:
        """
        Compute BRI using cluster concentration method.
        
        For each day t:
        1. Compute document embeddings
        2. Apply k-means clustering
        3. Calculate cluster proportions s_k,t
        4. Compute HHI: HHI_t = sum(s_k^2)
        5. Convert to concentration measure and scale to 0-100
        """
        self.logger.info("Computing BRI using cluster method")
        
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        bri_data = []
        
        for date, vectors in daily_vectors.items():
            if vectors.size == 0 or len(vectors) < 2:
                bri_data.append({
                    'date': date,
                    'n_docs': 0,
                    'n_clusters': 0,
                    'HHI': 0.0,
                    'BRI_t': 0.0
                })
                continue
            
            # Apply k-means clustering
            kmeans = KMeans(
                n_clusters=min(n_clusters, len(vectors)),
                random_state=self.random_state,
                n_init=10
            )
            
            try:
                cluster_labels = kmeans.fit_predict(vectors)
                
                # Calculate cluster proportions
                unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                cluster_proportions = counts / len(cluster_labels)
                
                # Calculate Herfindahl-Hirschman Index
                HHI = np.sum(cluster_proportions ** 2)
                
                # Convert to concentration measure (0-1 scale)
                # HHI ranges from 1/k to 1, where k is number of clusters
                k = len(unique_labels)
                HHI_min = 1.0 / k if k > 0 else 0.0
                HHI_max = 1.0
                
                if HHI_max > HHI_min:
                    concentration = (HHI - HHI_min) / (HHI_max - HHI_min)
                else:
                    concentration = 0.0
                
                # Scale to 0-100
                BRI_t = 100.0 * concentration
                
                bri_data.append({
                    'date': date,
                    'n_docs': len(vectors),
                    'n_clusters': k,
                    'HHI': HHI,
                    'BRI_t': BRI_t
                })
                
            except Exception as e:
                self.logger.warning(f"Error clustering vectors for {date}: {e}")
                bri_data.append({
                    'date': date,
                    'n_docs': len(vectors),
                    'n_clusters': 0,
                    'HHI': 0.0,
                    'BRI_t': 0.0
                })
        
        bri_df = pd.DataFrame(bri_data)
        bri_df['date'] = pd.to_datetime(bri_df['date'])
        bri_df = bri_df.sort_values('date')
        
        self.logger.info(f"Computed cluster BRI for {len(bri_df)} days")
        
        return bri_df
    
    def compute_bri_tfidf_entropy(self, daily_corpus: Dict[str, List[str]], 
                                 vectorizer) -> pd.DataFrame:
        """Compute BRI using TF-IDF vectors and entropy method."""
        self.logger.info("Computing BRI using TF-IDF entropy method")
        
        bri_data = []
        
        for date, texts in daily_corpus.items():
            if not texts:
                bri_data.append({
                    'date': date,
                    'raw_count': 0,
                    'vocab_size': 0,
                    'H_t': 0.0,
                    'H_norm': 0.0,
                    'BRI_t': 0.0
                })
                continue
            
            # Transform texts to TF-IDF
            tfidf_matrix = vectorizer.transform(texts).toarray()
            
            # Sum TF-IDF scores across documents for each term
            term_scores = np.sum(tfidf_matrix, axis=0)
            
            # Calculate probabilities
            total_score = np.sum(term_scores)
            if total_score == 0:
                bri_data.append({
                    'date': date,
                    'raw_count': 0,
                    'vocab_size': 0,
                    'H_t': 0.0,
                    'H_norm': 0.0,
                    'BRI_t': 0.0
                })
                continue
            
            p_w = term_scores / total_score
            
            # Calculate Shannon entropy
            H_t = -np.sum(p_w * np.log(p_w + 1e-10))
            
            # Normalize by vocabulary size
            V_t = len(term_scores)
            H_norm = safe_divide(H_t, np.log(V_t), 0.0)
            
            # Calculate behavioral concentration
            C_t = 1.0 - H_norm
            
            # Scale to 0-100
            BRI_t = 100.0 * C_t
            
            bri_data.append({
                'date': date,
                'raw_count': len(texts),
                'vocab_size': V_t,
                'H_t': H_t,
                'H_norm': H_norm,
                'BRI_t': BRI_t
            })
        
        bri_df = pd.DataFrame(bri_data)
        bri_df['date'] = pd.to_datetime(bri_df['date'])
        bri_df = bri_df.sort_values('date')
        
        return bri_df
    
    def compute_rolling_bri(self, bri_df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """Compute rolling average BRI for smoothing."""
        bri_df = bri_df.copy()
        bri_df[f'BRI_rolling_{window}'] = bri_df['BRI_t'].rolling(window=window, min_periods=1).mean()
        return bri_df
    
    def normalize_bri(self, bri_df: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """Normalize BRI values using specified method."""
        bri_df = bri_df.copy()
        
        if method == 'zscore':
            mean_bri = bri_df['BRI_t'].mean()
            std_bri = bri_df['BRI_t'].std()
            bri_df['BRI_normalized'] = (bri_df['BRI_t'] - mean_bri) / std_bri
        
        elif method == 'minmax':
            min_bri = bri_df['BRI_t'].min()
            max_bri = bri_df['BRI_t'].max()
            bri_df['BRI_normalized'] = (bri_df['BRI_t'] - min_bri) / (max_bri - min_bri)
        
        elif method == 'robust':
            median_bri = bri_df['BRI_t'].median()
            mad_bri = np.median(np.abs(bri_df['BRI_t'] - median_bri))
            bri_df['BRI_normalized'] = (bri_df['BRI_t'] - median_bri) / mad_bri
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return bri_df
    
    def detect_bri_spikes(self, bri_df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
        """Detect BRI spikes above threshold standard deviations."""
        bri_df = bri_df.copy()
        
        # Calculate rolling statistics
        rolling_mean = bri_df['BRI_t'].rolling(window=30, min_periods=10).mean()
        rolling_std = bri_df['BRI_t'].rolling(window=30, min_periods=10).std()
        
        # Detect spikes
        bri_df['BRI_spike'] = (bri_df['BRI_t'] - rolling_mean) > (threshold * rolling_std)
        
        return bri_df
    
    def compute_correlation_with_market(self, bri_df: pd.DataFrame, 
                                      market_df: pd.DataFrame) -> Dict[str, float]:
        """Compute correlation between BRI and market variables."""
        # Merge dataframes on date
        merged_df = pd.merge(bri_df, market_df, on='date', how='inner')
        
        correlations = {}
        
        # Calculate correlations with various market variables
        market_cols = ['VIX', 'returns', 'realized_vol', 'volatility']
        
        for col in market_cols:
            if col in merged_df.columns:
                corr = merged_df['BRI_t'].corr(merged_df[col])
                correlations[f'BRI_vs_{col}'] = corr
        
        return correlations
    
    def compute_bri_summary_stats(self, bri_df: pd.DataFrame) -> Dict[str, float]:
        """Compute summary statistics for BRI."""
        return {
            'mean': bri_df['BRI_t'].mean(),
            'std': bri_df['BRI_t'].std(),
            'min': bri_df['BRI_t'].min(),
            'max': bri_df['BRI_t'].max(),
            'median': bri_df['BRI_t'].median(),
            'q25': bri_df['BRI_t'].quantile(0.25),
            'q75': bri_df['BRI_t'].quantile(0.75),
            'skewness': bri_df['BRI_t'].skew(),
            'kurtosis': bri_df['BRI_t'].kurtosis()
        }
