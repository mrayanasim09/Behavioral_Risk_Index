"""
Vectorization module for BRI pipeline.
Implements TF-IDF and FinBERT embeddings for text vectorization.
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import torch
from sentence_transformers import SentenceTransformer
import logging

from utils import ensure_directory

class Vectorizer:
    """Text vectorization class for BRI pipeline."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.tfidf_vectorizer = None
        self.finbert_model = None
        self.lda_model = None
        self.kmeans_model = None
        
        # Check for GPU availability
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize FinBERT model
        self._load_finbert_model()
    
    def _load_finbert_model(self):
        """Load FinBERT model for financial text embeddings."""
        try:
            # Try to load a financial sentiment model
            model_name = "yiyanghkust/finbert-tone"
            self.finbert_model = SentenceTransformer(model_name, device=self.device)
            self.logger.info(f"Loaded FinBERT model: {model_name}")
        except Exception as e:
            try:
                # Fallback to general sentence transformer
                model_name = "all-MiniLM-L6-v2"
                self.finbert_model = SentenceTransformer(model_name, device=self.device)
                self.logger.warning(f"FinBERT not available, using {model_name}: {e}")
            except Exception as e2:
                self.logger.error(f"Failed to load any sentence transformer model: {e2}")
                self.finbert_model = None
    
    def build_tfidf(self, corpus: List[str], max_features: int = 20000, 
                   ngram_range: Tuple[int, int] = (1, 2), min_df: int = 2) -> TfidfVectorizer:
        """Build TF-IDF vectorizer from corpus."""
        self.logger.info(f"Building TF-IDF vectorizer with {len(corpus)} documents")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=0.95,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Fit on corpus
        self.tfidf_vectorizer.fit(corpus)
        
        self.logger.info(f"TF-IDF vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        
        return self.tfidf_vectorizer
    
    def transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """Transform texts using TF-IDF vectorizer."""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call build_tfidf first.")
        
        return self.tfidf_vectorizer.transform(texts).toarray()
    
    def embed_documents(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings using FinBERT model."""
        if self.finbert_model is None:
            self.logger.warning("FinBERT model not available, falling back to TF-IDF")
            if self.tfidf_vectorizer is None:
                raise ValueError("Neither FinBERT nor TF-IDF available")
            return self.transform_tfidf(texts)
        
        self.logger.info(f"Generating embeddings for {len(texts)} documents")
        
        # Process in batches to avoid memory issues
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.finbert_model.encode(
                batch_texts, 
                convert_to_tensor=False,
                show_progress_bar=True
            )
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def fit_lda(self, texts: List[str], n_topics: int = 10, 
                random_state: int = 42) -> LatentDirichletAllocation:
        """Fit LDA topic model on texts."""
        if self.tfidf_vectorizer is None:
            self.logger.info("Building TF-IDF vectorizer for LDA")
            self.build_tfidf(texts)
        
        # Transform texts to TF-IDF
        tfidf_matrix = self.transform_tfidf(texts)
        
        # Fit LDA
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            max_iter=100,
            learning_method='online'
        )
        
        self.lda_model.fit(tfidf_matrix)
        
        self.logger.info(f"LDA model fitted with {n_topics} topics")
        
        return self.lda_model
    
    def get_topic_distributions(self, texts: List[str]) -> np.ndarray:
        """Get topic distributions for texts using LDA."""
        if self.lda_model is None:
            raise ValueError("LDA model not fitted. Call fit_lda first.")
        
        tfidf_matrix = self.transform_tfidf(texts)
        return self.lda_model.transform(tfidf_matrix)
    
    def fit_kmeans(self, embeddings: np.ndarray, n_clusters: int = 10, 
                  random_state: int = 42) -> KMeans:
        """Fit K-means clustering on embeddings."""
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )
        
        self.kmeans_model.fit(embeddings)
        
        self.logger.info(f"K-means model fitted with {n_clusters} clusters")
        
        return self.kmeans_model
    
    def get_cluster_labels(self, embeddings: np.ndarray) -> np.ndarray:
        """Get cluster labels for embeddings."""
        if self.kmeans_model is None:
            raise ValueError("K-means model not fitted. Call fit_kmeans first.")
        
        return self.kmeans_model.predict(embeddings)
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers."""
        if self.kmeans_model is None:
            raise ValueError("K-means model not fitted.")
        
        return self.kmeans_model.cluster_centers_
    
    def vectorize_daily_corpus(self, daily_corpus: Dict[str, List[str]], 
                             method: str = 'tfidf') -> Dict[str, np.ndarray]:
        """Vectorize daily corpus using specified method."""
        self.logger.info(f"Vectorizing daily corpus using {method}")
        
        daily_vectors = {}
        
        for date, texts in daily_corpus.items():
            if not texts:
                daily_vectors[date] = np.array([])
                continue
            
            if method == 'tfidf':
                if self.tfidf_vectorizer is None:
                    # Fit on all texts first
                    all_texts = [text for texts_list in daily_corpus.values() for text in texts_list]
                    self.build_tfidf(all_texts)
                
                vectors = self.transform_tfidf(texts)
                daily_vectors[date] = vectors
            
            elif method == 'finbert':
                vectors = self.embed_documents(texts)
                daily_vectors[date] = vectors
            
            else:
                raise ValueError(f"Unknown vectorization method: {method}")
        
        return daily_vectors
    
    def save_models(self, output_dir: str = "models"):
        """Save trained models to disk."""
        ensure_directory(output_dir)
        
        if self.tfidf_vectorizer is not None:
            with open(os.path.join(output_dir, "tfidf.pkl"), 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            self.logger.info("Saved TF-IDF vectorizer")
        
        if self.lda_model is not None:
            with open(os.path.join(output_dir, "lda.pkl"), 'wb') as f:
                pickle.dump(self.lda_model, f)
            self.logger.info("Saved LDA model")
        
        if self.kmeans_model is not None:
            with open(os.path.join(output_dir, "kmeans.pkl"), 'wb') as f:
                pickle.dump(self.kmeans_model, f)
            self.logger.info("Saved K-means model")
    
    def load_models(self, model_dir: str = "models"):
        """Load trained models from disk."""
        if os.path.exists(os.path.join(model_dir, "tfidf.pkl")):
            with open(os.path.join(model_dir, "tfidf.pkl"), 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            self.logger.info("Loaded TF-IDF vectorizer")
        
        if os.path.exists(os.path.join(model_dir, "lda.pkl")):
            with open(os.path.join(model_dir, "lda.pkl"), 'rb') as f:
                self.lda_model = pickle.load(f)
            self.logger.info("Loaded LDA model")
        
        if os.path.exists(os.path.join(model_dir, "kmeans.pkl")):
            with open(os.path.join(model_dir, "kmeans.pkl"), 'rb') as f:
                self.kmeans_model = pickle.load(f)
            self.logger.info("Loaded K-means model")
    
    def get_feature_names(self) -> List[str]:
        """Get feature names from TF-IDF vectorizer."""
        if self.tfidf_vectorizer is None:
            return []
        return self.tfidf_vectorizer.get_feature_names_out().tolist()
    
    def get_top_words_per_topic(self, n_words: int = 10) -> Dict[int, List[str]]:
        """Get top words for each LDA topic."""
        if self.lda_model is None or self.tfidf_vectorizer is None:
            return {}
        
        feature_names = self.get_feature_names()
        topics = {}
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics[topic_idx] = top_words
        
        return topics
