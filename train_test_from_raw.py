#!/usr/bin/env python3
"""
Train/Test from raw CSVs with strict no-leakage controls.
Train: 2020-01-01 → 2025-12-31
Test:  2017-01-01 → 2018-12-31

Pipeline:
- Load raw CSVs from data/raw
- Phase2 (clean + sentiment) and Phase3 (features) per split
- Normalize features using scalers fit ONLY on training
- Learn feature weights via PCA on training only
- Compute BRI for train and test using learned weights; scale to 0–100 using training only
- Validate against VIX (collected live) without affecting learned parameters
"""

import os
import sys
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime, date

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

sys.path.append('src')
from pipeline_phases import Phase2DataPreprocessing, Phase3FeatureEngineering, Phase5AnalysisValidation, Phase7FinalDeliverables
from preprocess import TextPreprocessor
from data_collect import DataCollector
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train_test_from_raw')


def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def read_csv_with_date(path: str, date_col_candidates: List[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        logger.warning(f"Missing file: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Find date column
    col = next((c for c in date_col_candidates if c in df.columns), None)
    if col is None:
        return df
    # Normalize to tz-naive midnight
    df['date'] = pd.to_datetime(df[col], errors='coerce', utc=True).dt.tz_convert(None).dt.normalize()
    return df


def filter_window(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    if df.empty or 'date' not in df.columns:
        return pd.DataFrame()
    return df[(df['date'] >= pd.Timestamp(start)) & (df['date'] <= pd.Timestamp(end))].copy()


def build_features(config: Dict, news_df: pd.DataFrame, reddit_df: pd.DataFrame) -> pd.DataFrame:
    text_pre = TextPreprocessor(config)
    phase2 = Phase2DataPreprocessing(text_pre)
    phase3 = Phase3FeatureEngineering()

    gdelt_clean = phase2.clean_gdelt_data(news_df) if not news_df.empty else pd.DataFrame()
    reddit_clean = phase2.clean_reddit_text(reddit_df) if not reddit_df.empty else pd.DataFrame()
    sentiment = phase2.perform_sentiment_analysis(reddit_clean, gdelt_clean)
    features = phase3.create_behavioral_features(sentiment, gdelt_clean, reddit_clean)
    return features


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    # Known engineered features; select intersection with df
    candidates = [
        'sentiment_volatility', 'news_tone', 'media_herding', 'polarity_skew',
        'event_density', 'engagement_index', 'fear_index', 'overconfidence_index'
    ]
    return [c for c in candidates if c in df.columns]


@dataclass
class Scalers:
    min_: Dict[str, float]
    max_: Dict[str, float]


def fit_minmax_scalers(df: pd.DataFrame, feature_cols: List[str]) -> Scalers:
    min_ = {}
    max_ = {}
    for c in feature_cols:
        series = df[c].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        min_[c] = float(series.min())
        max_[c] = float(series.max()) if float(series.max()) != float(series.min()) else float(series.min()) + 1e-6
    return Scalers(min_, max_)


def transform_minmax(df: pd.DataFrame, feature_cols: List[str], scalers: Scalers) -> pd.DataFrame:
    out = df.copy()
    for c in feature_cols:
        mn = scalers.min_[c]
        mx = scalers.max_[c]
        out[c] = ((out[c].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0) - mn) / (mx - mn)).clip(0.0, 1.0)
    return out


def learn_pca_weights(df_norm: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    X = df_norm[feature_cols].values
    if X.shape[0] < 2:
        # fallback equal weights
        w = {c: 1.0 / len(feature_cols) for c in feature_cols}
        return w
    pca = PCA(n_components=1, random_state=42)
    pca.fit(X)
    # weights proportional to absolute loadings of PC1
    loadings = np.abs(pca.components_[0])
    loadings = loadings / (loadings.sum() + 1e-12)
    return {c: float(w) for c, w in zip(feature_cols, loadings)}


def weighted_bri(df_norm: pd.DataFrame, feature_cols: List[str], weights: Dict[str, float]) -> pd.Series:
    wvec = np.array([weights[c] for c in feature_cols])
    X = df_norm[feature_cols].values
    bri = X.dot(wvec)
    # scale to 0-100
    return pd.Series(bri * 100.0, index=df_norm.index).clip(0.0, 100.0)


def main():
    config = load_config()
    out_dir = 'output/train_test_from_raw'
    os.makedirs(out_dir, exist_ok=True)

    # Load raw CSVs
    market = read_csv_with_date('data/raw/market_2017-01-01_2025-12-31.csv', ['Date', 'date'])
    news = read_csv_with_date('data/raw/news_2017-01-01_2025-12-31.csv', ['date'])
    reddit = read_csv_with_date('data/raw/reddit_2020-01-01_2024-12-31.csv', ['date'])

    # Define windows
    train_start = date(2020, 1, 1)
    train_end = date(2025, 12, 31)
    test_start = date(2017, 1, 1)
    test_end = date(2018, 12, 31)

    # Filter windows
    market_train = filter_window(market, train_start, train_end)
    market_test = filter_window(market, test_start, test_end)
    news_train = filter_window(news, train_start, train_end)
    news_test = filter_window(news, test_start, test_end)
    reddit_train = filter_window(reddit, train_start, train_end)
    reddit_test = filter_window(reddit, test_start, test_end)

    # Build features per split (no shared state)
    logger.info('Building training features...')
    feats_train = build_features(config, news_train, reddit_train)
    logger.info('Building test features...')
    feats_test = build_features(config, news_test, reddit_test)
    # Ensure test has a date axis; if missing/empty, synthesize from available sources
    if (feats_test is None) or feats_test.empty or ('date' not in feats_test.columns):
        # Prefer news dates, else market dates
        candidate_df = news_test if (not news_test.empty and 'date' in news_test.columns) else market_test
        if candidate_df is not None and not candidate_df.empty and 'date' in candidate_df.columns:
            dates = (
                pd.to_datetime(candidate_df['date'], errors='coerce').dt.normalize().dropna().sort_values().unique()
            )
            feats_test = pd.DataFrame({'date': dates})
        else:
            logger.warning('No basis for test dates; creating empty test features')
            feats_test = pd.DataFrame({'date': pd.to_datetime([])})

    feature_cols = select_feature_columns(feats_train)
    # Ensure test has all training feature columns (fill with zeros if missing)
    for c in feature_cols:
        if c not in feats_test.columns:
            feats_test[c] = 0.0

    # Fit scalers on training only
    scalers = fit_minmax_scalers(feats_train, feature_cols)
    feats_train_norm = transform_minmax(feats_train, feature_cols, scalers)
    feats_test_norm = transform_minmax(feats_test, feature_cols, scalers)

    # Learn weights on training only
    weights = learn_pca_weights(feats_train_norm, feature_cols)

    # Compute BRI
    bri_train = pd.DataFrame({
        'date': feats_train_norm['date'],
        'bri': weighted_bri(feats_train_norm, feature_cols, weights)
    })
    bri_test = pd.DataFrame({
        'date': feats_test_norm['date'],
        'bri': weighted_bri(feats_test_norm, feature_cols, weights)
    })

    # Validation vs VIX (does not change learned params)
    dc = DataCollector(config)
    phase5 = Phase5AnalysisValidation(dc)
    vix_train = phase5.collect_vix_data(train_start.isoformat(), train_end.isoformat())
    vix_test = phase5.collect_vix_data(test_start.isoformat(), test_end.isoformat())

    val_train = phase5.run_validation_analysis(bri_train, vix_train, market_train)
    val_test = phase5.run_validation_analysis(bri_test, vix_test, market_test)

    # Save artifacts
    os.makedirs(out_dir, exist_ok=True)
    bri_train.to_csv(os.path.join(out_dir, 'bri_train.csv'), index=False)
    bri_test.to_csv(os.path.join(out_dir, 'bri_test.csv'), index=False)
    with open(os.path.join(out_dir, 'weights.json'), 'w') as f:
        json.dump(weights, f, indent=2)
    with open(os.path.join(out_dir, 'scalers.json'), 'w') as f:
        json.dump({'min_': scalers.min_, 'max_': scalers.max_}, f, indent=2)
    with open(os.path.join(out_dir, 'validation_train.json'), 'w') as f:
        json.dump(val_train, f, indent=2, default=str)
    with open(os.path.join(out_dir, 'validation_test.json'), 'w') as f:
        json.dump(val_test, f, indent=2, default=str)

    # Summary
    summary = {
        'train_window': [str(train_start), str(train_end)],
        'test_window': [str(test_start), str(test_end)],
        'feature_columns': feature_cols,
        'train_rows': int(len(bri_train)),
        'test_rows': int(len(bri_test))
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info('✅ Train/Test completed. Artifacts saved in output/train_test_from_raw')


if __name__ == '__main__':
    main()


