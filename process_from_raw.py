#!/usr/bin/env python3
"""
Process BRI from previously downloaded raw CSVs in data/raw without re-downloading.
"""

import os
import sys
import json
from datetime import datetime
import pandas as pd

sys.path.append('src')
from pipeline_phases import Phase2DataPreprocessing, Phase3FeatureEngineering, Phase4BRICalculation, Phase5AnalysisValidation, Phase7FinalDeliverables
from preprocess import TextPreprocessor
from data_collect import DataCollector

import yaml
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('process_from_raw')


def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def read_market_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        logger.warning(f"Market CSV not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Normalize market date
    date_col = 'Date' if 'Date' in df.columns else ('date' if 'date' in df.columns else None)
    if date_col:
        # Force parse as UTC to avoid mixed tz, then drop tz and normalize
        parsed = pd.to_datetime(df[date_col], errors='coerce', utc=True)
        df['date'] = parsed.dt.tz_convert(None).dt.normalize()
    return df


def read_news_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        logger.warning(f"News CSV not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
    # Normalize expected column names for downstream processing
    if 'AvgTone' not in df.columns and 'tone' in df.columns:
        df['AvgTone'] = df['tone']
    if 'GoldsteinScale' not in df.columns and 'goldstein_scale' in df.columns:
        df['GoldsteinScale'] = df['goldstein_scale']
    return df


def read_reddit_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        logger.warning(f"Reddit CSV not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
    if 'combined_text' not in df.columns and {'title','text'}.issubset(df.columns):
        df['combined_text'] = df['title'] + ' ' + df['text'].fillna('')
    return df


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--market', default='data/raw/market_2017-01-01_2025-12-31.csv')
    ap.add_argument('--news', default='data/raw/news_2017-01-01_2025-12-31.csv')
    ap.add_argument('--reddit', default='data/raw/reddit_2020-01-01_2024-12-31.csv')
    ap.add_argument('--out', default='output/processed_from_raw')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    config = load_config()

    market_df = read_market_csv(args.market)
    news_df = read_news_csv(args.news)
    reddit_df = read_reddit_csv(args.reddit)

    # Derive processing window as intersection of available sources
    available_dates = []
    for df in (market_df, news_df, reddit_df):
        if not df.empty and 'date' in df.columns:
            available_dates.append((df['date'].min(), df['date'].max()))
    if not available_dates:
        logger.error('No raw data available to process.')
        sys.exit(1)
    start = max(d[0] for d in available_dates)
    end = min(d[1] for d in available_dates)
    logger.info(f'Processing window (intersection): {start.date()} → {end.date()}')

    # Trim to intersection
    def trim(df):
        if df.empty: return df
        return df[(df['date'] >= start) & (df['date'] <= end)].copy()

    market_df = trim(market_df)
    news_df = trim(news_df)
    reddit_df = trim(reddit_df)

    # Initialize phases
    text_pre = TextPreprocessor(config)
    phase2 = Phase2DataPreprocessing(text_pre)
    phase3 = Phase3FeatureEngineering()
    phase4 = Phase4BRICalculation()
    phase7 = Phase7FinalDeliverables()

    # DataCollector only for VIX in Phase 5
    dc = DataCollector(config)
    phase5 = Phase5AnalysisValidation(dc)

    # Phase 2: clean and sentiment
    gdelt_clean = phase2.clean_gdelt_data(news_df) if not news_df.empty else pd.DataFrame()
    reddit_clean = phase2.clean_reddit_text(reddit_df) if not reddit_df.empty else pd.DataFrame()
    sentiment_data = phase2.perform_sentiment_analysis(reddit_clean, gdelt_clean)

    # Phase 3: features
    features = phase3.create_behavioral_features(sentiment_data, gdelt_clean, reddit_clean)

    # Phase 4: BRI
    bri_df = phase4.calculate_bri(features)

    # Phase 5: validation (VIX) using same date range
    vix_df = phase5.collect_vix_data(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    validation_results = phase5.run_validation_analysis(bri_df, vix_df, market_df)

    # Save deliverables
    deliverables = {
        'bri_timeseries': bri_df,
        'validation_results': validation_results
    }
    phase7.save_all_deliverables(args.out, deliverables)

    # Write summary
    summary = {
        'window': {'start': str(start.date()), 'end': str(end.date())},
        'rows': len(bri_df),
        'created': datetime.now().isoformat(),
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"✅ Processing complete. Outputs in {args.out}")


if __name__ == '__main__':
    main()


