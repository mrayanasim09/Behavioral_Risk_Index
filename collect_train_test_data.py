#!/usr/bin/env python3
"""
Train/Test Data Collection Script
Collects training data (2020-2024) and test data (2018-2019) for proper validation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collect import DataCollector
from reddit_api import RedditAPIClient
from gdelt_processor import GDELTProcessor
import pandas as pd
import yaml
from datetime import datetime, timedelta
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def collect_market_data_period(start_date, end_date, period_name):
    """Collect market data for a specific period"""
    logger.info(f"📈 Collecting {period_name} market data: {start_date} to {end_date}")
    
    config = load_config()
    collector = DataCollector(config)
    
    try:
        market_data = collector.collect_market_data(start_date, end_date)
        
        if market_data.empty:
            logger.warning(f"No market data collected for {period_name}")
            return pd.DataFrame()
        
        # Ensure date column exists
        if 'Date' in market_data.columns:
            market_data['date'] = pd.to_datetime(market_data['Date']).dt.date
        elif 'date' not in market_data.columns:
            logger.error("No date column found in market data")
            return pd.DataFrame()
        
        logger.info(f"✅ {period_name} market data: {len(market_data):,} records")
        logger.info(f"   Date range: {market_data['date'].min()} to {market_data['date'].max()}")
        
        return market_data
        
    except Exception as e:
        logger.error(f"❌ Error collecting {period_name} market data: {e}")
        return pd.DataFrame()

def collect_reddit_data_period(start_date, end_date, period_name, limit_per_subreddit=100):
    """Collect Reddit data for a specific period"""
    logger.info(f"📱 Collecting {period_name} Reddit data: {start_date} to {end_date}")
    
    try:
        reddit_client = RedditAPIClient()
        config = load_config()
        subreddits = config['data_sources']['reddit']['subreddits'][:10]  # Limit to top 10 for efficiency
        
        all_posts = []
        successful_subreddits = 0
        
        for i, subreddit in enumerate(subreddits):
            try:
                logger.info(f"   Collecting from r/{subreddit} ({i+1}/{len(subreddits)})")
                posts = reddit_client.collect_subreddit_data(
                    subreddit, 
                    limit=limit_per_subreddit,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if posts:
                    all_posts.extend(posts)
                    successful_subreddits += 1
                    logger.info(f"   ✅ r/{subreddit}: {len(posts)} posts")
                else:
                    logger.info(f"   ⚠️  r/{subreddit}: No posts found")
                    
            except Exception as e:
                logger.warning(f"   ❌ r/{subreddit}: {e}")
        
        if all_posts:
            posts_data = [post.to_dict() for post in all_posts]
            reddit_df = pd.DataFrame(posts_data)
            
            # Convert created_utc to datetime
            reddit_df['created_utc'] = pd.to_datetime(reddit_df['created_utc'], unit='s')
            reddit_df['date'] = reddit_df['created_utc'].dt.date
            
            logger.info(f"✅ {period_name} Reddit data: {len(reddit_df):,} posts")
            logger.info(f"   Date range: {reddit_df['date'].min()} to {reddit_df['date'].max()}")
            logger.info(f"   Successful subreddits: {successful_subreddits}/{len(subreddits)}")
            
            return reddit_df
        else:
            logger.warning(f"No Reddit data collected for {period_name}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"❌ Error collecting {period_name} Reddit data: {e}")
        return pd.DataFrame()

def collect_gdelt_data_period(start_date, end_date, period_name):
    """Collect GDELT data for a specific period"""
    logger.info(f"📰 Collecting {period_name} GDELT data: {start_date} to {end_date}")
    
    try:
        config = load_config()
        gdelt_processor = GDELTProcessor()
        
        # Create sample data for the period
        gdelt_data = gdelt_processor.create_sample_data(start_date, end_date)
        
        logger.info(f"✅ {period_name} GDELT data: {len(gdelt_data):,} events")
        logger.info(f"   Date range: {gdelt_data['date'].min()} to {gdelt_data['date'].max()}")
        
        return gdelt_data
        
    except Exception as e:
        logger.error(f"❌ Error collecting {period_name} GDELT data: {e}")
        return pd.DataFrame()

def analyze_data_distribution(train_data, test_data, data_type):
    """Analyze data distribution between train and test sets"""
    logger.info(f"\n📊 {data_type} Data Distribution Analysis:")
    
    if train_data.empty and test_data.empty:
        logger.warning(f"No {data_type} data available")
        return
    
    if not train_data.empty:
        train_days = (train_data['date'].max() - train_data['date'].min()).days + 1
        train_records = len(train_data)
        logger.info(f"   Training Set: {train_records:,} records over {train_days} days")
        logger.info(f"   Date range: {train_data['date'].min()} to {train_data['date'].max()}")
    
    if not test_data.empty:
        test_days = (test_data['date'].max() - test_data['date'].min()).days + 1
        test_records = len(test_data)
        logger.info(f"   Test Set: {test_records:,} records over {test_days} days")
        logger.info(f"   Date range: {test_data['date'].min()} to {test_data['date'].max()}")
    
    if not train_data.empty and not test_data.empty:
        total_records = train_records + test_records
        train_ratio = train_records / total_records
        test_ratio = test_records / total_records
        
        logger.info(f"   Distribution: {train_ratio:.1%} train, {test_ratio:.1%} test")
        logger.info(f"   Total: {total_records:,} records")

def save_data_sets(market_train, market_test, reddit_train, reddit_test, gdelt_train, gdelt_test):
    """Save train and test datasets"""
    logger.info("\n💾 Saving train/test datasets...")
    
    # Create output directories
    train_dir = "data/train"
    test_dir = "data/test"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Save market data
    if not market_train.empty:
        market_train.to_csv(f"{train_dir}/market_data.csv", index=False)
        logger.info(f"✅ Training market data saved: {len(market_train):,} records")
    
    if not market_test.empty:
        market_test.to_csv(f"{test_dir}/market_data.csv", index=False)
        logger.info(f"✅ Test market data saved: {len(market_test):,} records")
    
    # Save Reddit data
    if not reddit_train.empty:
        reddit_train.to_csv(f"{train_dir}/reddit_data.csv", index=False)
        logger.info(f"✅ Training Reddit data saved: {len(reddit_train):,} records")
    
    if not reddit_test.empty:
        reddit_test.to_csv(f"{test_dir}/reddit_data.csv", index=False)
        logger.info(f"✅ Test Reddit data saved: {len(reddit_test):,} records")
    
    # Save GDELT data
    if not gdelt_train.empty:
        gdelt_train.to_csv(f"{train_dir}/gdelt_data.csv", index=False)
        logger.info(f"✅ Training GDELT data saved: {len(gdelt_train):,} records")
    
    if not gdelt_test.empty:
        gdelt_test.to_csv(f"{test_dir}/gdelt_data.csv", index=False)
        logger.info(f"✅ Test GDELT data saved: {len(gdelt_test):,} records")

def main():
    """Main function to collect train/test data"""
    logger.info("🚀 BRI Train/Test Data Collection")
    logger.info("=" * 60)
    
    # Define train/test periods
    train_start = datetime(2020, 1, 1).date()
    train_end = datetime(2024, 12, 31).date()
    test_start = datetime(2018, 1, 1).date()
    test_end = datetime(2019, 12, 31).date()
    
    logger.info(f"📅 Training Period: {train_start} to {train_end} (5 years)")
    logger.info(f"📅 Test Period: {test_start} to {test_end} (2 years)")
    logger.info(f"📊 Train/Test Split: ~71% / ~29%")
    
    # Collect training data (2020-2024)
    logger.info("\n" + "="*40)
    logger.info("COLLECTING TRAINING DATA (2020-2024)")
    logger.info("="*40)
    
    market_train = collect_market_data_period(train_start, train_end, "Training")
    reddit_train = collect_reddit_data_period(train_start, train_end, "Training", limit_per_subreddit=50)
    gdelt_train = collect_gdelt_data_period(train_start, train_end, "Training")
    
    # Collect test data (2018-2019)
    logger.info("\n" + "="*40)
    logger.info("COLLECTING TEST DATA (2018-2019)")
    logger.info("="*40)
    
    market_test = collect_market_data_period(test_start, test_end, "Test")
    reddit_test = collect_reddit_data_period(test_start, test_end, "Test", limit_per_subreddit=30)
    gdelt_test = collect_gdelt_data_period(test_start, test_end, "Test")
    
    # Analyze distributions
    logger.info("\n" + "="*40)
    logger.info("DATA DISTRIBUTION ANALYSIS")
    logger.info("="*40)
    
    analyze_data_distribution(market_train, market_test, "Market")
    analyze_data_distribution(reddit_train, reddit_test, "Reddit")
    analyze_data_distribution(gdelt_train, gdelt_test, "GDELT")
    
    # Save datasets
    save_data_sets(market_train, market_test, reddit_train, reddit_test, gdelt_train, gdelt_test)
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("🎯 TRAIN/TEST DATA COLLECTION SUMMARY")
    logger.info("="*60)
    
    train_total = sum([len(df) for df in [market_train, reddit_train, gdelt_train] if not df.empty])
    test_total = sum([len(df) for df in [market_test, reddit_test, gdelt_test] if not df.empty])
    total_records = train_total + test_total
    
    logger.info(f"📊 Training Data: {train_total:,} records")
    logger.info(f"📊 Test Data: {test_total:,} records")
    logger.info(f"📊 Total Data: {total_records:,} records")
    logger.info(f"📈 Train/Test Ratio: {train_total/total_records:.1%} / {test_total/total_records:.1%}")
    
    logger.info(f"\n✅ Train/test data collection completed!")
    logger.info(f"📁 Training data: data/train/")
    logger.info(f"📁 Test data: data/test/")
    logger.info(f"🎯 Ready for model training and validation!")

if __name__ == "__main__":
    main()
