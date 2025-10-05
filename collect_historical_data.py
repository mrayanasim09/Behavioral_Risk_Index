#!/usr/bin/env python3
"""
Historical Data Collection Script
This script collects historical data for the BRI project
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def collect_market_data(start_date, end_date):
    """Collect historical market data"""
    logger.info(f"Collecting market data from {start_date} to {end_date}")
    
    config = load_config()
    collector = DataCollector(config)
    
    # Collect market data
    market_data = collector.collect_market_data(start_date, end_date)
    
    logger.info(f"Market data collected: {market_data.shape}")
    logger.info(f"Columns: {list(market_data.columns)}")
    
    # Check if date column exists
    if 'date' in market_data.columns:
        logger.info(f"Date range: {market_data['date'].min()} to {market_data['date'].max()}")
    elif 'Date' in market_data.columns:
        logger.info(f"Date range: {market_data['Date'].min()} to {market_data['Date'].max()}")
        market_data['date'] = market_data['Date']
    
    if 'symbol' in market_data.columns:
        logger.info(f"Symbols: {market_data['symbol'].nunique()}")
    elif 'Symbol' in market_data.columns:
        logger.info(f"Symbols: {market_data['Symbol'].nunique()}")
        market_data['symbol'] = market_data['Symbol']
    
    return market_data

def collect_reddit_data(start_date, end_date, limit_per_subreddit=50):
    """Collect historical Reddit data"""
    logger.info(f"Collecting Reddit data from {start_date} to {end_date}")
    
    config = load_config()
    reddit_client = RedditAPIClient()
    
    # Get subreddits from config
    subreddits = config['data_sources']['reddit']['subreddits']
    
    all_posts = []
    successful_subreddits = 0
    
    for i, subreddit in enumerate(subreddits[:10]):  # Limit to first 10 for demo
        try:
            logger.info(f"Collecting from r/{subreddit} ({i+1}/10)")
            posts = reddit_client.collect_subreddit_data(
                subreddit, 
                limit=limit_per_subreddit,
                start_date=start_date,
                end_date=end_date
            )
            
            if posts:
                all_posts.extend(posts)
                successful_subreddits += 1
                logger.info(f"Collected {len(posts)} posts from r/{subreddit}")
            else:
                logger.info(f"No posts found in r/{subreddit}")
                
        except Exception as e:
            logger.warning(f"Error collecting from r/{subreddit}: {e}")
    
    # Convert to DataFrame
    if all_posts:
        posts_data = [post.to_dict() for post in all_posts]
        reddit_df = pd.DataFrame(posts_data)
        
        # Convert created_utc to datetime
        reddit_df['created_utc'] = pd.to_datetime(reddit_df['created_utc'], unit='s')
        reddit_df['date'] = reddit_df['created_utc'].dt.date
        
        logger.info(f"Reddit data collected: {reddit_df.shape}")
        logger.info(f"Date range: {reddit_df['date'].min()} to {reddit_df['date'].max()}")
        logger.info(f"Successful subreddits: {successful_subreddits}")
        logger.info(f"Average posts per day: {len(reddit_df) / ((end_date - start_date).days + 1):.1f}")
        
        return reddit_df
    else:
        logger.warning("No Reddit data collected")
        return pd.DataFrame()

def collect_gdelt_data(start_date, end_date):
    """Collect historical GDELT data"""
    logger.info(f"Collecting GDELT data from {start_date} to {end_date}")
    
    config = load_config()
    gdelt_processor = GDELTProcessor(config)
    
    # For demo purposes, create sample data
    # In production, you would use the GDELT API
    gdelt_data = gdelt_processor.create_sample_data(start_date, end_date)
    
    logger.info(f"GDELT data collected: {gdelt_data.shape}")
    logger.info(f"Date range: {gdelt_data['date'].min()} to {gdelt_data['date'].max()}")
    
    return gdelt_data

def analyze_data_coverage(market_data, reddit_data, gdelt_data):
    """Analyze data coverage and quality"""
    logger.info("\n" + "="*60)
    logger.info("DATA COVERAGE ANALYSIS")
    logger.info("="*60)
    
    # Market data analysis
    if not market_data.empty:
        market_days = (market_data['date'].max() - market_data['date'].min()).days + 1
        market_symbols = market_data['symbol'].nunique()
        logger.info(f"Market Data:")
        logger.info(f"  - Time period: {market_days} days")
        logger.info(f"  - Symbols: {market_symbols}")
        logger.info(f"  - Records: {len(market_data):,}")
        logger.info(f"  - Coverage: {len(market_data) / (market_days * market_symbols) * 100:.1f}%")
    
    # Reddit data analysis
    if not reddit_data.empty:
        reddit_days = (reddit_data['date'].max() - reddit_data['date'].min()).days + 1
        reddit_subreddits = reddit_data['subreddit'].nunique()
        logger.info(f"\nReddit Data:")
        logger.info(f"  - Time period: {reddit_days} days")
        logger.info(f"  - Subreddits: {reddit_subreddits}")
        logger.info(f"  - Posts: {len(reddit_data):,}")
        logger.info(f"  - Average posts/day: {len(reddit_data) / reddit_days:.1f}")
        
        # Top subreddits
        top_subreddits = reddit_data['subreddit'].value_counts().head(5)
        logger.info(f"  - Top subreddits: {dict(top_subreddits)}")
    
    # GDELT data analysis
    if not gdelt_data.empty:
        gdelt_days = (gdelt_data['date'].max() - gdelt_data['date'].min()).days + 1
        logger.info(f"\nGDELT Data:")
        logger.info(f"  - Time period: {gdelt_days} days")
        logger.info(f"  - Events: {len(gdelt_data):,}")
        logger.info(f"  - Average events/day: {len(gdelt_data) / gdelt_days:.1f}")
    
    # Overall coverage
    total_days = max([
        (market_data['date'].max() - market_data['date'].min()).days + 1 if not market_data.empty else 0,
        (reddit_data['date'].max() - reddit_data['date'].min()).days + 1 if not reddit_data.empty else 0,
        (gdelt_data['date'].max() - gdelt_data['date'].min()).days + 1 if not gdelt_data.empty else 0
    ])
    
    logger.info(f"\nOverall Coverage:")
    logger.info(f"  - Total time period: {total_days} days")
    logger.info(f"  - Data sources: {sum([not df.empty for df in [market_data, reddit_data, gdelt_data]])}/3")
    logger.info(f"  - Total records: {sum([len(df) for df in [market_data, reddit_data, gdelt_data]]):,}")

def main():
    """Main function to collect historical data"""
    logger.info("🚀 Starting Historical Data Collection")
    
    # Define date ranges for different periods
    date_ranges = {
        "1_week": (datetime(2024, 1, 1), datetime(2024, 1, 7)),
        "1_month": (datetime(2024, 1, 1), datetime(2024, 1, 31)),
        "3_months": (datetime(2024, 1, 1), datetime(2024, 3, 31)),
        "6_months": (datetime(2024, 1, 1), datetime(2024, 6, 30)),
        "1_year": (datetime(2024, 1, 1), datetime(2024, 12, 31)),
        "2_years": (datetime(2023, 1, 1), datetime(2024, 12, 31)),
        "5_years": (datetime(2020, 1, 1), datetime(2024, 12, 31))
    }
    
    # Choose period (start with 1 month for demo)
    period = "1_month"
    start_date, end_date = date_ranges[period]
    
    logger.info(f"Collecting data for period: {period} ({start_date.date()} to {end_date.date()})")
    
    # Collect data
    market_data = collect_market_data(start_date.date(), end_date.date())
    reddit_data = collect_reddit_data(start_date.date(), end_date.date(), limit_per_subreddit=20)
    gdelt_data = collect_gdelt_data(start_date.date(), end_date.date())
    
    # Analyze coverage
    analyze_data_coverage(market_data, reddit_data, gdelt_data)
    
    # Save data
    output_dir = f"output/historical_{period}"
    os.makedirs(output_dir, exist_ok=True)
    
    if not market_data.empty:
        market_data.to_csv(f"{output_dir}/market_data.csv", index=False)
        logger.info(f"Market data saved to {output_dir}/market_data.csv")
    
    if not reddit_data.empty:
        reddit_data.to_csv(f"{output_dir}/reddit_data.csv", index=False)
        logger.info(f"Reddit data saved to {output_dir}/reddit_data.csv")
    
    if not gdelt_data.empty:
        gdelt_data.to_csv(f"{output_dir}/gdelt_data.csv", index=False)
        logger.info(f"GDELT data saved to {output_dir}/gdelt_data.csv")
    
    logger.info(f"\n✅ Historical data collection completed!")
    logger.info(f"📁 Data saved to: {output_dir}/")
    
    # Show data summary
    total_records = sum([len(df) for df in [market_data, reddit_data, gdelt_data]])
    logger.info(f"📊 Total records collected: {total_records:,}")
    
    return market_data, reddit_data, gdelt_data

if __name__ == "__main__":
    main()
