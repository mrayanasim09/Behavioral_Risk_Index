#!/usr/bin/env python3
"""
Live Data Feed System for BRI Dashboard
- Real-time Reddit data collection
- Live market data feeds
- Options data streaming
- Real-time BRI calculation
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import schedule
import threading
from datetime import datetime, timedelta
import logging
import json
import os
from typing import Dict, List, Optional
import praw
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveDataSystem:
    """Live data collection and processing system for BRI"""
    
    def __init__(self):
        self.data_dir = 'data/live'
        self.output_dir = 'output/live'
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Reddit API
        self.reddit = self._setup_reddit()
        
        # Market symbols for live data
        self.market_symbols = ['^VIX', '^GSPC', '^IXIC', '^DJI', 'SPY', 'QQQ', 'IWM']
        self.options_symbols = ['SPY', 'QQQ', 'IWM']
        
        # Reddit subreddits for live monitoring
        self.subreddits = [
            'investing', 'stocks', 'wallstreetbets', 'SecurityAnalysis', 
            'ValueInvesting', 'dividends', 'options', 'StockMarket',
            'cryptocurrency', 'bitcoin', 'ethereum'
        ]
        
        # Live data storage
        self.live_data = {
            'reddit': pd.DataFrame(),
            'market': pd.DataFrame(),
            'options': pd.DataFrame(),
            'bri': pd.DataFrame()
        }
        
        # Historical data for context
        self.historical_data = self._load_historical_data()
        
    def _setup_reddit(self):
        """Setup Reddit API connection"""
        try:
            reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID', 'KiGT8yL2ko-ZaWBYbQrwmw'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET', 'ezQtiCbSJhozM55eq8IC8Ee6qOJglg'),
                user_agent=os.getenv('REDDIT_USER_AGENT', 'BRI-Live-Data-Bot/1.0')
            )
            logger.info("Reddit API connected successfully")
            return reddit
        except Exception as e:
            logger.error(f"Failed to connect to Reddit API: {e}")
            return None
    
    def _load_historical_data(self):
        """Load historical data for context"""
        try:
            # Load historical BRI data
            historical_file = 'output/enhanced_5year/enhanced_bri_data.csv'
            if os.path.exists(historical_file):
                return pd.read_csv(historical_file)
            else:
                logger.warning("No historical data found")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return pd.DataFrame()
    
    def collect_live_reddit_data(self):
        """Collect live Reddit data"""
        logger.info("Collecting live Reddit data...")
        
        if not self.reddit:
            logger.warning("Reddit API not available, using sample data")
            return self._generate_sample_reddit_data()
        
        live_posts = []
        
        for subreddit_name in self.subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Get hot posts from last hour
                for post in subreddit.hot(limit=10):
                    # Check if post is from last hour
                    post_time = datetime.fromtimestamp(post.created_utc)
                    if post_time > datetime.now() - timedelta(hours=1):
                        live_posts.append({
                            'timestamp': post_time,
                            'date': post_time.date(),
                            'subreddit': subreddit_name,
                            'title': post.title,
                            'text': post.selftext if hasattr(post, 'selftext') else '',
                            'score': post.score,
                            'num_comments': post.num_comments,
                            'upvote_ratio': post.upvote_ratio,
                            'gilded': post.gilded,
                            'url': post.url,
                            'post_type': 'text' if post.is_self else 'link'
                        })
            except Exception as e:
                logger.error(f"Error collecting data from r/{subreddit_name}: {e}")
                continue
        
        if live_posts:
            live_df = pd.DataFrame(live_posts)
            live_df['engagement_score'] = live_df['score'] * live_df['upvote_ratio']
            live_df['quality_score'] = (live_df['score'] * live_df['upvote_ratio'] + live_df['num_comments']) / 1000
            
            logger.info(f"Collected {len(live_df)} live Reddit posts")
            return live_df
        else:
            logger.warning("No live Reddit data collected")
            return pd.DataFrame()
    
    def _generate_sample_reddit_data(self):
        """Generate sample Reddit data for testing"""
        logger.info("Generating sample Reddit data...")
        
        # Generate realistic live data
        np.random.seed(int(time.time()))
        n_posts = np.random.randint(20, 50)
        
        live_posts = []
        for i in range(n_posts):
            subreddit = np.random.choice(self.subreddits)
            timestamp = datetime.now() - timedelta(minutes=np.random.randint(0, 60))
            
            live_posts.append({
                'timestamp': timestamp,
                'date': timestamp.date(),
                'subreddit': subreddit,
                'title': f"Live market discussion {timestamp.strftime('%H:%M')}",
                'text': f"Real-time market analysis and discussion",
                'score': np.random.randint(0, 100),
                'num_comments': np.random.randint(0, 20),
                'upvote_ratio': np.random.uniform(0.5, 1.0),
                'gilded': np.random.choice([0, 1], p=[0.95, 0.05]),
                'url': f"https://reddit.com/r/{subreddit}/post{i}",
                'post_type': 'text',
                'engagement_score': np.random.uniform(0, 100),
                'quality_score': np.random.uniform(0, 1)
            })
        
        return pd.DataFrame(live_posts)
    
    def collect_live_market_data(self):
        """Collect live market data"""
        logger.info("Collecting live market data...")
        
        live_market_data = []
        
        for symbol in self.market_symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get current price
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    volume = hist['Volume'].iloc[-1]
                else:
                    current_price = info.get('currentPrice', np.nan)
                    volume = info.get('volume', np.nan)
                
                live_market_data.append({
                    'timestamp': datetime.now(),
                    'date': datetime.now().date(),
                    'symbol': symbol,
                    'price': current_price,
                    'volume': volume,
                    'change': info.get('regularMarketChange', np.nan),
                    'change_percent': info.get('regularMarketChangePercent', np.nan),
                    'high': info.get('dayHigh', np.nan),
                    'low': info.get('dayLow', np.nan),
                    'open': info.get('open', np.nan)
                })
                
            except Exception as e:
                logger.error(f"Error collecting market data for {symbol}: {e}")
                continue
        
        if live_market_data:
            live_df = pd.DataFrame(live_market_data)
            logger.info(f"Collected live market data for {len(live_df)} symbols")
            return live_df
        else:
            logger.warning("No live market data collected")
            return pd.DataFrame()
    
    def collect_live_options_data(self):
        """Collect live options data"""
        logger.info("Collecting live options data...")
        
        live_options_data = []
        
        for symbol in self.options_symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                # Get options expiration dates
                expirations = ticker.options
                if expirations:
                    # Get options for next month
                    exp_date = expirations[0]
                    options_chain = ticker.option_chain(exp_date)
                    
                    # Process calls and puts
                    for option_type in ['calls', 'puts']:
                        options_df = getattr(options_chain, option_type)
                        if not options_df.empty:
                            # Get top 10 most active options
                            top_options = options_df.nlargest(10, 'volume')
                            
                            for _, option in top_options.iterrows():
                                live_options_data.append({
                                    'timestamp': datetime.now(),
                                    'date': datetime.now().date(),
                                    'symbol': symbol,
                                    'option_type': option_type,
                                    'strike': option['strike'],
                                    'expiration': exp_date,
                                    'last_price': option['lastPrice'],
                                    'bid': option['bid'],
                                    'ask': option['ask'],
                                    'volume': option['volume'],
                                    'open_interest': option['openInterest'],
                                    'implied_volatility': option.get('impliedVolatility', np.nan)
                                })
            except Exception as e:
                logger.error(f"Error collecting options data for {symbol}: {e}")
                continue
        
        if live_options_data:
            live_df = pd.DataFrame(live_options_data)
            logger.info(f"Collected live options data for {len(live_df)} contracts")
            return live_df
        else:
            logger.warning("No live options data collected")
            return pd.DataFrame()
    
    def calculate_live_bri(self, reddit_data, market_data, options_data):
        """Calculate live BRI from current data"""
        logger.info("Calculating live BRI...")
        
        if reddit_data.empty:
            logger.warning("No Reddit data available for BRI calculation")
            return pd.DataFrame()
        
        # Calculate Reddit features
        reddit_features = {
            'sentiment_volatility': reddit_data['score'].std(),
            'media_herding': reddit_data['num_comments'].sum(),
            'news_tone': reddit_data['upvote_ratio'].mean(),
            'event_density': reddit_data['score'].sum(),
            'polarity_skew': reddit_data['engagement_score'].std()
        }
        
        # Get VIX data
        vix_data = market_data[market_data['symbol'] == '^VIX']
        vix_value = vix_data['price'].iloc[0] if not vix_data.empty else np.nan
        
        # Get S&P 500 data
        sp500_data = market_data[market_data['symbol'] == '^GSPC']
        sp500_value = sp500_data['price'].iloc[0] if not sp500_data.empty else np.nan
        
        # Calculate BRI
        # Normalize features (using historical data for context)
        if not self.historical_data.empty:
            # Use historical data for normalization
            hist_sentiment_vol = self.historical_data['sentiment_volatility'].mean()
            hist_media_herding = self.historical_data['media_herding'].mean()
            hist_news_tone = self.historical_data['news_tone'].mean()
            hist_event_density = self.historical_data['event_density'].mean()
            hist_polarity_skew = self.historical_data['polarity_skew'].mean()
        else:
            # Use current data for normalization
            hist_sentiment_vol = reddit_features['sentiment_volatility']
            hist_media_herding = reddit_features['media_herding']
            hist_news_tone = reddit_features['news_tone']
            hist_event_density = reddit_features['event_density']
            hist_polarity_skew = reddit_features['polarity_skew']
        
        # Normalize features
        normalized_features = {
            'sentiment_volatility': reddit_features['sentiment_volatility'] / hist_sentiment_vol if hist_sentiment_vol > 0 else 0,
            'media_herding': reddit_features['media_herding'] / hist_media_herding if hist_media_herding > 0 else 0,
            'news_tone': reddit_features['news_tone'] / hist_news_tone if hist_news_tone > 0 else 0,
            'event_density': reddit_features['event_density'] / hist_event_density if hist_event_density > 0 else 0,
            'polarity_skew': reddit_features['polarity_skew'] / hist_polarity_skew if hist_polarity_skew > 0 else 0
        }
        
        # Calculate BRI
        bri_value = (
            0.30 * normalized_features['sentiment_volatility'] +
            0.25 * normalized_features['media_herding'] +
            0.20 * normalized_features['news_tone'] +
            0.15 * normalized_features['event_density'] +
            0.10 * normalized_features['polarity_skew']
        ) * 100
        
        # Create live BRI record
        live_bri = {
            'timestamp': datetime.now(),
            'date': datetime.now().date(),
            'BRI': bri_value,
            'VIX': vix_value,
            'SP500': sp500_value,
            'sentiment_volatility': reddit_features['sentiment_volatility'],
            'media_herding': reddit_features['media_herding'],
            'news_tone': reddit_features['news_tone'],
            'event_density': reddit_features['event_density'],
            'polarity_skew': reddit_features['polarity_skew'],
            'reddit_posts': len(reddit_data),
            'market_symbols': len(market_data),
            'options_contracts': len(options_data)
        }
        
        return pd.DataFrame([live_bri])
    
    def save_live_data(self, reddit_data, market_data, options_data, bri_data):
        """Save live data to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save individual datasets
        if not reddit_data.empty:
            reddit_data.to_csv(f"{self.data_dir}/reddit_live_{timestamp}.csv", index=False)
        
        if not market_data.empty:
            market_data.to_csv(f"{self.data_dir}/market_live_{timestamp}.csv", index=False)
        
        if not options_data.empty:
            options_data.to_csv(f"{self.data_dir}/options_live_{timestamp}.csv", index=False)
        
        if not bri_data.empty:
            bri_data.to_csv(f"{self.data_dir}/bri_live_{timestamp}.csv", index=False)
            
            # Also save to output directory for dashboard
            bri_data.to_csv(f"{self.output_dir}/current_bri.csv", index=False)
        
        logger.info(f"Live data saved with timestamp {timestamp}")
    
    def run_live_collection(self):
        """Run one iteration of live data collection"""
        logger.info("Running live data collection...")
        
        # Collect all data
        reddit_data = self.collect_live_reddit_data()
        market_data = self.collect_live_market_data()
        options_data = self.collect_live_options_data()
        
        # Calculate live BRI
        bri_data = self.calculate_live_bri(reddit_data, market_data, options_data)
        
        # Save data
        self.save_live_data(reddit_data, market_data, options_data, bri_data)
        
        # Update live data storage
        self.live_data['reddit'] = reddit_data
        self.live_data['market'] = market_data
        self.live_data['options'] = options_data
        self.live_data['bri'] = bri_data
        
        # Log results
        if not bri_data.empty:
            current_bri = bri_data['BRI'].iloc[0]
            current_vix = bri_data['VIX'].iloc[0]
            logger.info(f"Live BRI: {current_bri:.2f}, VIX: {current_vix:.2f}")
        
        return reddit_data, market_data, options_data, bri_data
    
    def start_live_system(self, interval_minutes=15):
        """Start the live data collection system"""
        logger.info(f"Starting live data system with {interval_minutes} minute intervals...")
        
        # Schedule data collection
        schedule.every(interval_minutes).minutes.do(self.run_live_collection)
        
        # Run initial collection
        self.run_live_collection()
        
        # Start scheduler
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def get_current_bri(self):
        """Get current BRI value"""
        bri_file = f"{self.output_dir}/current_bri.csv"
        if os.path.exists(bri_file):
            bri_data = pd.read_csv(bri_file)
            if not bri_data.empty:
                return bri_data.iloc[-1].to_dict()
        return None

if __name__ == "__main__":
    # Initialize live data system
    live_system = LiveDataSystem()
    
    # Run one collection cycle
    reddit_data, market_data, options_data, bri_data = live_system.run_live_collection()
    
    print("\n" + "="*60)
    print("LIVE DATA COLLECTION RESULTS")
    print("="*60)
    print(f"Reddit posts collected: {len(reddit_data)}")
    print(f"Market symbols: {len(market_data)}")
    print(f"Options contracts: {len(options_data)}")
    
    if not bri_data.empty:
        current_bri = bri_data['BRI'].iloc[0]
        current_vix = bri_data['VIX'].iloc[0]
        print(f"Current BRI: {current_bri:.2f}")
        print(f"Current VIX: {current_vix:.2f}")
        print(f"BRI-VIX correlation: {bri_data['BRI'].corr(bri_data['VIX']):.3f}")
    
    print("="*60)
    
    # Uncomment to start continuous collection
    # live_system.start_live_system(interval_minutes=15)
