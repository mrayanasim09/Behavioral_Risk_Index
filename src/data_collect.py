"""
Data collection module for BRI pipeline.
Collects data from GDELT, Reddit, Twitter, and market sources.
"""

import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import praw
import tweepy
from dotenv import load_dotenv
import logging
from google.cloud import bigquery
import pandas as pd

from utils import ensure_directory, filter_spam_posts, chunk_dates, save_with_metadata

# Load environment variables
load_dotenv()

class DataCollector:
    """Main data collection class for BRI pipeline."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # API credentials
        self.gdelt_api_key = os.getenv('GDELT_API_KEY')
        self.newsapi_key = os.getenv('NEWSAPI_KEY')
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID') or "KiGT8yL2ko-ZaWBYbQrwmw"
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET') or "ezQtiCbSJhozM55eq8IC8Ee6qOJglg"
        self.reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'BRI-Research-Bot/1.0')
        
        # Initialize API clients
        self._setup_reddit()
        self._setup_twitter()
        self._setup_bigquery()
        
    def _setup_reddit(self):
        """Initialize Reddit API client."""
        if self.reddit_client_id and self.reddit_client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=self.reddit_client_id,
                    client_secret=self.reddit_client_secret,
                    user_agent=self.reddit_user_agent
                )
                self.reddit.read_only = True
                self.logger.info("Reddit API initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Reddit API: {e}")
                self.reddit = None
        else:
            self.reddit = None
            self.logger.warning("Reddit credentials not found")
    
    def _setup_twitter(self):
        """Initialize Twitter API client."""
        if self.twitter_bearer_token:
            try:
                self.twitter_client = tweepy.Client(
                    bearer_token=self.twitter_bearer_token,
                    wait_on_rate_limit=True
                )
                self.logger.info("Twitter API initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Twitter API: {e}")
                self.twitter_client = None
        else:
            self.twitter_client = None
            self.logger.warning("Twitter credentials not found")
    
    def _setup_bigquery(self):
        """Initialize BigQuery client for GDELT data."""
        try:
            self.bigquery_client = bigquery.Client()
            self.logger.info("BigQuery client initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize BigQuery client: {e}")
            self.bigquery_client = None
    
    def collect_market_data(self, start_date: str, end_date: str, 
                          symbols: List[str] = None) -> pd.DataFrame:
        """Collect comprehensive market data including S&P 500, VIX, and other indices."""
        if symbols is None:
            # Comprehensive market data collection for research
            symbols = [
                '^GSPC',  # S&P 500
                '^VIX',   # VIX
                '^TNX',   # 10-Year Treasury
                '^FVX',   # 5-Year Treasury
                '^TYX',   # 30-Year Treasury
                '^DJI',   # Dow Jones
                '^IXIC',  # NASDAQ
                '^RUT',   # Russell 2000
                '^VIX9D', # VIX 9-day
                '^VVIX',  # VIX of VIX
                'SPY',    # SPDR S&P 500 ETF
                'QQQ',    # Invesco QQQ
                'IWM',    # iShares Russell 2000
                'GLD',    # Gold ETF
                'TLT',    # 20+ Year Treasury Bond ETF
                'HYG',    # High Yield Corporate Bond ETF
                'LQD',    # Investment Grade Corporate Bond ETF
                'EFA',    # EAFE ETF
                'EEM',    # Emerging Markets ETF
                'VXX',    # VIX Short-Term Futures ETF
                'UVXY',   # VIX Short-Term Futures 2x ETF
                'SVXY'    # VIX Short-Term Futures Inverse ETF
            ]
        
        self.logger.info(f"Collecting comprehensive market data from {start_date} to {end_date}")
        
        all_data = []
        
        for symbol in symbols:
            try:
                self.logger.info(f"Collecting data for {symbol}")
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date, auto_adjust=True, back_adjust=True)
                
                if not data.empty:
                    # Calculate returns and realized volatility
                    data[f'{symbol}_returns'] = data['Close'].pct_change()
                    data[f'{symbol}_log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
                    data[f'{symbol}_realized_vol'] = data[f'{symbol}_returns'].rolling(window=20).std() * np.sqrt(252)
                    data[f'{symbol}_realized_vol_5d'] = data[f'{symbol}_returns'].rolling(window=5).std() * np.sqrt(252)
                    data[f'{symbol}_realized_vol_10d'] = data[f'{symbol}_returns'].rolling(window=10).std() * np.sqrt(252)
                    
                    # Calculate additional metrics
                    data[f'{symbol}_high_low_spread'] = (data['High'] - data['Low']) / data['Close']
                    data[f'{symbol}_volume_ratio'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
                    
                    # Add symbol prefix to all columns except Date
                    data.columns = [f'{symbol}_{col}' if col != 'Date' else col for col in data.columns]
                    data = data.reset_index()
                    
                    all_data.append(data)
                    self.logger.info(f"Collected {len(data)} records for {symbol}")
                else:
                    self.logger.warning(f"No data found for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Error collecting data for {symbol}: {e}")
        
        if all_data:
            # Merge all data on Date
            combined_data = all_data[0]
            for data in all_data[1:]:
                combined_data = pd.merge(combined_data, data, on='Date', how='outer')
            
            # Sort by date
            combined_data = combined_data.sort_values('Date').reset_index(drop=True)
            
            # Calculate market-wide metrics
            combined_data = self._calculate_market_metrics(combined_data)
            
            # Save market data
            output_path = f"data/raw/market_{start_date}_{end_date}.csv"
            ensure_directory(os.path.dirname(output_path))
            combined_data.to_csv(output_path, index=False)
            
            return combined_data
        else:
            self.logger.error("No market data collected")
            return pd.DataFrame()
    
    def _calculate_market_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional market-wide metrics."""
        # Market-wide realized volatility (using S&P 500 as proxy)
        if '^GSPC_returns' in df.columns:
            df['market_realized_vol'] = df['^GSPC_returns'].rolling(window=20).std() * np.sqrt(252)
            df['market_realized_vol_5d'] = df['^GSPC_returns'].rolling(window=5).std() * np.sqrt(252)
            df['market_realized_vol_10d'] = df['^GSPC_returns'].rolling(window=10).std() * np.sqrt(252)
        
        # VIX term structure (if available)
        if '^VIX_Close' in df.columns and '^VIX9D_Close' in df.columns:
            df['vix_term_structure'] = df['^VIX_Close'] - df['^VIX9D_Close']
        
        # Market stress indicators
        if '^VIX_Close' in df.columns:
            df['vix_percentile'] = df['^VIX_Close'].rolling(window=252).rank(pct=True)
            df['vix_spike'] = (df['^VIX_Close'] > df['^VIX_Close'].rolling(window=20).mean() + 2 * df['^VIX_Close'].rolling(window=20).std()).astype(int)
        
        # Cross-asset correlations (if enough data)
        return_cols = [col for col in df.columns if col.endswith('_returns') and '^GSPC' in col]
        if len(return_cols) > 1:
            # Calculate rolling correlations with S&P 500
            sp500_returns_col = None
            for col in return_cols:
                if '^GSPC' in col and col.endswith('_returns'):
                    sp500_returns_col = col
                    break
            
            if sp500_returns_col:
                for col in return_cols:
                    if col != sp500_returns_col:
                        df[f'{col}_corr_sp500'] = df[sp500_returns_col].rolling(window=20).corr(df[col])
        
        return df
    
    def fetch_gdelt_financial_events(self, start_date: str, end_date: str, limit: int = 2000) -> pd.DataFrame:
        """Fetch GDELT financial events using Google BigQuery."""
        if not self.bigquery_client:
            self.logger.warning("BigQuery client not available, using sample data")
            return self._get_sample_news_data()
        
        self.logger.info(f"Fetching GDELT financial events from {start_date} to {end_date}")
        
        # Convert dates to GDELT format (YYYYMMDD)
        start_date_formatted = start_date.replace('-', '')
        end_date_formatted = end_date.replace('-', '')
        
        query = f"""
        SELECT 
            SQLDATE,
            Actor1Name,
            Actor2Name,
            EventRootCode,
            GoldsteinScale,
            AvgTone,
            SourceURL,
            Actor1CountryCode,
            Actor2CountryCode,
            IsRootEvent,
            EventCode,
            EventBaseCode,
            EventRootCode,
            QuadClass,
            NumMentions,
            NumSources,
            NumArticles,
            AvgTone,
            Actor1Geo_Type,
            Actor1Geo_FullName,
            Actor1Geo_CountryCode,
            Actor1Geo_ADM1Code,
            Actor1Geo_Lat,
            Actor1Geo_Long,
            Actor1Geo_FeatureID,
            Actor2Geo_Type,
            Actor2Geo_FullName,
            Actor2Geo_CountryCode,
            Actor2Geo_ADM1Code,
            Actor2Geo_Lat,
            Actor2Geo_Long,
            Actor2Geo_FeatureID,
            ActionGeo_Type,
            ActionGeo_FullName,
            ActionGeo_CountryCode,
            ActionGeo_ADM1Code,
            ActionGeo_Lat,
            ActionGeo_Long,
            ActionGeo_FeatureID,
            DATEADDED,
            SOURCEURL
        FROM `gdelt-bq.gdeltv2.events`
        WHERE (
            Actor1Name LIKE '%STOCK%' OR 
            Actor2Name LIKE '%MARKET%' OR 
            Actor1Name LIKE '%FINANCE%' OR
            Actor2Name LIKE '%FINANCE%' OR
            Actor1Name LIKE '%BANK%' OR
            Actor2Name LIKE '%BANK%' OR
            Actor1Name LIKE '%ECONOMY%' OR
            Actor2Name LIKE '%ECONOMY%' OR
            Actor1Name LIKE '%INVESTMENT%' OR
            Actor2Name LIKE '%INVESTMENT%' OR
            Actor1Name LIKE '%TRADING%' OR
            Actor2Name LIKE '%TRADING%' OR
            EventRootCode IN ('14', '15', '16', '17', '18') OR
            EventBaseCode IN ('140', '141', '142', '143', '144', '145', '146', '147', '148', '149',
                             '150', '151', '152', '153', '154', '155', '156', '157', '158', '159',
                             '160', '161', '162', '163', '164', '165', '166', '167', '168', '169',
                             '170', '171', '172', '173', '174', '175', '176', '177', '178', '179',
                             '180', '181', '182', '183', '184', '185', '186', '187', '188', '189')
        )
        AND SQLDATE BETWEEN {start_date_formatted} AND {end_date_formatted}
        ORDER BY SQLDATE DESC
        LIMIT {limit};
        """
        
        try:
            df = self.bigquery_client.query(query).to_dataframe()
            
            if not df.empty:
                # Convert SQLDATE to datetime
                df['date'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d')
                
                # Create headline from available fields
                df['headline'] = df.apply(lambda row: self._create_headline_from_gdelt(row), axis=1)
                
                # Filter out rows without meaningful headlines
                df = df[df['headline'].str.len() > 10]
                
                # Select relevant columns for news data
                news_df = df[['date', 'headline', 'SourceURL', 'AvgTone', 'GoldsteinScale', 
                             'Actor1Name', 'Actor2Name', 'EventRootCode']].copy()
                
                # Rename columns for consistency
                news_df.columns = ['date', 'headline', 'url', 'tone', 'goldstein_scale', 
                                 'actor1', 'actor2', 'event_code']
                
                # Add source
                news_df['source'] = 'GDELT'
                
                self.logger.info(f"Fetched {len(news_df)} GDELT financial events")
                
                return news_df
            else:
                self.logger.warning("No GDELT events found, using sample data")
                return self._get_sample_news_data()
                
        except Exception as e:
            self.logger.error(f"Error fetching GDELT data: {e}")
            return self._get_sample_news_data()
    
    def _create_headline_from_gdelt(self, row) -> str:
        """Create a headline from GDELT row data."""
        actor1 = str(row['Actor1Name']) if pd.notna(row['Actor1Name']) else ''
        actor2 = str(row['Actor2Name']) if pd.notna(row['Actor2Name']) else ''
        event_code = str(row['EventRootCode']) if pd.notna(row['EventRootCode']) else ''
        
        # Create a simple headline
        if actor1 and actor2:
            headline = f"{actor1} and {actor2} financial event"
        elif actor1:
            headline = f"{actor1} financial activity"
        elif actor2:
            headline = f"{actor2} financial activity"
        else:
            headline = f"Financial event {event_code}"
        
        return headline
    
    def collect_gdelt_news(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Collect news data from GDELT using BigQuery."""
        self.logger.info(f"Collecting GDELT news from {start_date} to {end_date}")
        
        # Use BigQuery to fetch GDELT data
        news_df = self.fetch_gdelt_financial_events(start_date, end_date, limit=5000)
        
        if not news_df.empty:
            # Save news data
            output_path = f"data/raw/news_{start_date}_{end_date}.csv"
            ensure_directory(os.path.dirname(output_path))
            news_df.to_csv(output_path, index=False)
            
            return news_df
        else:
            self.logger.warning("No GDELT news data collected, using sample data")
            return self._get_sample_news_data()
    
    def collect_reddit_data(self, start_date: str, end_date: str, 
                          subreddits: List[str] = None) -> pd.DataFrame:
        """Collect Reddit data from specified subreddits."""
        if not self.reddit:
            self.logger.warning("Reddit API not available, using sample data")
            return self._get_sample_reddit_data()
        
        if subreddits is None:
            # Comprehensive list of finance-related subreddits for research-grade data
            subreddits = [
                'investing', 'wallstreetbets', 'stocks', 'SecurityAnalysis', 'ValueInvesting',
                'dividends', 'options', 'pennystocks', 'StockMarket', 'investing_discussion',
                'financialindependence', 'personalfinance', 'cryptocurrency', 'bitcoin',
                'ethereum', 'cryptomarkets', 'cryptocurrencytrading', 'defi', 'altcoin',
                'financialcareers', 'accounting', 'tax', 'realestateinvesting', 'landlord',
                'fire', 'leanfire', 'fatfire', 'bogleheads', 'portfolios', 'wealthfront',
                'robinhood', 'tdameritrade', 'fidelity', 'schwab', 'etrade', 'webull',
                'trading', 'daytrading', 'swingtrading', 'algotrading', 'forex', 'futures',
                'commodities', 'bonds', 'fixedincome', 'reits', 'mutualfunds', 'etfs'
            ]
        
        self.logger.info(f"Collecting comprehensive Reddit data from {start_date} to {end_date}")
        self.logger.info(f"Targeting {len(subreddits)} subreddits for substantial data collection")
        
        all_posts = []
        start_timestamp = int(pd.to_datetime(start_date).timestamp())
        end_timestamp = int(pd.to_datetime(end_date).timestamp())
        
        for i, subreddit_name in enumerate(subreddits):
            try:
                self.logger.info(f"Collecting from r/{subreddit_name} ({i+1}/{len(subreddits)})")
                subreddit = self.reddit.subreddit(subreddit_name)
                subreddit_posts = []
                
                # Collect hot posts (popular content) - increased limit
                for submission in subreddit.hot(limit=2000):
                    if start_timestamp <= submission.created_utc <= end_timestamp:
                        subreddit_posts.append({
                            'date': pd.to_datetime(submission.created_utc, unit='s'),
                            'subreddit': subreddit_name,
                            'title': submission.title,
                            'text': submission.selftext,
                            'score': submission.score,
                            'num_comments': submission.num_comments,
                            'url': submission.url,
                            'upvote_ratio': getattr(submission, 'upvote_ratio', 0.5),
                            'gilded': getattr(submission, 'gilded', 0),
                            'post_type': 'hot'
                        })
                
                # Collect top posts (highly upvoted) - multiple time filters
                for time_filter in ['year', 'month', 'week']:
                    for submission in subreddit.top(time_filter=time_filter, limit=1000):
                        if start_timestamp <= submission.created_utc <= end_timestamp:
                            subreddit_posts.append({
                                'date': pd.to_datetime(submission.created_utc, unit='s'),
                                'subreddit': subreddit_name,
                                'title': submission.title,
                                'text': submission.selftext,
                                'score': submission.score,
                                'num_comments': submission.num_comments,
                                'url': submission.url,
                                'upvote_ratio': getattr(submission, 'upvote_ratio', 0.5),
                                'gilded': getattr(submission, 'gilded', 0),
                                'post_type': f'top_{time_filter}'
                            })
                
                # Collect new posts (recent content)
                for submission in subreddit.new(limit=500):
                    if start_timestamp <= submission.created_utc <= end_timestamp:
                        subreddit_posts.append({
                            'date': pd.to_datetime(submission.created_utc, unit='s'),
                            'subreddit': subreddit_name,
                            'title': submission.title,
                            'text': submission.selftext,
                            'score': submission.score,
                            'num_comments': submission.num_comments,
                            'url': submission.url,
                            'upvote_ratio': getattr(submission, 'upvote_ratio', 0.5),
                            'gilded': getattr(submission, 'gilded', 0),
                            'post_type': 'new'
                        })
                
                # Remove duplicates based on title and date
                unique_posts = []
                seen = set()
                for post in subreddit_posts:
                    key = (post['title'], post['date'].date())
                    if key not in seen:
                        seen.add(key)
                        unique_posts.append(post)
                
                self.logger.info(f"Collected {len(unique_posts)} unique posts from r/{subreddit_name}")
                all_posts.extend(unique_posts)
                
                # Rate limiting to avoid API limits
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error collecting Reddit data from r/{subreddit_name}: {e}")
        
        if all_posts:
            reddit_df = pd.DataFrame(all_posts)
            reddit_df = reddit_df.drop_duplicates(subset=['title', 'text'])
            reddit_df = filter_spam_posts(reddit_df, 'text')
            
            # Combine title and text
            reddit_df['combined_text'] = reddit_df['title'] + ' ' + reddit_df['text'].fillna('')
            
            # Add engagement metrics for research analysis
            reddit_df['engagement_score'] = reddit_df['score'] + (reddit_df['num_comments'] * 2) + (reddit_df['gilded'] * 10)
            reddit_df['quality_score'] = reddit_df['upvote_ratio'] * reddit_df['score']
            reddit_df['date'] = pd.to_datetime(reddit_df['date']).dt.date
            
            # Filter out very short posts (likely spam or low quality)
            reddit_df = reddit_df[reddit_df['combined_text'].str.len() > 50]
            
            # Sort by date
            reddit_df = reddit_df.sort_values('date').reset_index(drop=True)
            
            # Save Reddit data
            output_path = f"data/raw/reddit_{start_date}_{end_date}.csv"
            ensure_directory(os.path.dirname(output_path))
            reddit_df.to_csv(output_path, index=False)
            
            self.logger.info(f"Total Reddit posts collected: {len(reddit_df)}")
            self.logger.info(f"Date range: {reddit_df['date'].min()} to {reddit_df['date'].max()}")
            self.logger.info(f"Subreddits covered: {reddit_df['subreddit'].nunique()}")
            self.logger.info(f"Average posts per day: {len(reddit_df) / max(1, (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days):.1f}")
            
            return reddit_df
        else:
            self.logger.warning("No Reddit data collected, using sample data")
            return self._get_sample_reddit_data()
    
    def collect_twitter_data(self, start_date: str, end_date: str, 
                           keywords: List[str] = None) -> pd.DataFrame:
        """Collect Twitter data using search queries."""
        if not self.twitter_client:
            self.logger.warning("Twitter API not available, using sample data")
            return self._get_sample_twitter_data()
        
        if keywords is None:
            keywords = ['market', 'stocks', 'economy', 'finance', 'trading', 'investment']
        
        self.logger.info(f"Collecting Twitter data from {start_date} to {end_date}")
        
        all_tweets = []
        
        for keyword in keywords:
            try:
                # Search for tweets
                tweets = tweepy.Paginator(
                    self.twitter_client.search_recent_tweets,
                    query=f"{keyword} -is:retweet lang:en",
                    tweet_fields=['created_at', 'public_metrics', 'context_annotations'],
                    max_results=100
                ).flatten(limit=1000)
                
                for tweet in tweets:
                    all_tweets.append({
                        'date': tweet.created_at,
                        'text': tweet.text,
                        'retweet_count': tweet.public_metrics['retweet_count'],
                        'like_count': tweet.public_metrics['like_count'],
                        'reply_count': tweet.public_metrics['reply_count'],
                        'keyword': keyword
                    })
                
                self.logger.info(f"Collected {len([t for t in all_tweets if t['keyword'] == keyword])} tweets for keyword '{keyword}'")
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                self.logger.error(f"Error collecting Twitter data for keyword '{keyword}': {e}")
        
        if all_tweets:
            twitter_df = pd.DataFrame(all_tweets)
            twitter_df = filter_spam_posts(twitter_df, 'text')
            
            # Save Twitter data
            output_path = f"data/raw/twitter_{start_date}_{end_date}.csv"
            ensure_directory(os.path.dirname(output_path))
            twitter_df.to_csv(output_path, index=False)
            
            return twitter_df
        else:
            self.logger.warning("No Twitter data collected, using sample data")
            return self._get_sample_twitter_data()
    
    def collect_all_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Collect all data sources."""
        self.logger.info(f"Starting data collection from {start_date} to {end_date}")
        
        results = {}
        
        # Collect market data
        results['market'] = self.collect_market_data(start_date, end_date)
        
        # Collect news data
        results['news'] = self.collect_gdelt_news(start_date, end_date)
        
        # Collect social media data
        results['reddit'] = self.collect_reddit_data(start_date, end_date)
        results['twitter'] = self.collect_twitter_data(start_date, end_date)
        
        self.logger.info("Data collection completed")
        return results
    
    def _get_sample_news_data(self) -> pd.DataFrame:
        """Generate sample news data for testing."""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        sources = ['Reuters', 'Bloomberg', 'CNBC', 'WSJ', 'Financial Times']
        
        headlines = [
            "Market volatility spikes as investors react to economic uncertainty",
            "Fed signals potential rate hikes amid inflation concerns", 
            "Stocks tumble on recession fears and geopolitical tensions",
            "Corporate earnings disappoint as supply chain issues persist",
            "Central banks coordinate response to market stress",
            "Tech stocks rally on strong earnings and AI optimism",
            "Energy sector gains on oil price recovery",
            "Healthcare stocks mixed amid regulatory uncertainty",
            "Financial services show resilience in challenging market",
            "Consumer goods face headwinds from inflation"
        ]
        
        sample_data = []
        for date in dates:
            for _ in range(np.random.randint(5, 15)):  # 5-15 articles per day
                sample_data.append({
                    'date': date,
                    'source': np.random.choice(sources),
                    'headline': np.random.choice(headlines),
                    'url': f"https://example.com/news/{date.strftime('%Y%m%d')}_{np.random.randint(1000, 9999)}",
                    'text': np.random.choice(headlines) + " " + "Additional context about market conditions and investor sentiment.",
                    'tone': np.random.uniform(-10, 10),
                    'goldstein_scale': np.random.uniform(-10, 10),
                    'actor1': np.random.choice(['Federal Reserve', 'Stock Market', 'Investors', 'Banks', 'Economy']),
                    'actor2': np.random.choice(['Government', 'Market', 'Consumers', 'Business', 'Financial Sector']),
                    'event_code': np.random.choice(['14', '15', '16', '17', '18'])
                })
        
        return pd.DataFrame(sample_data)
    
    def _get_sample_reddit_data(self) -> pd.DataFrame:
        """Generate sample Reddit data for testing."""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        subreddits = ['investing', 'wallstreetbets', 'stocks', 'SecurityAnalysis']
        
        titles = [
            "What do you think about the current market conditions?",
            "My portfolio is down 20% this month, should I sell?",
            "Earnings season is coming up, which stocks to watch?",
            "Fed meeting next week - what to expect?",
            "Anyone else worried about inflation?",
            "Tech stocks are looking oversold, time to buy?",
            "Energy sector analysis - oil prices rising",
            "Healthcare stocks under pressure from regulations",
            "Financial services showing mixed signals",
            "Consumer spending trends and market impact"
        ]
        
        sample_data = []
        for date in dates:
            for _ in range(np.random.randint(10, 30)):  # 10-30 posts per day
                sample_data.append({
                    'date': date,
                    'subreddit': np.random.choice(subreddits),
                    'title': np.random.choice(titles),
                    'text': "Discussion about market trends and investment strategies. " + 
                           "What are your thoughts on the current economic environment?",
                    'score': np.random.randint(0, 100),
                    'num_comments': np.random.randint(0, 50),
                    'url': f"https://reddit.com/r/{np.random.choice(subreddits)}/post_{np.random.randint(1000, 9999)}"
                })
        
        df = pd.DataFrame(sample_data)
        df['combined_text'] = df['title'] + ' ' + df['text']
        return df
    
    def _get_sample_twitter_data(self) -> pd.DataFrame:
        """Generate sample Twitter data for testing."""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        keywords = ['market', 'stocks', 'economy', 'finance', 'trading']
        
        tweets = [
            "Market looking volatile today #stocks #trading",
            "Fed meeting next week could move markets significantly",
            "Earnings season starting - time to watch the numbers",
            "Inflation concerns weighing on investor sentiment",
            "Tech sector showing signs of recovery",
            "Energy stocks benefiting from oil price gains",
            "Healthcare sector facing regulatory headwinds",
            "Financial services adapting to new market conditions",
            "Consumer spending patterns shifting post-pandemic",
            "Global economic uncertainty affecting all sectors"
        ]
        
        sample_data = []
        for date in dates:
            for _ in range(np.random.randint(20, 50)):  # 20-50 tweets per day
                sample_data.append({
                    'date': date,
                    'text': np.random.choice(tweets),
                    'retweet_count': np.random.randint(0, 100),
                    'like_count': np.random.randint(0, 500),
                    'reply_count': np.random.randint(0, 20),
                    'keyword': np.random.choice(keywords)
                })
        
        df = pd.DataFrame(sample_data)
        df['combined_text'] = df['text']  # For Twitter, text is the main content
        return df
