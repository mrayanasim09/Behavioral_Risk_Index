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

from .utils import ensure_directory, filter_spam_posts, chunk_dates, save_with_metadata

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
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'BRI-Research-Bot/1.0')
        
        # Initialize API clients
        self._setup_reddit()
        self._setup_twitter()
        
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
    
    def collect_market_data(self, start_date: str, end_date: str, 
                          symbols: List[str] = None) -> pd.DataFrame:
        """Collect market data including S&P500, VIX, and other indices."""
        if symbols is None:
            symbols = ['^GSPC', '^VIX', '^TNX', '^FVX', '^TYX']  # S&P500, VIX, Treasury yields
        
        self.logger.info(f"Collecting market data from {start_date} to {end_date}")
        
        market_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if not data.empty:
                    # Calculate returns and realized volatility
                    data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
                    data['realized_vol'] = data['returns'].rolling(window=5).apply(
                        lambda x: np.sqrt(np.sum(x**2)), raw=True
                    )
                    
                    market_data[symbol] = data
                    self.logger.info(f"Collected {len(data)} records for {symbol}")
                else:
                    self.logger.warning(f"No data found for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Error collecting data for {symbol}: {e}")
        
        # Combine all market data
        if market_data:
            combined_data = pd.concat(market_data.values(), keys=market_data.keys(), axis=1)
            combined_data.columns = [f"{symbol}_{col}" for symbol, col in combined_data.columns]
            combined_data = combined_data.reset_index()
            
            # Save market data
            output_path = f"data/raw/market_{start_date}_{end_date}.csv"
            ensure_directory(os.path.dirname(output_path))
            combined_data.to_csv(output_path, index=False)
            
            return combined_data
        else:
            self.logger.error("No market data collected")
            return pd.DataFrame()
    
    def collect_gdelt_news(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Collect news data from GDELT API."""
        if not self.gdelt_api_key:
            self.logger.warning("GDELT API key not found, using sample data")
            return self._get_sample_news_data()
        
        self.logger.info(f"Collecting GDELT news from {start_date} to {end_date}")
        
        # GDELT API endpoint
        base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        
        all_news = []
        date_chunks = chunk_dates(pd.to_datetime(start_date), pd.to_datetime(end_date), 7)
        
        for chunk_start, chunk_end in date_chunks:
            try:
                params = {
                    'query': 'sourcecountry:US AND (economy OR finance OR market OR stock OR investment)',
                    'format': 'json',
                    'startdatetime': chunk_start.strftime('%Y%m%d%H%M%S'),
                    'enddatetime': chunk_end.strftime('%Y%m%d%H%M%S'),
                    'maxrecords': 250,
                    'sort': 'date'
                }
                
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if 'articles' in data:
                    for article in data['articles']:
                        all_news.append({
                            'date': pd.to_datetime(article.get('date', chunk_start)),
                            'source': article.get('source', 'Unknown'),
                            'headline': article.get('title', ''),
                            'url': article.get('url', ''),
                            'text': article.get('snippet', '')
                        })
                
                self.logger.info(f"Collected {len(data.get('articles', []))} articles for {chunk_start.date()}")
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                self.logger.error(f"Error collecting GDELT data for {chunk_start.date()}: {e}")
                time.sleep(5)
        
        if all_news:
            news_df = pd.DataFrame(all_news)
            news_df = news_df.dropna(subset=['headline'])
            news_df = news_df[news_df['headline'].str.len() > 10]  # Filter very short headlines
            
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
            subreddits = ['investing', 'wallstreetbets', 'stocks', 'SecurityAnalysis', 'ValueInvesting']
        
        self.logger.info(f"Collecting Reddit data from {start_date} to {end_date}")
        
        all_posts = []
        start_timestamp = int(pd.to_datetime(start_date).timestamp())
        end_timestamp = int(pd.to_datetime(end_date).timestamp())
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Collect hot posts
                for submission in subreddit.hot(limit=1000):
                    if start_timestamp <= submission.created_utc <= end_timestamp:
                        all_posts.append({
                            'date': pd.to_datetime(submission.created_utc, unit='s'),
                            'subreddit': subreddit_name,
                            'title': submission.title,
                            'text': submission.selftext,
                            'score': submission.score,
                            'num_comments': submission.num_comments,
                            'url': submission.url
                        })
                
                # Collect top posts
                for submission in subreddit.top(time_filter='month', limit=500):
                    if start_timestamp <= submission.created_utc <= end_timestamp:
                        all_posts.append({
                            'date': pd.to_datetime(submission.created_utc, unit='s'),
                            'subreddit': subreddit_name,
                            'title': submission.title,
                            'text': submission.selftext,
                            'score': submission.score,
                            'num_comments': submission.num_comments,
                            'url': submission.url
                        })
                
                self.logger.info(f"Collected {len([p for p in all_posts if p['subreddit'] == subreddit_name])} posts from r/{subreddit_name}")
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                self.logger.error(f"Error collecting Reddit data from r/{subreddit_name}: {e}")
        
        if all_posts:
            reddit_df = pd.DataFrame(all_posts)
            reddit_df = reddit_df.drop_duplicates(subset=['title', 'text'])
            reddit_df = filter_spam_posts(reddit_df, 'text')
            
            # Combine title and text
            reddit_df['combined_text'] = reddit_df['title'] + ' ' + reddit_df['text'].fillna('')
            
            # Save Reddit data
            output_path = f"data/raw/reddit_{start_date}_{end_date}.csv"
            ensure_directory(os.path.dirname(output_path))
            reddit_df.to_csv(output_path, index=False)
            
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
                    'text': np.random.choice(headlines) + " " + "Additional context about market conditions and investor sentiment."
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
        
        return pd.DataFrame(sample_data)
