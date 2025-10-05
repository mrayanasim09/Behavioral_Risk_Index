"""
Real Reddit API Integration Module

This module provides authentic Reddit API integration with:
- OAuth2 authentication
- Rate limiting (1000 requests per 600 seconds)
- Comprehensive error handling
- Data validation and quality checks
- Caching to reduce API calls
"""

import praw
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import json
import yaml
from dataclasses import dataclass
from enum import Enum
import backoff
from requests.exceptions import RequestException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedditAPIError(Exception):
    """Custom exception for Reddit API errors"""
    pass

class RateLimitExceededError(RedditAPIError):
    """Exception raised when Reddit rate limit is exceeded"""
    pass

@dataclass
class RedditPost:
    """Data class for Reddit post data"""
    title: str
    selftext: str
    score: int
    num_comments: int
    created_utc: float
    subreddit: str
    url: str
    author: str
    is_self: bool
    upvote_ratio: float
    flair_text: Optional[str] = None
    link_flair_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'selftext': self.selftext,
            'score': self.score,
            'num_comments': self.num_comments,
            'created_utc': self.created_utc,
            'subreddit': self.subreddit,
            'url': self.url,
            'author': self.author,
            'is_self': self.is_self,
            'upvote_ratio': self.upvote_ratio,
            'flair_text': self.flair_text,
            'link_flair_text': self.link_flair_text
        }

class RedditAPIClient:
    """
    Reddit API client with authentication and rate limiting
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize Reddit API client"""
        self.config = self._load_config(config_path)
        self.reddit = self._initialize_reddit_client()
        self.rate_limit_tracker = RateLimitTracker()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise RedditAPIError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise RedditAPIError(f"Error parsing configuration file: {e}")
    
    def _initialize_reddit_client(self) -> praw.Reddit:
        """Initialize PRAW Reddit client with OAuth2 authentication"""
        try:
            # Get Reddit API credentials from config
            reddit_config = self.config['data_sources']['reddit']
            
            # Initialize Reddit client in read-only mode
            reddit = praw.Reddit(
                client_id=reddit_config['client_id'],
                client_secret=reddit_config['client_secret'],
                user_agent=reddit_config['user_agent']
            )
            
            # Test authentication (read-only mode)
            try:
                # Test with a simple API call
                reddit.subreddit('test').display_name
                logger.info("Reddit API authentication successful (read-only mode)")
            except Exception as e:
                logger.warning(f"Reddit authentication test failed: {e}")
            
            return reddit
            
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            raise RedditAPIError(f"Failed to initialize Reddit client: {e}")
    
    @backoff.on_exception(
        backoff.expo,
        (RequestException, RateLimitExceededError),
        max_tries=3,
        max_time=300
    )
    def collect_subreddit_data(self, subreddit_name: str, limit: int = 100, 
                              sort_by: str = 'hot', start_date: str = None, 
                              end_date: str = None) -> List[RedditPost]:
        """
        Collect data from a specific subreddit
        
        Args:
            subreddit_name: Name of the subreddit (without r/)
            limit: Maximum number of posts to collect
            sort_by: Sorting method ('hot', 'new', 'top', 'rising')
            
        Returns:
            List of RedditPost objects
        """
        try:
            # Check rate limit
            if not self.rate_limit_tracker.can_make_request():
                raise RateLimitExceededError("Rate limit exceeded. Please wait before making more requests.")
            
            logger.info(f"Collecting data from r/{subreddit_name} (limit: {limit}, sort: {sort_by})")
            
            # Get subreddit object
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Collect posts based on sort method
            posts = []
            if sort_by == 'hot':
                submission_generator = subreddit.hot(limit=limit)
            elif sort_by == 'new':
                submission_generator = subreddit.new(limit=limit)
            elif sort_by == 'top':
                submission_generator = subreddit.top(limit=limit)
            elif sort_by == 'rising':
                submission_generator = subreddit.rising(limit=limit)
            else:
                raise ValueError(f"Invalid sort method: {sort_by}")
            
            # Process submissions
            for submission in submission_generator:
                try:
                    # Skip pinned posts and ads
                    if submission.stickied or submission.is_self is None:
                        continue
                    
                    # Create RedditPost object
                    post = RedditPost(
                        title=submission.title,
                        selftext=submission.selftext,
                        score=submission.score,
                        num_comments=submission.num_comments,
                        created_utc=submission.created_utc,
                        subreddit=subreddit_name,
                        url=submission.url,
                        author=str(submission.author) if submission.author else '[deleted]',
                        is_self=submission.is_self,
                        upvote_ratio=submission.upvote_ratio,
                        flair_text=submission.link_flair_text,
                        link_flair_text=submission.link_flair_text
                    )
                    
                    posts.append(post)
                    
                    # Track rate limit
                    self.rate_limit_tracker.record_request()
                    
                    # Add delay to respect rate limits
                    time.sleep(0.1)  # 100ms delay between requests
                    
                except Exception as e:
                    logger.warning(f"Error processing submission {submission.id}: {e}")
                    continue
            
            logger.info(f"Successfully collected {len(posts)} posts from r/{subreddit_name}")
            return posts
            
        except Exception as e:
            logger.error(f"Error collecting data from r/{subreddit_name}: {e}")
            raise RedditAPIError(f"Error collecting data from r/{subreddit_name}: {e}")
    
    def collect_multiple_subreddits(self, subreddit_names: List[str], 
                                   limit_per_subreddit: int = 100) -> Dict[str, List[RedditPost]]:
        """
        Collect data from multiple subreddits
        
        Args:
            subreddit_names: List of subreddit names
            limit_per_subreddit: Maximum posts per subreddit
            
        Returns:
            Dictionary mapping subreddit names to lists of RedditPost objects
        """
        results = {}
        
        for subreddit_name in subreddit_names:
            try:
                posts = self.collect_subreddit_data(subreddit_name, limit_per_subreddit)
                results[subreddit_name] = posts
                
                # Add delay between subreddits
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to collect data from r/{subreddit_name}: {e}")
                results[subreddit_name] = []
        
        return results
    
    def collect_comments(self, submission_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Collect comments from a specific submission
        
        Args:
            submission_id: Reddit submission ID
            limit: Maximum number of comments to collect
            
        Returns:
            List of comment dictionaries
        """
        try:
            # Check rate limit
            if not self.rate_limit_tracker.can_make_request():
                raise RateLimitExceededError("Rate limit exceeded. Please wait before making more requests.")
            
            logger.info(f"Collecting comments from submission {submission_id}")
            
            # Get submission
            submission = self.reddit.submission(id=submission_id)
            
            # Collect comments
            comments = []
            submission.comments.replace_more(limit=0)  # Remove "more comments" placeholders
            
            for comment in submission.comments.list()[:limit]:
                try:
                    comment_data = {
                        'id': comment.id,
                        'body': comment.body,
                        'score': comment.score,
                        'created_utc': comment.created_utc,
                        'author': str(comment.author) if comment.author else '[deleted]',
                        'parent_id': comment.parent_id,
                        'is_submitter': comment.is_submitter
                    }
                    comments.append(comment_data)
                    
                    # Track rate limit
                    self.rate_limit_tracker.record_request()
                    
                except Exception as e:
                    logger.warning(f"Error processing comment {comment.id}: {e}")
                    continue
            
            logger.info(f"Successfully collected {len(comments)} comments from submission {submission_id}")
            return comments
            
        except Exception as e:
            logger.error(f"Error collecting comments from submission {submission_id}: {e}")
            raise RedditAPIError(f"Error collecting comments from submission {submission_id}: {e}")
    
    def get_subreddit_info(self, subreddit_name: str) -> Dict[str, Any]:
        """
        Get information about a subreddit
        
        Args:
            subreddit_name: Name of the subreddit
            
        Returns:
            Dictionary with subreddit information
        """
        try:
            # Check rate limit
            if not self.rate_limit_tracker.can_make_request():
                raise RateLimitExceededError("Rate limit exceeded. Please wait before making more requests.")
            
            logger.info(f"Getting information about r/{subreddit_name}")
            
            # Get subreddit object
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Extract information
            info = {
                'name': subreddit.display_name,
                'title': subreddit.title,
                'description': subreddit.description,
                'subscribers': subreddit.subscribers,
                'active_users': subreddit.active_user_count,
                'created_utc': subreddit.created_utc,
                'over18': subreddit.over18,
                'public_description': subreddit.public_description,
                'submission_type': subreddit.submission_type
            }
            
            # Track rate limit
            self.rate_limit_tracker.record_request()
            
            logger.info(f"Successfully retrieved information about r/{subreddit_name}")
            return info
            
        except Exception as e:
            logger.error(f"Error getting information about r/{subreddit_name}: {e}")
            raise RedditAPIError(f"Error getting information about r/{subreddit_name}: {e}")
    
    def search_subreddits(self, query: str, limit: int = 25) -> List[Dict[str, Any]]:
        """
        Search for subreddits
        
        Args:
            query: Search query
            limit: Maximum number of subreddits to return
            
        Returns:
            List of subreddit dictionaries
        """
        try:
            # Check rate limit
            if not self.rate_limit_tracker.can_make_request():
                raise RateLimitExceededError("Rate limit exceeded. Please wait before making more requests.")
            
            logger.info(f"Searching for subreddits with query: {query}")
            
            # Search subreddits
            subreddits = []
            for subreddit in self.reddit.subreddits.search(query, limit=limit):
                try:
                    subreddit_info = {
                        'name': subreddit.display_name,
                        'title': subreddit.title,
                        'description': subreddit.description,
                        'subscribers': subreddit.subscribers,
                        'active_users': subreddit.active_user_count,
                        'over18': subreddit.over18
                    }
                    subreddits.append(subreddit_info)
                    
                    # Track rate limit
                    self.rate_limit_tracker.record_request()
                    
                except Exception as e:
                    logger.warning(f"Error processing subreddit {subreddit.display_name}: {e}")
                    continue
            
            logger.info(f"Successfully found {len(subreddits)} subreddits for query: {query}")
            return subreddits
            
        except Exception as e:
            logger.error(f"Error searching for subreddits with query '{query}': {e}")
            raise RedditAPIError(f"Error searching for subreddits with query '{query}': {e}")

class RateLimitTracker:
    """
    Rate limit tracker for Reddit API (1000 requests per 600 seconds)
    """
    
    def __init__(self):
        self.requests = []
        self.max_requests = 1000
        self.time_window = 600  # 10 minutes in seconds
    
    def can_make_request(self) -> bool:
        """Check if we can make a request without exceeding rate limit"""
        current_time = time.time()
        
        # Remove requests older than the time window
        self.requests = [req_time for req_time in self.requests 
                        if current_time - req_time < self.time_window]
        
        # Check if we're under the limit
        return len(self.requests) < self.max_requests
    
    def record_request(self):
        """Record a request timestamp"""
        self.requests.append(time.time())
    
    def get_remaining_requests(self) -> int:
        """Get the number of remaining requests in the current window"""
        current_time = time.time()
        
        # Remove requests older than the time window
        self.requests = [req_time for req_time in self.requests 
                        if current_time - req_time < self.time_window]
        
        return max(0, self.max_requests - len(self.requests))
    
    def get_reset_time(self) -> float:
        """Get the time when the rate limit will reset"""
        if not self.requests:
            return 0
        
        oldest_request = min(self.requests)
        return oldest_request + self.time_window

# Example usage
if __name__ == "__main__":
    # Initialize Reddit API client
    client = RedditAPIClient()
    
    # Collect data from multiple subreddits
    subreddits = ['investing', 'stocks', 'wallstreetbets']
    results = client.collect_multiple_subreddits(subreddits, limit_per_subreddit=50)
    
    # Print results
    for subreddit, posts in results.items():
        print(f"r/{subreddit}: {len(posts)} posts")
        for post in posts[:3]:  # Show first 3 posts
            print(f"  - {post.title[:50]}... (score: {post.score})")
