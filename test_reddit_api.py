#!/usr/bin/env python3
"""
Test Reddit API Integration
This script tests the Reddit API connection and data collection
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from reddit_api import RedditAPIClient
from data_validation import DataValidator
import pandas as pd
import json

def test_reddit_api():
    """Test Reddit API connection and data collection"""
    print("🔗 Testing Reddit API Integration...")
    
    try:
        # Initialize Reddit API client
        client = RedditAPIClient()
        print("✅ Reddit API client initialized successfully")
        
        # Test subreddit data collection
        print("📊 Collecting data from r/investing...")
        posts = client.collect_subreddit_data('investing', limit=10)
        print(f"✅ Collected {len(posts)} posts from r/investing")
        
        # Display sample posts
        if posts:
            print("\n📝 Sample posts:")
            for i, post in enumerate(posts[:3]):
                print(f"  {i+1}. {post.title[:50]}... (Score: {post.score})")
        
        # Test data validation
        print("\n🧪 Testing data validation...")
        validator = DataValidator()
        
        # Convert posts to DataFrame
        posts_data = [post.to_dict() for post in posts]
        df = pd.DataFrame(posts_data)
        
        # Validate data
        report = validator.validate_data(df, 'reddit')
        print(f"✅ Data validation completed. Overall score: {report.overall_score:.2f}")
        print(f"   Quality level: {report.quality_level.value}")
        print(f"   Missing data: {report.missing_data_percentage:.2f}%")
        
        # Test multiple subreddits
        print("\n📊 Testing multiple subreddits...")
        subreddits = ['investing', 'stocks', 'wallstreetbets']
        all_posts = client.collect_multiple_subreddits(subreddits, limit_per_subreddit=5)
        
        total_posts = sum(len(posts) for posts in all_posts.values())
        print(f"✅ Collected {total_posts} posts from {len(subreddits)} subreddits")
        
        # Display summary by subreddit
        for subreddit, posts in all_posts.items():
            print(f"   r/{subreddit}: {len(posts)} posts")
        
        # Test rate limiting
        print("\n⏱️  Testing rate limiting...")
        remaining_requests = client.rate_limit_tracker.get_remaining_requests()
        print(f"✅ Remaining requests in current window: {remaining_requests}")
        
        print("\n🎉 Reddit API integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Reddit API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_quality():
    """Test data quality validation"""
    print("\n🧪 Testing Data Quality Validation...")
    
    try:
        # Create test data
        test_data = pd.DataFrame({
            'title': [
                'Great investment opportunity in tech stocks',
                'Market analysis shows bullish trends',
                'Short post',
                'Another great investment opportunity with detailed analysis',
                'Market volatility concerns investors worldwide'
            ],
            'subreddit': ['investing', 'stocks', 'investing', 'stocks', 'investing'],
            'score': [100, 250, 5, 300, 150],
            'created_utc': [1640995200, 1640995300, 1640995400, 1640995500, 1640995600],
            'num_comments': [10, 25, 1, 30, 15],
            'author': ['user1', 'user2', 'user3', 'user4', 'user5'],
            'is_self': [True, True, False, True, True],
            'upvote_ratio': [0.85, 0.92, 0.45, 0.88, 0.75]
        })
        
        # Test validation
        validator = DataValidator()
        report = validator.validate_data(test_data, 'reddit')
        
        print(f"✅ Data quality validation completed")
        print(f"   Overall score: {report.overall_score:.2f}")
        print(f"   Quality level: {report.quality_level.value}")
        print(f"   Missing data: {report.missing_data_percentage:.2f}%")
        
        # Display validation results
        print("\n📋 Validation Results:")
        for result in report.validation_results:
            status = "✅" if result.is_valid else "❌"
            print(f"   {status} {result.message} (Score: {result.score:.2f})")
        
        # Display recommendations
        if report.recommendations:
            print("\n💡 Recommendations:")
            for rec in report.recommendations:
                print(f"   - {rec}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data quality test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 BRI Reddit API Integration Test")
    print("=" * 50)
    
    # Test Reddit API
    reddit_success = test_reddit_api()
    
    # Test data quality
    quality_success = test_data_quality()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   Reddit API: {'✅ PASS' if reddit_success else '❌ FAIL'}")
    print(f"   Data Quality: {'✅ PASS' if quality_success else '❌ FAIL'}")
    
    if reddit_success and quality_success:
        print("\n🎉 All tests passed! Your Reddit API integration is working correctly.")
        print("\n📋 Next Steps:")
        print("   1. Run the full pipeline: python refactored_bri_pipeline.py")
        print("   2. Start the web app: python app.py")
        print("   3. Access the API: http://localhost:5000")
        print("   4. View monitoring: http://localhost:3000")
    else:
        print("\n❌ Some tests failed. Please check the configuration and try again.")
        print("   - Verify Reddit API credentials in config.yaml")
        print("   - Check internet connection")
        print("   - Ensure all dependencies are installed")

if __name__ == "__main__":
    main()
