"""
NewsAPI US Agent - U.S. business and market news headlines
Requires API key from https://newsapi.org/
"""

import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ..market_data.base_agent import MarketDataAgent
import os

class NewsAPIUSAgent(MarketDataAgent):
    """NewsAPI US agent for financial news and sentiment"""
    
    def __init__(self):
        super().__init__("NEWSAPI")
        self.api_key = os.getenv("NEWSAPI_US_KEY")
        self.base_url = "https://newsapi.org/v2"
    
    def get_latest_news(self, category: str = "business", page_size: int = 10) -> Dict[str, Any]:
        """Get latest US financial news"""
        if not self.api_key:
            return {"error": "NewsAPI US key not configured"}
        
        cache_key = f"newsapi_us_{category}_{page_size}"
        cached_data = self._cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            params = {
                "country": "us",
                "category": category,
                "pageSize": page_size,
                "apiKey": self.api_key
            }
            data = self._make_request(f"{self.base_url}/top-headlines", params=params)
            
            if "articles" in data:
                articles = data["articles"]
                processed_articles = []
                
                for article in articles:
                    processed_article = {
                        "title": article.get("title", ""),
                        "description": article.get("description", ""),
                        "url": article.get("url", ""),
                        "published_at": article.get("publishedAt", ""),
                        "source": article.get("source", {}).get("name", ""),
                        "content": article.get("content", ""),
                        "sentiment": self._analyze_sentiment((article.get("title", "") or "") + " " + (article.get("description", "") or ""))
                    }
                    processed_articles.append(processed_article)
                
                news_data = {
                    "category": category,
                    "total_results": data.get("totalResults", 0),
                    "articles": processed_articles,
                    "sentiment_summary": self._calculate_sentiment_summary(processed_articles),
                    "last_updated": datetime.now().isoformat()
                }
            else:
                news_data = {"error": "No news data available"}
            
            self._cache_set(cache_key, news_data)
            return news_data
            
        except Exception as e:
            print(f"Error fetching NewsAPI US data: {e}")
            return {"error": str(e)}
    
    def search_news(self, query: str, page_size: int = 10, from_date: str = None, to_date: str = None) -> Dict[str, Any]:
        """Search for specific financial news"""
        if not self.api_key:
            return {"error": "NewsAPI US key not configured"}
        
        cache_key = f"newsapi_us_search_{query}_{page_size}_{from_date}_{to_date}"
        cached_data = self._cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Use provided dates or default to 7 days ago
            if not from_date:
                from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            if not to_date:
                to_date = datetime.now().strftime("%Y-%m-%d")
            
            params = {
                "q": query,
                "from": from_date,
                "to": to_date,
                "language": "en",
                "sortBy": "popularity",  # Changed from publishedAt to popularity
                "pageSize": page_size,
                "apiKey": self.api_key
            }
            data = self._make_request(f"{self.base_url}/everything", params=params)
            
            if "articles" in data:
                articles = data["articles"]
                processed_articles = []
                
                for article in articles:
                    processed_article = {
                        "title": article.get("title", ""),
                        "description": article.get("description", ""),
                        "url": article.get("url", ""),
                        "published_at": article.get("publishedAt", ""),
                        "source": article.get("source", {}).get("name", ""),
                        "content": article.get("content", ""),
                        "sentiment": self._analyze_sentiment((article.get("title", "") or "") + " " + (article.get("description", "") or ""))
                    }
                    processed_articles.append(processed_article)
                
                search_data = {
                    "query": query,
                    "total_results": data.get("totalResults", 0),
                    "articles": processed_articles,
                    "sentiment_summary": self._calculate_sentiment_summary(processed_articles),
                    "last_updated": datetime.now().isoformat()
                }
            else:
                search_data = {"error": "No search results available"}
            
            self._cache_set(cache_key, search_data)
            return search_data
            
        except Exception as e:
            print(f"Error searching NewsAPI US data: {e}")
            return {"error": str(e)}
    
    def get_market_sentiment(self) -> Dict[str, Any]:
        """Get overall market sentiment from news"""
        try:
            # Get business news
            business_news = self.get_latest_news("business", 20)
            
            # Get technology news (often affects markets)
            tech_news = self.get_latest_news("technology", 10)
            
            # Combine and analyze sentiment
            all_articles = []
            if "articles" in business_news:
                all_articles.extend(business_news["articles"])
            if "articles" in tech_news:
                all_articles.extend(tech_news["articles"])
            
            sentiment_data = {
                "overall_sentiment": self._calculate_overall_sentiment(all_articles),
                "business_sentiment": business_news.get("sentiment_summary", {}),
                "tech_sentiment": tech_news.get("sentiment_summary", {}),
                "total_articles": len(all_articles),
                "last_updated": datetime.now().isoformat()
            }
            
            return sentiment_data
            
        except Exception as e:
            print(f"Error calculating market sentiment: {e}")
            return {"error": str(e)}
    
    def _analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis based on keywords"""
        if not text:
            return "neutral"
        
        text_lower = text.lower()
        
        # Positive keywords
        positive_words = [
            "surge", "jump", "rise", "gain", "up", "positive", "growth", "profit",
            "earnings", "beat", "exceed", "strong", "bullish", "rally", "recovery"
        ]
        
        # Negative keywords
        negative_words = [
            "fall", "drop", "decline", "loss", "down", "negative", "weak", "bearish",
            "crash", "plunge", "concern", "risk", "worry", "fear", "sell-off"
        ]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_sentiment_summary(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate sentiment summary from articles"""
        if not articles:
            return {"positive": 0, "negative": 0, "neutral": 0, "overall": "neutral"}
        
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        
        for article in articles:
            sentiment = article.get("sentiment", "neutral")
            sentiment_counts[sentiment] += 1
        
        total = len(articles)
        if sentiment_counts["positive"] > sentiment_counts["negative"]:
            overall = "positive"
        elif sentiment_counts["negative"] > sentiment_counts["positive"]:
            overall = "negative"
        else:
            overall = "neutral"
        
        return {
            "positive": sentiment_counts["positive"],
            "negative": sentiment_counts["negative"],
            "neutral": sentiment_counts["neutral"],
            "positive_percent": (sentiment_counts["positive"] / total) * 100,
            "negative_percent": (sentiment_counts["negative"] / total) * 100,
            "neutral_percent": (sentiment_counts["neutral"] / total) * 100,
            "overall": overall
        }
    
    def _calculate_overall_sentiment(self, articles: List[Dict[str, Any]]) -> str:
        """Calculate overall market sentiment"""
        if not articles:
            return "neutral"
        
        summary = self._calculate_sentiment_summary(articles)
        return summary["overall"]
    
    def get_data(self, **kwargs) -> Dict[str, Any]:
        """Main data fetching method"""
        data_type = kwargs.get("data_type", "latest")
        category = kwargs.get("category", "business")
        query = kwargs.get("query", "")
        
        if data_type == "search" and query:
            return self.search_news(query)
        elif data_type == "sentiment":
            return self.get_market_sentiment()
        else:
            return self.get_latest_news(category) 