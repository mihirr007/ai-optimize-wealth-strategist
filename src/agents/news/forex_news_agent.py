"""
ForexNewsAPI Agent - Macroeconomic and forex-specific news stream
Requires API key from https://forexnewsapi.com/
"""

import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ..market_data.base_agent import MarketDataAgent
import os

class ForexNewsAgent(MarketDataAgent):
    """ForexNewsAPI data agent for forex and macroeconomic news"""
    
    def __init__(self):
        super().__init__("FOREX_NEWS")
        self.api_key = os.getenv("FOREX_NEWS_API_KEY")
        self.base_url = "https://forexnewsapi.com/api/v1"
    
    def get_latest_forex_news(self, limit: int = 20) -> Dict[str, Any]:
        """Get latest forex news"""
        if not self.api_key:
            return {"error": "ForexNewsAPI key not configured"}
        
        cache_key = f"forex_news_latest_{limit}"
        cached_data = self._cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            params = {
                "api_key": self.api_key,
                "limit": limit
            }
            data = self._make_request(f"{self.base_url}/news", params=params)
            
            if "data" in data and isinstance(data["data"], list):
                articles = []
                for article in data["data"]:
                    processed_article = {
                        "id": article.get("id", ""),
                        "title": article.get("title", ""),
                        "summary": article.get("summary", ""),
                        "url": article.get("url", ""),
                        "source": article.get("source", ""),
                        "published_at": article.get("published_at", ""),
                        "currency": article.get("currency", ""),
                        "impact": article.get("impact", "medium"),
                        "sentiment": self._analyze_sentiment(article.get("title", "") + " " + article.get("summary", "")),
                        "last_updated": datetime.now().isoformat()
                    }
                    articles.append(processed_article)
                
                news_data = {
                    "total_results": len(articles),
                    "articles": articles,
                    "sentiment_summary": self._calculate_sentiment_summary(articles),
                    "last_updated": datetime.now().isoformat()
                }
            else:
                news_data = {"error": "No forex news data available"}
            
            self._cache_set(cache_key, news_data)
            return news_data
            
        except Exception as e:
            print(f"Error fetching ForexNewsAPI data: {e}")
            return {"error": str(e)}
    
    def get_currency_news(self, currency: str, limit: int = 10) -> Dict[str, Any]:
        """Get news for specific currency"""
        if not self.api_key:
            return {"error": "ForexNewsAPI key not configured", "currency": currency}
        
        cache_key = f"forex_news_currency_{currency}_{limit}"
        cached_data = self._cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            params = {
                "api_key": self.api_key,
                "currency": currency,
                "limit": limit
            }
            data = self._make_request(f"{self.base_url}/news/currency", params=params)
            
            if "data" in data and isinstance(data["data"], list):
                articles = []
                for article in data["data"]:
                    processed_article = {
                        "id": article.get("id", ""),
                        "title": article.get("title", ""),
                        "summary": article.get("summary", ""),
                        "url": article.get("url", ""),
                        "source": article.get("source", ""),
                        "published_at": article.get("published_at", ""),
                        "currency": article.get("currency", currency),
                        "impact": article.get("impact", "medium"),
                        "sentiment": self._analyze_sentiment(article.get("title", "") + " " + article.get("summary", "")),
                        "last_updated": datetime.now().isoformat()
                    }
                    articles.append(processed_article)
                
                currency_news = {
                    "currency": currency,
                    "total_results": len(articles),
                    "articles": articles,
                    "sentiment_summary": self._calculate_sentiment_summary(articles),
                    "last_updated": datetime.now().isoformat()
                }
            else:
                currency_news = {"error": "No currency news available", "currency": currency}
            
            self._cache_set(cache_key, currency_news)
            return currency_news
            
        except Exception as e:
            print(f"Error fetching ForexNewsAPI currency news for {currency}: {e}")
            return {"error": str(e), "currency": currency}
    
    def get_macro_news(self, limit: int = 15) -> Dict[str, Any]:
        """Get macroeconomic news"""
        if not self.api_key:
            return {"error": "ForexNewsAPI key not configured"}
        
        cache_key = f"forex_news_macro_{limit}"
        cached_data = self._cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            params = {
                "api_key": self.api_key,
                "category": "macroeconomic",
                "limit": limit
            }
            data = self._make_request(f"{self.base_url}/news/category", params=params)
            
            if "data" in data and isinstance(data["data"], list):
                articles = []
                for article in data["data"]:
                    processed_article = {
                        "id": article.get("id", ""),
                        "title": article.get("title", ""),
                        "summary": article.get("summary", ""),
                        "url": article.get("url", ""),
                        "source": article.get("source", ""),
                        "published_at": article.get("published_at", ""),
                        "category": article.get("category", "macroeconomic"),
                        "impact": article.get("impact", "medium"),
                        "sentiment": self._analyze_sentiment(article.get("title", "") + " " + article.get("summary", "")),
                        "last_updated": datetime.now().isoformat()
                    }
                    articles.append(processed_article)
                
                macro_news = {
                    "category": "macroeconomic",
                    "total_results": len(articles),
                    "articles": articles,
                    "sentiment_summary": self._calculate_sentiment_summary(articles),
                    "last_updated": datetime.now().isoformat()
                }
            else:
                macro_news = {"error": "No macroeconomic news available"}
            
            self._cache_set(cache_key, macro_news)
            return macro_news
            
        except Exception as e:
            print(f"Error fetching ForexNewsAPI macro news: {e}")
            return {"error": str(e)}
    
    def get_forex_sentiment(self) -> Dict[str, Any]:
        """Get overall forex market sentiment"""
        try:
            # Get general forex news
            forex_news = self.get_latest_forex_news(30)
            
            # Get major currency news
            major_currencies = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD"]
            currency_news = []
            
            for currency in major_currencies:
                currency_data = self.get_currency_news(currency, 5)
                if "articles" in currency_data:
                    currency_news.extend(currency_data["articles"])
            
            # Get macro news
            macro_news = self.get_macro_news(20)
            
            # Combine all articles
            all_articles = []
            if "articles" in forex_news:
                all_articles.extend(forex_news["articles"])
            all_articles.extend(currency_news)
            if "articles" in macro_news:
                all_articles.extend(macro_news["articles"])
            
            sentiment_data = {
                "overall_sentiment": self._calculate_overall_sentiment(all_articles),
                "forex_news_sentiment": forex_news.get("sentiment_summary", {}),
                "currency_news_count": len(currency_news),
                "macro_news_sentiment": macro_news.get("sentiment_summary", {}),
                "total_articles": len(all_articles),
                "last_updated": datetime.now().isoformat()
            }
            
            return sentiment_data
            
        except Exception as e:
            print(f"Error calculating ForexNewsAPI sentiment: {e}")
            return {"error": str(e)}
    
    def _analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis based on keywords"""
        if not text:
            return "neutral"
        
        text_lower = text.lower()
        
        # Positive keywords for forex
        positive_words = [
            "surge", "jump", "rise", "gain", "up", "positive", "growth", "strong",
            "bullish", "rally", "recovery", "higher", "better", "strengthen",
            "positive", "increase", "improve", "boost", "support"
        ]
        
        # Negative keywords for forex
        negative_words = [
            "fall", "drop", "decline", "loss", "down", "negative", "weak", "bearish",
            "crash", "plunge", "concern", "risk", "worry", "fear", "sell-off",
            "negative", "decrease", "lower", "worse", "weaken", "pressure"
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
        """Calculate overall forex sentiment"""
        if not articles:
            return "neutral"
        
        summary = self._calculate_sentiment_summary(articles)
        return summary["overall"]
    
    def get_data(self, **kwargs) -> Dict[str, Any]:
        """Main data fetching method"""
        data_type = kwargs.get("data_type", "latest")
        currency = kwargs.get("currency", "")
        
        if data_type == "currency":
            return self.get_currency_news(currency)
        elif data_type == "macro":
            return self.get_macro_news()
        elif data_type == "sentiment":
            return self.get_forex_sentiment()
        else:
            return self.get_latest_forex_news() 