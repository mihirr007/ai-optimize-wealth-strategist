"""
FXStreet Agent - High-frequency forex and rate-related headlines (JSON API)
Requires API key from https://www.fxstreet.com/
"""

import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ..market_data.base_agent import MarketDataAgent
import os

class FXStreetAgent(MarketDataAgent):
    """FXStreet data agent for high-frequency forex news"""
    
    def __init__(self):
        super().__init__("FXSTREET")
        self.api_key = os.getenv("FXSTREET_API_KEY")
        self.base_url = "https://www.fxstreet.com/api/v1"
    
    def get_latest_forex_news(self, limit: int = 25) -> Dict[str, Any]:
        """Get latest forex news from FXStreet"""
        if not self.api_key:
            return {"error": "FXStreet API key not configured"}
        
        cache_key = f"fxstreet_news_latest_{limit}"
        cached_data = self._cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            params = {
                "api_key": self.api_key,
                "limit": limit,
                "category": "forex"
            }
            data = self._make_request(f"{self.base_url}/news", params=params)
            
            if "articles" in data and isinstance(data["articles"], list):
                articles = []
                for article in data["articles"]:
                    processed_article = {
                        "id": article.get("id", ""),
                        "title": article.get("title", ""),
                        "summary": article.get("summary", ""),
                        "url": article.get("url", ""),
                        "source": article.get("source", "FXStreet"),
                        "published_at": article.get("published_at", ""),
                        "category": article.get("category", "forex"),
                        "currency_pair": article.get("currency_pair", ""),
                        "impact_level": article.get("impact_level", "medium"),
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
                news_data = {"error": "No FXStreet news data available"}
            
            self._cache_set(cache_key, news_data)
            return news_data
            
        except Exception as e:
            print(f"Error fetching FXStreet data: {e}")
            return {"error": str(e)}
    
    def get_currency_pair_news(self, currency_pair: str, limit: int = 10) -> Dict[str, Any]:
        """Get news for specific currency pair"""
        if not self.api_key:
            return {"error": "FXStreet API key not configured", "currency_pair": currency_pair}
        
        cache_key = f"fxstreet_pair_{currency_pair}_{limit}"
        cached_data = self._cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            params = {
                "api_key": self.api_key,
                "currency_pair": currency_pair,
                "limit": limit
            }
            data = self._make_request(f"{self.base_url}/news/pair", params=params)
            
            if "articles" in data and isinstance(data["articles"], list):
                articles = []
                for article in data["articles"]:
                    processed_article = {
                        "id": article.get("id", ""),
                        "title": article.get("title", ""),
                        "summary": article.get("summary", ""),
                        "url": article.get("url", ""),
                        "source": article.get("source", "FXStreet"),
                        "published_at": article.get("published_at", ""),
                        "currency_pair": article.get("currency_pair", currency_pair),
                        "impact_level": article.get("impact_level", "medium"),
                        "sentiment": self._analyze_sentiment(article.get("title", "") + " " + article.get("summary", "")),
                        "last_updated": datetime.now().isoformat()
                    }
                    articles.append(processed_article)
                
                pair_news = {
                    "currency_pair": currency_pair,
                    "total_results": len(articles),
                    "articles": articles,
                    "sentiment_summary": self._calculate_sentiment_summary(articles),
                    "last_updated": datetime.now().isoformat()
                }
            else:
                pair_news = {"error": "No currency pair news available", "currency_pair": currency_pair}
            
            self._cache_set(cache_key, pair_news)
            return pair_news
            
        except Exception as e:
            print(f"Error fetching FXStreet currency pair news for {currency_pair}: {e}")
            return {"error": str(e), "currency_pair": currency_pair}
    
    def get_rate_news(self, limit: int = 15) -> Dict[str, Any]:
        """Get interest rate and central bank news"""
        if not self.api_key:
            return {"error": "FXStreet API key not configured"}
        
        cache_key = f"fxstreet_rate_{limit}"
        cached_data = self._cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            params = {
                "api_key": self.api_key,
                "category": "rates",
                "limit": limit
            }
            data = self._make_request(f"{self.base_url}/news/category", params=params)
            
            if "articles" in data and isinstance(data["articles"], list):
                articles = []
                for article in data["articles"]:
                    processed_article = {
                        "id": article.get("id", ""),
                        "title": article.get("title", ""),
                        "summary": article.get("summary", ""),
                        "url": article.get("url", ""),
                        "source": article.get("source", "FXStreet"),
                        "published_at": article.get("published_at", ""),
                        "category": article.get("category", "rates"),
                        "central_bank": article.get("central_bank", ""),
                        "impact_level": article.get("impact_level", "medium"),
                        "sentiment": self._analyze_sentiment(article.get("title", "") + " " + article.get("summary", "")),
                        "last_updated": datetime.now().isoformat()
                    }
                    articles.append(processed_article)
                
                rate_news = {
                    "category": "rates",
                    "total_results": len(articles),
                    "articles": articles,
                    "sentiment_summary": self._calculate_sentiment_summary(articles),
                    "last_updated": datetime.now().isoformat()
                }
            else:
                rate_news = {"error": "No rate news available"}
            
            self._cache_set(cache_key, rate_news)
            return rate_news
            
        except Exception as e:
            print(f"Error fetching FXStreet rate news: {e}")
            return {"error": str(e)}
    
    def get_high_frequency_news(self, limit: int = 20) -> Dict[str, Any]:
        """Get high-frequency news updates"""
        if not self.api_key:
            return {"error": "FXStreet API key not configured"}
        
        cache_key = f"fxstreet_hf_{limit}"
        cached_data = self._cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            params = {
                "api_key": self.api_key,
                "frequency": "high",
                "limit": limit
            }
            data = self._make_request(f"{self.base_url}/news/high-frequency", params=params)
            
            if "articles" in data and isinstance(data["articles"], list):
                articles = []
                for article in data["articles"]:
                    processed_article = {
                        "id": article.get("id", ""),
                        "title": article.get("title", ""),
                        "summary": article.get("summary", ""),
                        "url": article.get("url", ""),
                        "source": article.get("source", "FXStreet"),
                        "published_at": article.get("published_at", ""),
                        "frequency": "high",
                        "impact_level": article.get("impact_level", "medium"),
                        "sentiment": self._analyze_sentiment(article.get("title", "") + " " + article.get("summary", "")),
                        "last_updated": datetime.now().isoformat()
                    }
                    articles.append(processed_article)
                
                hf_news = {
                    "frequency": "high",
                    "total_results": len(articles),
                    "articles": articles,
                    "sentiment_summary": self._calculate_sentiment_summary(articles),
                    "last_updated": datetime.now().isoformat()
                }
            else:
                hf_news = {"error": "No high-frequency news available"}
            
            self._cache_set(cache_key, hf_news)
            return hf_news
            
        except Exception as e:
            print(f"Error fetching FXStreet high-frequency news: {e}")
            return {"error": str(e)}
    
    def get_forex_sentiment(self) -> Dict[str, Any]:
        """Get overall forex market sentiment from FXStreet"""
        try:
            # Get general forex news
            forex_news = self.get_latest_forex_news(30)
            
            # Get major currency pair news
            major_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CAD", "AUD/USD"]
            pair_news = []
            
            for pair in major_pairs:
                pair_data = self.get_currency_pair_news(pair, 5)
                if "articles" in pair_data:
                    pair_news.extend(pair_data["articles"])
            
            # Get rate news
            rate_news = self.get_rate_news(20)
            
            # Get high-frequency news
            hf_news = self.get_high_frequency_news(15)
            
            # Combine all articles
            all_articles = []
            if "articles" in forex_news:
                all_articles.extend(forex_news["articles"])
            all_articles.extend(pair_news)
            if "articles" in rate_news:
                all_articles.extend(rate_news["articles"])
            if "articles" in hf_news:
                all_articles.extend(hf_news["articles"])
            
            sentiment_data = {
                "overall_sentiment": self._calculate_overall_sentiment(all_articles),
                "forex_news_sentiment": forex_news.get("sentiment_summary", {}),
                "pair_news_count": len(pair_news),
                "rate_news_sentiment": rate_news.get("sentiment_summary", {}),
                "hf_news_sentiment": hf_news.get("sentiment_summary", {}),
                "total_articles": len(all_articles),
                "last_updated": datetime.now().isoformat()
            }
            
            return sentiment_data
            
        except Exception as e:
            print(f"Error calculating FXStreet sentiment: {e}")
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
            "positive", "increase", "improve", "boost", "support", "hawkish"
        ]
        
        # Negative keywords for forex
        negative_words = [
            "fall", "drop", "decline", "loss", "down", "negative", "weak", "bearish",
            "crash", "plunge", "concern", "risk", "worry", "fear", "sell-off",
            "negative", "decrease", "lower", "worse", "weaken", "pressure", "dovish"
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
        currency_pair = kwargs.get("currency_pair", "")
        
        if data_type == "pair":
            return self.get_currency_pair_news(currency_pair)
        elif data_type == "rates":
            return self.get_rate_news()
        elif data_type == "high_frequency":
            return self.get_high_frequency_news()
        elif data_type == "sentiment":
            return self.get_forex_sentiment()
        else:
            return self.get_latest_forex_news() 