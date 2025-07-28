"""
Finnhub Agent - Real-time financial news headlines, sentiment analysis, press releases
Requires API key from https://finnhub.io/
"""

import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ..market_data.base_agent import MarketDataAgent
import os

class FinnhubAgent(MarketDataAgent):
    """Finnhub data agent for financial news and sentiment"""
    
    def __init__(self):
        super().__init__("FINNHUB")
        self.api_key = os.getenv("FINNHUB_API_KEY")
        self.base_url = "https://finnhub.io/api/v1"
    
    def get_latest_news(self, category: str = "general", min_id: int = 0) -> Dict[str, Any]:
        """Get latest financial news"""
        if not self.api_key:
            return {"error": "Finnhub API key not configured"}
        
        cache_key = f"finnhub_news_{category}_{min_id}"
        cached_data = self._cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            params = {
                "category": category,
                "minId": min_id,
                "token": self.api_key
            }
            data = self._make_request(f"{self.base_url}/news", params=params)
            
            if isinstance(data, list):
                articles = []
                for article in data:
                    processed_article = {
                        "id": article.get("id", 0),
                        "category": article.get("category", ""),
                        "datetime": article.get("datetime", 0),
                        "headline": article.get("headline", ""),
                        "summary": article.get("summary", ""),
                        "url": article.get("url", ""),
                        "source": article.get("source", ""),
                        "related": article.get("related", ""),
                        "image": article.get("image", ""),
                        "sentiment": self._analyze_sentiment(article.get("headline", "") + " " + article.get("summary", "")),
                        "published_at": datetime.fromtimestamp(article.get("datetime", 0)).isoformat() if article.get("datetime") else ""
                    }
                    articles.append(processed_article)
                
                news_data = {
                    "category": category,
                    "total_results": len(articles),
                    "articles": articles,
                    "sentiment_summary": self._calculate_sentiment_summary(articles),
                    "last_updated": datetime.now().isoformat()
                }
            else:
                news_data = {"error": "No news data available"}
            
            self._cache_set(cache_key, news_data)
            return news_data
            
        except Exception as e:
            print(f"Error fetching Finnhub news data: {e}")
            return {"error": str(e)}
    
    def get_company_news(self, symbol: str, from_date: str = None, to_date: str = None) -> Dict[str, Any]:
        """Get company-specific news"""
        if not self.api_key:
            return {"error": "Finnhub API key not configured", "symbol": symbol}
        
        cache_key = f"finnhub_company_{symbol}"
        cached_data = self._cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Set default dates if not provided (last 7 days)
            if not from_date:
                from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            if not to_date:
                to_date = datetime.now().strftime("%Y-%m-%d")
            
            params = {
                "symbol": symbol,
                "from": from_date,
                "to": to_date,
                "token": self.api_key
            }
            
            data = self._make_request(f"{self.base_url}/company-news", params=params)
            
            if isinstance(data, list):
                articles = []
                for article in data:
                    processed_article = {
                        "id": article.get("id", 0),
                        "category": article.get("category", ""),
                        "datetime": article.get("datetime", 0),
                        "headline": article.get("headline", ""),
                        "summary": article.get("summary", ""),
                        "url": article.get("url", ""),
                        "source": article.get("source", ""),
                        "related": article.get("related", ""),
                        "image": article.get("image", ""),
                        "sentiment": self._analyze_sentiment(article.get("headline", "") + " " + article.get("summary", "")),
                        "published_at": datetime.fromtimestamp(article.get("datetime", 0)).isoformat() if article.get("datetime") else ""
                    }
                    articles.append(processed_article)
                
                company_news = {
                    "symbol": symbol,
                    "total_results": len(articles),
                    "articles": articles,
                    "sentiment_summary": self._calculate_sentiment_summary(articles),
                    "last_updated": datetime.now().isoformat()
                }
            else:
                company_news = {"error": "No company news available", "symbol": symbol}
            
            self._cache_set(cache_key, company_news)
            return company_news
            
        except Exception as e:
            print(f"Error fetching Finnhub company news for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
    def get_market_sentiment(self) -> Dict[str, Any]:
        """Get overall market sentiment from news"""
        try:
            # Get general news
            general_news = self.get_latest_news("general", 50)
            
            # Get company news for major stocks
            major_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
            company_news = []
            
            for symbol in major_symbols:
                symbol_news = self.get_company_news(symbol)
                if "articles" in symbol_news:
                    company_news.extend(symbol_news["articles"])
            
            # Combine all articles
            all_articles = []
            if "articles" in general_news:
                all_articles.extend(general_news["articles"])
            all_articles.extend(company_news)
            
            sentiment_data = {
                "overall_sentiment": self._calculate_overall_sentiment(all_articles),
                "general_news_sentiment": general_news.get("sentiment_summary", {}),
                "company_news_count": len(company_news),
                "total_articles": len(all_articles),
                "last_updated": datetime.now().isoformat()
            }
            
            return sentiment_data
            
        except Exception as e:
            print(f"Error calculating Finnhub market sentiment: {e}")
            return {"error": str(e)}
    
    def get_press_releases(self, symbol: str = None) -> Dict[str, Any]:
        """Get press releases"""
        if not self.api_key:
            return {"error": "Finnhub API key not configured"}
        
        cache_key = f"finnhub_press_{symbol or 'general'}"
        cached_data = self._cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            params = {
                "token": self.api_key
            }
            if symbol:
                params["symbol"] = symbol
            
            data = self._make_request(f"{self.base_url}/press-releases", params=params)
            
            if isinstance(data, list):
                releases = []
                for release in data:
                    processed_release = {
                        "symbol": release.get("symbol", ""),
                        "headline": release.get("headline", ""),
                        "summary": release.get("summary", ""),
                        "url": release.get("url", ""),
                        "datetime": release.get("datetime", 0),
                        "published_at": datetime.fromtimestamp(release.get("datetime", 0)).isoformat() if release.get("datetime") else ""
                    }
                    releases.append(processed_release)
                
                press_data = {
                    "symbol": symbol,
                    "total_results": len(releases),
                    "releases": releases,
                    "last_updated": datetime.now().isoformat()
                }
            else:
                press_data = {"error": "No press releases available"}
            
            self._cache_set(cache_key, press_data)
            return press_data
            
        except Exception as e:
            print(f"Error fetching Finnhub press releases: {e}")
            return {"error": str(e)}
    
    def _analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis based on keywords"""
        if not text:
            return "neutral"
        
        text_lower = text.lower()
        
        # Positive keywords
        positive_words = [
            "surge", "jump", "rise", "gain", "up", "positive", "growth", "profit",
            "earnings", "beat", "exceed", "strong", "bullish", "rally", "recovery",
            "positive", "increase", "higher", "better", "success", "win"
        ]
        
        # Negative keywords
        negative_words = [
            "fall", "drop", "decline", "loss", "down", "negative", "weak", "bearish",
            "crash", "plunge", "concern", "risk", "worry", "fear", "sell-off",
            "negative", "decrease", "lower", "worse", "fail", "lose"
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
        category = kwargs.get("category", "general")
        symbol = kwargs.get("symbol", "")
        
        if data_type == "company":
            return self.get_company_news(symbol)
        elif data_type == "sentiment":
            return self.get_market_sentiment()
        elif data_type == "press":
            return self.get_press_releases(symbol)
        else:
            return self.get_latest_news(category) 