"""
Comprehensive market data service that coordinates all market data agents
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import os
from dotenv import load_dotenv

# Import market data agents
from agents.market_data.yfinance_agent import YFinanceAgent
from agents.market_data.polygon_agent import PolygonAgent
from agents.market_data.technical_indicators_agent import TechnicalIndicatorsAgent
# from agents.market_data.fmp_agent import FMPAgent
# from agents.market_data.twelve_data_agent import TwelveDataAgent
# from agents.market_data.marketstack_agent import MarketstackAgent

from agents.news.newsapi_us_agent import NewsAPIUSAgent
# from agents.news.newsapi_canada_agent import NewsAPICanadaAgent
from agents.news.finnhub_agent import FinnhubAgent
# from agents.news.forex_news_agent import ForexNewsAgent
# from agents.news.fxstreet_agent import FXStreetAgent

from agents.economic.fred_agent import FREDAgent
# from agents.economic.polygon_economic_agent import PolygonEconomicAgent

load_dotenv()

class MarketDataService:
    """Comprehensive market data service"""
    
    def __init__(self):
        # Initialize market data agents
        self.yfinance_agent = YFinanceAgent()
        self.polygon_agent = PolygonAgent()
        self.technical_indicators_agent = TechnicalIndicatorsAgent()
        # self.fmp_agent = FMPAgent()
        # self.twelve_data_agent = TwelveDataAgent()
        # self.marketstack_agent = MarketstackAgent()
        
        # Initialize news agents
        self.newsapi_us_agent = NewsAPIUSAgent()
        # self.newsapi_canada_agent = NewsAPICanadaAgent()
        self.finnhub_agent = FinnhubAgent()
        # self.forex_news_agent = ForexNewsAgent()
        # self.fxstreet_agent = FXStreetAgent()
        
        # Initialize economic agents
        self.fred_agent = FREDAgent()
        # self.polygon_economic_agent = PolygonEconomicAgent()
        
        self.agents = {
            "yfinance": self.yfinance_agent,
            "polygon": self.polygon_agent,
            "technical_indicators": self.technical_indicators_agent,
            # "fmp": self.fmp_agent,
            # "twelve_data": self.twelve_data_agent,
            # "marketstack": self.marketstack_agent,
            "newsapi_us": self.newsapi_us_agent,
            # "newsapi_canada": self.newsapi_canada_agent,
            "finnhub": self.finnhub_agent,
            # "forex_news": self.forex_news_agent,
            # "fxstreet": self.fxstreet_agent,
            "fred": self.fred_agent,
            # "polygon_economic": self.polygon_economic_agent,
        }
    
    def get_comprehensive_market_data(self, symbols: list) -> dict:
        """
        Fetches comprehensive market data for a list of symbols from all integrated agents (Phase 1, 2, and 3).
        Returns a structured dictionary with organized data for display.
        """
        raw_results = {}
        
        # Check if it's weekend to use last week's data
        from datetime import datetime, timedelta
        now = datetime.now()
        is_weekend = now.weekday() >= 5  # Saturday = 5, Sunday = 6
        
        # Calculate last week's date range for weekend data
        if is_weekend:
            # Get last Friday's date
            days_since_friday = (now.weekday() - 4) % 7
            last_friday = now - timedelta(days=days_since_friday)
            from_date = (last_friday - timedelta(days=7)).strftime('%Y-%m-%d')
            to_date = last_friday.strftime('%Y-%m-%d')
        else:
            # Use current week for weekday data
            from_date = (now - timedelta(days=7)).strftime('%Y-%m-%d')
            to_date = now.strftime('%Y-%m-%d')
        
        # Fetch data from all agents
        # YFinance
        try:
            raw_results["yfinance"] = self.yfinance_agent.get_portfolio_data(symbols)
        except Exception as e:
            raw_results["yfinance"] = {"error": str(e)}
        
        # Polygon.io
        try:
            if symbols:
                raw_results["polygon"] = self.polygon_agent.get_stock_data(symbols[0])
            else:
                raw_results["polygon"] = {"error": "No symbols provided"}
        except Exception as e:
            raw_results["polygon"] = {"error": str(e)}
        
        # Technical Indicators (Phase 3)
        try:
            raw_results["technical_indicators"] = self.technical_indicators_agent.get_portfolio_data(symbols)
        except Exception as e:
            raw_results["technical_indicators"] = {"error": str(e)}
        
        # NewsAPI US - Fetch ticker-specific news
        try:
            ticker_news = []
            seen_articles = set()  # To avoid duplicates
            
            # Company name mappings for better search
            company_names = {
                "AAPL": ["Apple", "Apple Inc", "iPhone", "iPad", "Mac"],
                "MSFT": ["Microsoft", "Microsoft Corporation", "Windows", "Office", "Azure"],
                "GOOGL": ["Google", "Alphabet", "YouTube", "Android", "Chrome"],
                "TSLA": ["Tesla", "Tesla Inc", "Elon Musk"],
                "NVDA": ["NVIDIA", "NVIDIA Corporation", "GPU"],
                "AMZN": ["Amazon", "Amazon.com", "AWS"],
                "META": ["Meta", "Facebook", "Instagram", "WhatsApp"],
                "NFLX": ["Netflix", "Netflix Inc"],
                "JPM": ["JPMorgan", "JPMorgan Chase", "JP Morgan"],
                "JNJ": ["Johnson & Johnson", "J&J"]
            }
            
            for symbol in symbols:
                # Get company names for this symbol
                company_terms = company_names.get(symbol, [symbol])
                
                # Search with multiple terms for better results
                for term in company_terms[:2]:  # Use first 2 terms to avoid too many requests
                    symbol_news = self.newsapi_us_agent.search_news(term, page_size=6, from_date=from_date, to_date=to_date)
                    if "error" not in symbol_news and "articles" in symbol_news:
                        for article in symbol_news["articles"]:
                            # Create unique identifier for deduplication
                            article_id = f"{article.get('title', '')}_{article.get('url', '')}"
                            if article_id not in seen_articles:
                                article["related_ticker"] = symbol
                                article["search_term"] = term
                                ticker_news.append(article)
                                seen_articles.add(article_id)
            
            raw_results["newsapi_us"] = {
                "articles": ticker_news,
                "total_results": len(ticker_news),
                "last_updated": datetime.now().isoformat(),
                "date_range": f"{from_date} to {to_date}" if is_weekend else "Current week"
            }
        except Exception as e:
            raw_results["newsapi_us"] = {"error": str(e)}
        
        # Finnhub - Fetch ticker-specific news
        try:
            ticker_news = []
            seen_articles = set()  # To avoid duplicates
            
            for symbol in symbols:
                # Get company-specific news for the ticker
                symbol_news = self.finnhub_agent.get_company_news(symbol, from_date=from_date, to_date=to_date)
                if "error" not in symbol_news and "articles" in symbol_news:
                    for article in symbol_news["articles"]:
                        # Create unique identifier for deduplication
                        article_id = f"{article.get('id', '')}_{article.get('headline', '')}"
                        if article_id not in seen_articles:
                            article["related_ticker"] = symbol
                            ticker_news.append(article)
                            seen_articles.add(article_id)
            
            raw_results["finnhub"] = {
                "articles": ticker_news,
                "total_results": len(ticker_news),
                "last_updated": datetime.now().isoformat(),
                "date_range": f"{from_date} to {to_date}" if is_weekend else "Current week"
            }
        except Exception as e:
            raw_results["finnhub"] = {"error": str(e)}
        
        # FRED
        try:
            raw_results["fred"] = self.fred_agent.get_economic_indicators()
        except Exception as e:
            raw_results["fred"] = {"error": str(e)}
        
        # Structure the data for display
        structured_data = self._structure_market_data(raw_results, symbols)
        return structured_data
    
    def _structure_market_data(self, raw_results: dict, symbols: list) -> dict:
        """Structure raw market data into organized format for display"""
        structured = {
            "available_sources": [],
            "error_sources": [],
            "price_data": {},
            "news_data": [],
            "news_sentiment": {},
            "economic_indicators": {},
            "technical_data": {},
            "sector_performance": {},
            "last_updated": datetime.now().isoformat()
        }
        
        # Process price data from various sources
        for symbol in symbols:
            structured["price_data"][symbol] = {}
            
            # Try to get price from YFinance
            if "yfinance" in raw_results and "error" not in raw_results["yfinance"]:
                yf_data = raw_results["yfinance"].get("portfolio", {}).get(symbol, {})
                if yf_data:
                    # Map YFinance data structure to our format
                    current_price = yf_data.get("current_price", 0)
                    if current_price and current_price > 0:
                        structured["price_data"][symbol] = {
                            "price": float(current_price),
                            "change": yf_data.get("price_change_1d", 0),
                            "change_percent": yf_data.get("price_change_1d", 0),
                            "volume": yf_data.get("volume", 0),
                            "market_cap": yf_data.get("market_cap", 0),
                            "pe_ratio": yf_data.get("pe_ratio", 0)
                        }
                    else:
                        print(f"⚠️  No valid price data for {symbol} from YFinance")
            

            
            # Try to get price from Polygon
            if "polygon" in raw_results and "error" not in raw_results["polygon"]:
                poly_data = raw_results["polygon"]
                if not structured["price_data"][symbol].get("price"):
                    structured["price_data"][symbol]["price"] = poly_data.get("price", 0)
                    structured["price_data"][symbol]["change"] = poly_data.get("change", 0)
                    structured["price_data"][symbol]["change_percent"] = poly_data.get("change_percent", 0)
        
        # Process news data - organize by ticker
        ticker_news = {}
        all_news = []
        
        def validate_company_relevance(article, ticker):
            """Validate that article is actually about the specific company"""
            title = article.get("title", "").lower()
            description = article.get("description", "").lower()
            content = article.get("content", "").lower()
            
            # Company-specific keywords
            company_keywords = {
                "AAPL": ["apple", "iphone", "ipad", "mac", "tim cook", "ios"],
                "MSFT": ["microsoft", "windows", "office", "azure", "satya nadella", "xbox"],
                "GOOGL": ["google", "alphabet", "youtube", "android", "chrome", "sundar pichai"],
                "TSLA": ["tesla", "elon musk", "model s", "model 3", "cybertruck"],
                "NVDA": ["nvidia", "gpu", "jensen huang", "rtx", "ai chips"],
                "AMZN": ["amazon", "aws", "jeff bezos", "prime", "echo"],
                "META": ["meta", "facebook", "instagram", "mark zuckerberg", "whatsapp"],
                "NFLX": ["netflix", "streaming", "ted sarandos"],
                "JPM": ["jpmorgan", "jp morgan", "jamie dimon", "chase"],
                "JNJ": ["johnson & johnson", "j&j", "pharmaceuticals"]
            }
            
            keywords = company_keywords.get(ticker, [ticker.lower()])
            text_to_check = f"{title} {description} {content}"
            
            # Check if any company-specific keyword is mentioned
            return any(keyword in text_to_check for keyword in keywords)
        
        if "newsapi_us" in raw_results and "error" not in raw_results["newsapi_us"]:
            news_data = raw_results["newsapi_us"].get("articles", [])
            for article in news_data:
                ticker = article.get("related_ticker", "General")
                if ticker != "General" and validate_company_relevance(article, ticker):
                    if ticker not in ticker_news:
                        ticker_news[ticker] = []
                    ticker_news[ticker].append(article)
        
        if "finnhub" in raw_results and "error" not in raw_results["finnhub"]:
            news_data = raw_results["finnhub"].get("articles", [])
            for article in news_data:
                ticker = article.get("related_ticker", "General")
                if ticker != "General" and validate_company_relevance(article, ticker):
                    if ticker not in ticker_news:
                        ticker_news[ticker] = []
                    ticker_news[ticker].append(article)
        
        # Only show ticker-specific news, no general news
        structured["ticker_news"] = ticker_news
        structured["news_data"] = []  # Empty since we only want ticker-specific news
        
        # Process sentiment data
        if "newsapi_us" in raw_results and "error" not in raw_results["newsapi_us"]:
            sentiment = raw_results["newsapi_us"].get("sentiment", {})
            structured["news_sentiment"] = {
                "positive": sentiment.get("positive", 0),
                "neutral": sentiment.get("neutral", 0),
                "negative": sentiment.get("negative", 0),
                "overall_sentiment": sentiment.get("overall_sentiment", "Neutral")
            }
        
        # Process economic indicators
        if "fred" in raw_results and "error" not in raw_results["fred"]:
            indicators = raw_results["fred"].get("indicators", {})
            structured["economic_indicators"] = indicators
        
        # Process technical data (if available)
        for symbol in symbols:
            structured["technical_data"][symbol] = {}
            # Add technical indicators if available from any source
            if "technical_indicators" in raw_results and "error" not in raw_results["technical_indicators"]:
                tech_data = raw_results["technical_indicators"].get("portfolio", {}).get(symbol, {})
                if tech_data:
                    structured["technical_data"][symbol] = tech_data
        
        # Track available and error sources
        for source_name, source_data in raw_results.items():
            if "error" in source_data:
                structured["error_sources"].append(source_name)
            else:
                structured["available_sources"].append(source_name)
        
        return structured
    
    def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Get data for a single stock"""
        return self.yfinance_agent.get_stock_data(symbol)
    
    def get_portfolio_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get data for a portfolio of stocks"""
        return self.yfinance_agent.get_portfolio_data(symbols)
    
    def get_market_indices(self) -> Dict[str, Any]:
        """Get major market indices"""
        return self.yfinance_agent.get_market_indices()
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        status = {}
        for name, agent in self.agents.items():
            status[name] = agent.get_status()
        return status
    
    def _generate_data_summary(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all market data sources"""
        summary = {
            "total_sources": len(sources),
            "available_sources": [],
            "error_sources": [],
            "data_points": 0,
            "last_updated": datetime.now().isoformat()
        }
        
        for source_name, source_data in sources.items():
            if "error" in source_data:
                summary["error_sources"].append(source_name)
            else:
                summary["available_sources"].append(source_name)
                # Count data points based on source type
                if isinstance(source_data, dict):
                    if "portfolio" in source_data:
                        summary["data_points"] += len(source_data.get("portfolio", {}))
                    elif "articles" in source_data:
                        summary["data_points"] += len(source_data.get("articles", []))
                    elif "indicators" in source_data:
                        summary["data_points"] += len(source_data.get("indicators", {}))
                    else:
                        summary["data_points"] += len(source_data)
        
        return summary

# Global instance
market_data_service = MarketDataService() 