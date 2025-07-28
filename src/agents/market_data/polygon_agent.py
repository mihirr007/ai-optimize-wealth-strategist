"""
Polygon.io Agent - Real-time quotes for U.S. equities, forex, crypto, plus economic data
Requires API key from https://polygon.io/
"""

import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from .base_agent import MarketDataAgent
import os

class PolygonAgent(MarketDataAgent):
    """Polygon.io data agent for real-time market data"""
    
    def __init__(self):
        super().__init__("POLYGON")
        self.api_key = os.getenv("POLYGON_API_KEY")
        self.base_url = "https://api.polygon.io"
    
    def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time stock data"""
        if not self.api_key:
            return {"error": "Polygon API key not configured", "symbol": symbol}
        
        cache_key = f"polygon_stock_{symbol}"
        cached_data = self._cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Use simpler endpoint for basic quote data
            quote_url = f"{self.base_url}/v2/aggs/ticker/{symbol}/prev"
            quote_params = {"apikey": self.api_key, "adjusted": "true"}
            quote_data = self._make_request(quote_url, params=quote_params)
            
            # Process quote data
            if "results" in quote_data and quote_data["results"]:
                result = quote_data["results"][0]  # Get the most recent data
                data = {
                    "symbol": symbol,
                    "current_price": result.get("c", 0),  # Close price
                    "volume": result.get("v", 0),
                    "high": result.get("h", 0),
                    "low": result.get("l", 0),
                    "open": result.get("o", 0),
                    "price_change_1d": self._calculate_percentage_change(
                        result.get("o", 0),  # Open
                        result.get("c", 0)   # Close
                    ),
                    "last_updated": datetime.now().isoformat()
                }
            else:
                # Fallback to basic quote endpoint
                basic_url = f"{self.base_url}/v1/meta/symbols/{symbol}/company"
                basic_params = {"apikey": self.api_key}
                basic_data = self._make_request(basic_url, params=basic_params)
                
                if "error" not in basic_data:
                    data = {
                        "symbol": symbol,
                        "name": basic_data.get("name", symbol),
                        "sector": basic_data.get("sector", "Unknown"),
                        "industry": basic_data.get("industry", "Unknown"),
                        "current_price": 0,  # Not available in basic endpoint
                        "last_updated": datetime.now().isoformat(),
                        "note": "Basic company info only - price data unavailable"
                    }
                else:
                    data = {"error": "No data available", "symbol": symbol}
            
            self._cache_set(cache_key, data)
            return data
            
        except Exception as e:
            print(f"Error fetching Polygon data for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
    def get_forex_data(self, pair: str) -> Dict[str, Any]:
        """Get real-time forex data"""
        if not self.api_key:
            return {"error": "Polygon API key not configured", "pair": pair}
        
        cache_key = f"polygon_forex_{pair}"
        cached_data = self._cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Use forex aggregates endpoint
            url = f"{self.base_url}/v2/aggs/ticker/C:{pair}/prev"
            params = {"apikey": self.api_key, "adjusted": "true"}
            data = self._make_request(url, params=params)
            
            if "results" in data and data["results"]:
                result = data["results"][0]
                forex_data = {
                    "pair": pair,
                    "current_price": result.get("c", 0),
                    "volume": result.get("v", 0),
                    "high": result.get("h", 0),
                    "low": result.get("l", 0),
                    "open": result.get("o", 0),
                    "last_updated": datetime.now().isoformat()
                }
            else:
                forex_data = {"error": "No forex data available", "pair": pair}
            
            self._cache_set(cache_key, forex_data)
            return forex_data
            
        except Exception as e:
            print(f"Error fetching Polygon forex data for {pair}: {e}")
            return {"error": str(e), "pair": pair}
    
    def get_crypto_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time crypto data"""
        if not self.api_key:
            return {"error": "Polygon API key not configured", "symbol": symbol}
        
        cache_key = f"polygon_crypto_{symbol}"
        cached_data = self._cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Use crypto aggregates endpoint
            url = f"{self.base_url}/v2/aggs/ticker/X:{symbol}USD/prev"
            params = {"apikey": self.api_key, "adjusted": "true"}
            data = self._make_request(url, params=params)
            
            if "results" in data and data["results"]:
                result = data["results"][0]
                crypto_data = {
                    "symbol": symbol,
                    "current_price": result.get("c", 0),
                    "volume": result.get("v", 0),
                    "high": result.get("h", 0),
                    "low": result.get("l", 0),
                    "open": result.get("o", 0),
                    "last_updated": datetime.now().isoformat()
                }
            else:
                crypto_data = {"error": "No crypto data available", "symbol": symbol}
            
            self._cache_set(cache_key, crypto_data)
            return crypto_data
            
        except Exception as e:
            print(f"Error fetching Polygon crypto data for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
    def get_economic_indicators(self) -> Dict[str, Any]:
        """Get economic indicators"""
        if not self.api_key:
            return {"error": "Polygon API key not configured"}
        
        cache_key = "polygon_economic_indicators"
        cached_data = self._cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            indicators = {}
            
            # Get unemployment rate
            try:
                unemployment_url = f"{self.base_url}/v1/reference/indicators/UNRATE"
                unemployment_params = {"apikey": self.api_key}
                unemployment_data = self._make_request(unemployment_url, params=unemployment_params)
                if "results" in unemployment_data:
                    indicators["unemployment_rate"] = self._extract_latest_value(unemployment_data)
            except Exception as e:
                print(f"Error fetching unemployment data: {e}")
            
            # Get CPI data
            try:
                cpi_url = f"{self.base_url}/v1/reference/indicators/CPIAUCSL"
                cpi_params = {"apikey": self.api_key}
                cpi_data = self._make_request(cpi_url, params=cpi_params)
                if "results" in cpi_data:
                    indicators["cpi"] = self._extract_latest_value(cpi_data)
            except Exception as e:
                print(f"Error fetching CPI data: {e}")
            
            economic_data = {
                "indicators": indicators,
                "last_updated": datetime.now().isoformat()
            }
            
            self._cache_set(cache_key, economic_data)
            return economic_data
            
        except Exception as e:
            print(f"Error fetching Polygon economic data: {e}")
            return {"error": str(e)}
    
    def get_data(self, **kwargs) -> Dict[str, Any]:
        """Main data fetching method"""
        data_type = kwargs.get("data_type", "stock")
        symbol = kwargs.get("symbol", "")
        
        if data_type == "forex":
            return self.get_forex_data(symbol)
        elif data_type == "crypto":
            return self.get_crypto_data(symbol)
        elif data_type == "economic":
            return self.get_economic_indicators()
        else:
            return self.get_stock_data(symbol)
    
    def _calculate_percentage_change(self, old_price: float, new_price: float) -> float:
        """Calculate percentage change"""
        if old_price == 0:
            return 0.0
        return ((new_price - old_price) / old_price) * 100
    
    def _extract_latest_value(self, data: Dict[str, Any]) -> Optional[float]:
        """Extract the latest value from economic data"""
        if "results" in data and data["results"]:
            return data["results"][0].get("value", 0)
        return None 