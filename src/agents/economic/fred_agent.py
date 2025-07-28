"""
FRED Agent - U.S. macro indicators like CPI, interest rates, GDP
Requires API key from https://fred.stlouisfed.org/
"""

import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ..market_data.base_agent import MarketDataAgent
import os

class FREDAgent(MarketDataAgent):
    """FRED data agent for economic indicators"""
    
    def __init__(self):
        super().__init__("FRED")
        self.api_key = os.getenv("FRED_API_KEY")
        self.base_url = "https://api.stlouisfed.org/fred"
    
    def get_economic_indicators(self) -> Dict[str, Any]:
        """Get key economic indicators"""
        if not self.api_key:
            return {"error": "FRED API key not configured"}
        
        cache_key = "fred_economic_indicators"
        cached_data = self._cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Get unemployment rate
            unemployment_data = self._get_series("UNRATE", "Unemployment Rate")
            
            # Get CPI (inflation)
            cpi_data = self._get_series("CPIAUCSL", "Consumer Price Index")
            
            # Get GDP
            gdp_data = self._get_series("GDP", "Gross Domestic Product")
            
            # Get Federal Funds Rate
            fed_rate_data = self._get_series("FEDFUNDS", "Federal Funds Rate")
            
            # Get 10-Year Treasury Rate
            treasury_data = self._get_series("GS10", "10-Year Treasury Rate")
            
            economic_data = {
                "unemployment_rate": unemployment_data,
                "cpi": cpi_data,
                "gdp": gdp_data,
                "federal_funds_rate": fed_rate_data,
                "treasury_10y": treasury_data,
                "last_updated": datetime.now().isoformat()
            }
            
            self._cache_set(cache_key, economic_data)
            return economic_data
            
        except Exception as e:
            print(f"Error fetching FRED economic data: {e}")
            return {"error": str(e)}
    
    def get_inflation_data(self) -> Dict[str, Any]:
        """Get detailed inflation data"""
        if not self.api_key:
            return {"error": "FRED API key not configured"}
        
        cache_key = "fred_inflation_data"
        cached_data = self._cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Core CPI (excluding food and energy)
            core_cpi_data = self._get_series("CPILFESL", "Core CPI")
            
            # PCE Price Index (Fed's preferred measure)
            pce_data = self._get_series("PCEPI", "PCE Price Index")
            
            # Core PCE
            core_pce_data = self._get_series("PCEPILFE", "Core PCE")
            
            inflation_data = {
                "cpi_all_items": self._get_series("CPIAUCSL", "CPI All Items"),
                "cpi_core": core_cpi_data,
                "pce_price_index": pce_data,
                "core_pce": core_pce_data,
                "last_updated": datetime.now().isoformat()
            }
            
            self._cache_set(cache_key, inflation_data)
            return inflation_data
            
        except Exception as e:
            print(f"Error fetching FRED inflation data: {e}")
            return {"error": str(e)}
    
    def get_interest_rates(self) -> Dict[str, Any]:
        """Get interest rate data"""
        if not self.api_key:
            return {"error": "FRED API key not configured"}
        
        cache_key = "fred_interest_rates"
        cached_data = self._cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            rates_data = {
                "federal_funds_rate": self._get_series("FEDFUNDS", "Federal Funds Rate"),
                "treasury_3m": self._get_series("GS3M", "3-Month Treasury"),
                "treasury_2y": self._get_series("GS2", "2-Year Treasury"),
                "treasury_5y": self._get_series("GS5", "5-Year Treasury"),
                "treasury_10y": self._get_series("GS10", "10-Year Treasury"),
                "treasury_30y": self._get_series("GS30", "30-Year Treasury"),
                "prime_rate": self._get_series("PRIME", "Prime Rate"),
                "last_updated": datetime.now().isoformat()
            }
            
            self._cache_set(cache_key, rates_data)
            return rates_data
            
        except Exception as e:
            print(f"Error fetching FRED interest rate data: {e}")
            return {"error": str(e)}
    
    def get_labor_market_data(self) -> Dict[str, Any]:
        """Get labor market indicators"""
        if not self.api_key:
            return {"error": "FRED API key not configured"}
        
        cache_key = "fred_labor_market"
        cached_data = self._cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            labor_data = {
                "unemployment_rate": self._get_series("UNRATE", "Unemployment Rate"),
                "labor_force_participation": self._get_series("CIVPART", "Labor Force Participation"),
                "employment_population_ratio": self._get_series("EMRATIO", "Employment-Population Ratio"),
                "average_hourly_earnings": self._get_series("AHETPI", "Average Hourly Earnings"),
                "job_openings": self._get_series("JTSJOL", "Job Openings"),
                "last_updated": datetime.now().isoformat()
            }
            
            self._cache_set(cache_key, labor_data)
            return labor_data
            
        except Exception as e:
            print(f"Error fetching FRED labor market data: {e}")
            return {"error": str(e)}
    
    def get_market_indicators(self) -> Dict[str, Any]:
        """Get market-related economic indicators"""
        if not self.api_key:
            return {"error": "FRED API key not configured"}
        
        cache_key = "fred_market_indicators"
        cached_data = self._cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            market_data = {
                "consumer_sentiment": self._get_series("UMCSENT", "Consumer Sentiment"),
                "business_confidence": self._get_series("NAPM", "ISM Manufacturing PMI"),
                "housing_starts": self._get_series("HOUST", "Housing Starts"),
                "retail_sales": self._get_series("RSAFS", "Retail Sales"),
                "industrial_production": self._get_series("INDPRO", "Industrial Production"),
                "last_updated": datetime.now().isoformat()
            }
            
            self._cache_set(cache_key, market_data)
            return market_data
            
        except Exception as e:
            print(f"Error fetching FRED market indicators: {e}")
            return {"error": str(e)}
    
    def _get_series(self, series_id: str, title: str) -> Dict[str, Any]:
        """Get a specific economic series from FRED"""
        try:
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "limit": 1,
                "sort_order": "desc"
            }
            data = self._make_request(f"{self.base_url}/series/observations", params=params)
            
            if "observations" in data and data["observations"]:
                latest = data["observations"][0]
                return {
                    "series_id": series_id,
                    "title": title,
                    "value": float(latest.get("value", 0)),
                    "date": latest.get("date", ""),
                    "realtime_start": latest.get("realtime_start", ""),
                    "realtime_end": latest.get("realtime_end", "")
                }
            else:
                return {"error": f"No data available for {series_id}"}
                
        except Exception as e:
            print(f"Error fetching FRED series {series_id}: {e}")
            return {"error": str(e)}
    
    def get_data(self, **kwargs) -> Dict[str, Any]:
        """Main data fetching method"""
        data_type = kwargs.get("data_type", "economic")
        
        if data_type == "inflation":
            return self.get_inflation_data()
        elif data_type == "interest_rates":
            return self.get_interest_rates()
        elif data_type == "labor_market":
            return self.get_labor_market_data()
        elif data_type == "market_indicators":
            return self.get_market_indicators()
        else:
            return self.get_economic_indicators() 