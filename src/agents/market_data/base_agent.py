"""
Base class for all market data agents
Provides common functionality for data fetching, caching, and error handling
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import time
import requests
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

class MarketDataAgent(ABC):
    """Base class for market data agents"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.cache = {}
        self.cache_duration = int(os.getenv("MARKET_DATA_CACHE_DURATION", 300))
        self.rate_limit = int(os.getenv(f"{self.agent_name.upper()}_RATE_LIMIT", 60))
        self.last_request_time = 0
        self.request_count = 0
        self.request_window_start = time.time()
    
    @abstractmethod
    def get_data(self, **kwargs) -> Dict[str, Any]:
        """Get data from the API"""
        pass
    
    def _rate_limit_check(self):
        """Check and enforce rate limits"""
        current_time = time.time()
        
        # Reset counter if window has passed
        if current_time - self.request_window_start >= 60:
            self.request_count = 0
            self.request_window_start = current_time
        
        # Check if we're at the limit
        if self.request_count >= self.rate_limit:
            sleep_time = 60 - (current_time - self.request_window_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.request_count = 0
                self.request_window_start = time.time()
        
        self.request_count += 1
        self.last_request_time = current_time
    
    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache"""
        if key in self.cache:
            cached_data = self.cache[key]
            if datetime.now().isoformat() < cached_data.get("expires_at", "1970-01-01"):
                return cached_data.get("data")
        return None
    
    def _cache_set(self, key: str, data: Dict[str, Any]):
        """Set data in cache"""
        expires_at = (datetime.now() + timedelta(seconds=self.cache_duration)).isoformat()
        self.cache[key] = {
            "data": data,
            "expires_at": expires_at,
            "cached_at": datetime.now().isoformat()
        }
    
    def _make_request(self, url: str, headers: Dict = None, params: Dict = None) -> Dict[str, Any]:
        """Make HTTP request with rate limiting and error handling"""
        self._rate_limit_check()
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error in {self.agent_name}: {e}")
            return {"error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status and health"""
        return {
            "agent_name": self.agent_name,
            "cache_size": len(self.cache),
            "request_count": self.request_count,
            "rate_limit": self.rate_limit,
            "last_request": datetime.fromtimestamp(self.last_request_time).isoformat() if self.last_request_time > 0 else None
        } 