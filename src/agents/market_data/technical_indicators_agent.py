"""
Technical Indicators Agent - Calculates indicators from YFinance data
Free alternative to Alpha Vantage for technical analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from .base_agent import MarketDataAgent
import yfinance as yf

class TechnicalIndicatorsAgent(MarketDataAgent):
    """Technical indicators agent using YFinance data"""
    
    def __init__(self):
        super().__init__("TECHNICAL_INDICATORS")
    
    def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Get technical indicators for a stock"""
        cache_key = f"technical_indicators_{symbol}"
        cached_data = self._cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            if hist.empty:
                return {"error": f"No historical data available for {symbol}", "symbol": symbol}
            indicators = self._calculate_technical_indicators(hist)
            data = {"symbol": symbol, **indicators, "last_updated": datetime.now().isoformat()}
            self._cache_set(cache_key, data)
            return data
        except Exception as e:
            print(f"Error calculating technical indicators for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
    def get_portfolio_data(self, symbols: List[str]) -> Dict[str, Any]:
        portfolio_data = {}
        for symbol in symbols:
            portfolio_data[symbol] = self.get_stock_data(symbol)
        return {
            "portfolio": portfolio_data,
            "summary": self._calculate_portfolio_summary(portfolio_data),
            "last_updated": datetime.now().isoformat()
        }
    
    def get_data(self, **kwargs) -> Dict[str, Any]:
        symbol = kwargs.get("symbol")
        symbols = kwargs.get("symbols", [])
        if symbol:
            return self.get_stock_data(symbol)
        elif symbols:
            return self.get_portfolio_data(symbols)
        else:
            return {"error": "No symbol or symbols provided"}
    
    def _calculate_technical_indicators(self, hist: pd.DataFrame) -> Dict[str, float]:
        indicators = {}
        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        if len(close) < 20:
            return indicators
        # RSI
        indicators["rsi"] = self._calculate_rsi(close)
        # MACD
        macd, macd_signal, macd_hist = self._calculate_macd(close)
        indicators["macd"] = macd
        indicators["macd_signal"] = macd_signal
        indicators["macd_histogram"] = macd_hist
        # SMA
        indicators["sma_20"] = close.rolling(window=20).mean().iloc[-1]
        indicators["sma_50"] = close.rolling(window=50).mean().iloc[-1]
        indicators["sma_200"] = close.rolling(window=200).mean().iloc[-1]
        # EMA
        indicators["ema_12"] = close.ewm(span=12).mean().iloc[-1]
        indicators["ema_26"] = close.ewm(span=26).mean().iloc[-1]
        # Bollinger Bands
        sma = close.rolling(window=20).mean()
        std = close.rolling(window=20).std()
        indicators["bollinger_upper"] = (sma + 2*std).iloc[-1]
        indicators["bollinger_middle"] = sma.iloc[-1]
        indicators["bollinger_lower"] = (sma - 2*std).iloc[-1]
        # Stochastic Oscillator
        low_min = low.rolling(window=14).min()
        high_max = high.rolling(window=14).max()
        k_percent = 100 * ((close - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=3).mean()
        indicators["stochastic_k"] = k_percent.iloc[-1]
        indicators["stochastic_d"] = d_percent.iloc[-1]
        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        indicators["atr"] = tr.rolling(window=14).mean().iloc[-1]
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
    
    def _calculate_portfolio_summary(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        valid_data = {k: v for k, v in portfolio_data.items() if "error" not in v}
        if not valid_data:
            return {"error": "No valid technical data available"}
        rsi_values = [data.get("rsi") for data in valid_data.values() if data.get("rsi") is not None]
        avg_rsi = sum(rsi_values) / len(rsi_values) if rsi_values else None
        macd_values = [data.get("macd") for data in valid_data.values() if data.get("macd") is not None]
        avg_macd = sum(macd_values) / len(macd_values) if macd_values else None
        return {
            "average_rsi": avg_rsi,
            "average_macd": avg_macd,
            "symbols_count": len(valid_data),
            "last_updated": datetime.now().isoformat()
        }
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "name": "Technical Indicators",
            "available": True,
            "description": "Free technical indicators calculated from YFinance data",
            "features": ["RSI", "MACD", "SMA", "EMA", "Bollinger Bands", "Stochastic", "ATR"],
            "last_updated": datetime.now().isoformat()
        } 