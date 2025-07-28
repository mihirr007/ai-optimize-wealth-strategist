"""
YFinance Agent - Delayed stock data, fundamentals, historical prices
Free tier with rate limits
"""

import yfinance as yf
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from .base_agent import MarketDataAgent

class YFinanceAgent(MarketDataAgent):
    """YFinance data agent for stock and ETF data"""
    
    def __init__(self):
        super().__init__("YFINANCE")
    
    def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive stock/ETF data"""
        cache_key = f"yfinance_stock_{symbol}"
        cached_data = self._cache_get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get basic info
            info = ticker.info
            
            # Get historical data
            hist = ticker.history(period="1mo")
            
            # Get financial statements
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cashflow = ticker.cashflow
            
            # Get current price with fallbacks
            current_price = info.get("currentPrice")
            if not current_price or current_price == 0:
                # Try regular market price
                current_price = info.get("regularMarketPrice")
            if not current_price or current_price == 0:
                # Try previous close
                current_price = info.get("previousClose")
            if not current_price or current_price == 0:
                # Use latest historical close
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
            
            # Get volume with fallbacks
            volume = info.get("volume")
            if not volume or volume == 0:
                # Use latest historical volume
                if not hist.empty:
                    volume = hist['Volume'].iloc[-1]
            
            # Calculate price change
            price_change_1d = info.get("regularMarketChangePercent")
            if not price_change_1d and not hist.empty and len(hist) >= 2:
                # Calculate from historical data
                prev_close = hist['Close'].iloc[-2]
                current_close = hist['Close'].iloc[-1]
                price_change_1d = ((current_close - prev_close) / prev_close) * 100
            
            data = {
                "symbol": symbol,
                "name": info.get("longName", symbol),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price_to_book": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "current_price": current_price,
                "volume": volume,
                "avg_volume": info.get("averageVolume"),
                "price_change_1d": price_change_1d,
                "price_change_1m": self._calculate_return(hist, 1) if not hist.empty else None,
                "price_change_3m": self._calculate_return(hist, 3) if not hist.empty else None,
                "price_change_1y": self._calculate_return(hist, 12) if not hist.empty else None,
                "volatility": self._calculate_volatility(hist) if not hist.empty else None,
                "financials": self._process_financials(financials),
                "balance_sheet": self._process_balance_sheet(balance_sheet),
                "cashflow": self._process_cashflow(cashflow),
                "last_updated": datetime.now().isoformat()
            }
            
            self._cache_set(cache_key, data)
            return data
            
        except Exception as e:
            print(f"Error fetching YFinance data for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
    def get_portfolio_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get data for multiple symbols"""
        portfolio_data = {}
        
        for symbol in symbols:
            portfolio_data[symbol] = self.get_stock_data(symbol)
        
        return {
            "portfolio": portfolio_data,
            "summary": self._calculate_portfolio_summary(portfolio_data),
            "last_updated": datetime.now().isoformat()
        }
    
    def get_market_indices(self) -> Dict[str, Any]:
        """Get major market indices"""
        indices = {
            "^GSPC": "S&P 500",
            "^DJI": "Dow Jones",
            "^IXIC": "NASDAQ",
            "^GSPTSE": "TSX",
            "^VIX": "VIX Volatility"
        }
        
        data = {}
        for symbol, name in indices.items():
            data[name] = self.get_stock_data(symbol)
        
        return {
            "indices": data,
            "last_updated": datetime.now().isoformat()
        }
    
    def get_data(self, **kwargs) -> Dict[str, Any]:
        """Main data fetching method"""
        symbol = kwargs.get("symbol")
        symbols = kwargs.get("symbols", [])
        data_type = kwargs.get("data_type", "stock")
        
        if data_type == "portfolio" and symbols:
            return self.get_portfolio_data(symbols)
        elif data_type == "indices":
            return self.get_market_indices()
        elif symbol:
            return self.get_stock_data(symbol)
        else:
            return {"error": "No symbol or symbols provided"}
    
    def _calculate_return(self, hist_data: pd.DataFrame, months: int) -> Optional[float]:
        """Calculate return over specified period"""
        if len(hist_data) < months * 20:
            return None
        
        start_price = hist_data['Close'].iloc[-(months * 20)]
        end_price = hist_data['Close'].iloc[-1]
        return ((end_price - start_price) / start_price) * 100
    
    def _calculate_volatility(self, hist_data: pd.DataFrame) -> Optional[float]:
        """Calculate historical volatility"""
        if len(hist_data) < 20:
            return None
        
        returns = hist_data['Close'].pct_change().dropna()
        return returns.std() * (252 ** 0.5) * 100
    
    def _process_financials(self, financials: pd.DataFrame) -> Dict[str, Any]:
        """Process financial statements"""
        if financials.empty:
            return {}
        
        try:
            return {
                "revenue": financials.loc["Total Revenue"].iloc[0] if "Total Revenue" in financials.index else None,
                "net_income": financials.loc["Net Income"].iloc[0] if "Net Income" in financials.index else None,
                "ebitda": financials.loc["EBITDA"].iloc[0] if "EBITDA" in financials.index else None,
                "gross_profit": financials.loc["Gross Profit"].iloc[0] if "Gross Profit" in financials.index else None
            }
        except:
            return {}
    
    def _process_balance_sheet(self, balance_sheet: pd.DataFrame) -> Dict[str, Any]:
        """Process balance sheet data"""
        if balance_sheet.empty:
            return {}
        
        try:
            return {
                "total_assets": balance_sheet.loc["Total Assets"].iloc[0] if "Total Assets" in balance_sheet.index else None,
                "total_liabilities": balance_sheet.loc["Total Liabilities"].iloc[0] if "Total Liabilities" in balance_sheet.index else None,
                "total_equity": balance_sheet.loc["Total Equity"].iloc[0] if "Total Equity" in balance_sheet.index else None,
                "cash": balance_sheet.loc["Cash"].iloc[0] if "Cash" in balance_sheet.index else None,
                "debt": balance_sheet.loc["Total Debt"].iloc[0] if "Total Debt" in balance_sheet.index else None
            }
        except:
            return {}
    
    def _process_cashflow(self, cashflow: pd.DataFrame) -> Dict[str, Any]:
        """Process cash flow data"""
        if cashflow.empty:
            return {}
        
        try:
            return {
                "operating_cash_flow": cashflow.loc["Operating Cash Flow"].iloc[0] if "Operating Cash Flow" in cashflow.index else None,
                "free_cash_flow": cashflow.loc["Free Cash Flow"].iloc[0] if "Free Cash Flow" in cashflow.index else None,
                "capital_expenditure": cashflow.loc["Capital Expenditure"].iloc[0] if "Capital Expenditure" in cashflow.index else None
            }
        except:
            return {}
    
    def _calculate_portfolio_summary(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate portfolio summary statistics"""
        total_value = 0
        total_gain_loss = 0
        symbols = []
        
        for symbol, data in portfolio_data.items():
            if "error" not in data and data.get("current_price"):
                total_value += data["current_price"]
                symbols.append(symbol)
        
        return {
            "total_symbols": len(symbols),
            "symbols": symbols,
            "total_value": total_value,
            "last_updated": datetime.now().isoformat()
        } 