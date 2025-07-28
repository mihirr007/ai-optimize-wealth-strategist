from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class RiskTolerance(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class AccountType(str, Enum):
    RRSP = "rrsp"
    TFSA = "tfsa"
    NON_REGISTERED = "non_registered"
    RESP = "resp"
    CORPORATE = "corporate"
    LIRA = "lira"
    LIF = "lif"


class AssetClass(str, Enum):
    CANADIAN_EQUITY = "canadian_equity"
    US_EQUITY = "us_equity"
    INTERNATIONAL_EQUITY = "international_equity"
    FIXED_INCOME = "fixed_income"
    CASH = "cash"
    ALTERNATIVES = "alternatives"
    REAL_ESTATE = "real_estate"


class InvestmentStyle(str, Enum):
    PASSIVE_INDEXING = "passive_indexing"
    DIVIDEND_GROWTH = "dividend_growth"
    ESG = "esg"
    FACTOR_INVESTING = "factor_investing"
    TACTICAL = "tactical"
    CANADIAN_CORE = "canadian_core"


class PortfolioHolding(BaseModel):
    symbol: str
    name: str
    quantity: float
    market_value: float
    cost_basis: float
    account_type: AccountType
    asset_class: AssetClass
    sector: Optional[str] = None
    country: Optional[str] = None
    esg_score: Optional[float] = None


class Account(BaseModel):
    account_type: AccountType
    account_number: str
    balance: float
    holdings: List[PortfolioHolding] = []
    contribution_room: Optional[float] = None
    withdrawal_restrictions: Optional[str] = None


class ClientProfile(BaseModel):
    client_id: str
    name: str
    age: int
    risk_tolerance: RiskTolerance
    time_horizon: int  # years
    income: float
    tax_bracket: float
    province: str
    marital_status: str
    dependents: int = 0
    
    # Financial goals
    retirement_age: int
    retirement_income_target: float
    emergency_fund_target: float
    
    # Debt and liabilities
    mortgage_balance: float = 0
    other_debt: float = 0
    
    # Insurance
    life_insurance_coverage: float = 0
    disability_insurance: bool = False
    
    # Estate planning
    estate_value: float = 0
    has_will: bool = False
    has_power_of_attorney: bool = False


class Portfolio(BaseModel):
    client_id: str
    total_value: float
    accounts: List[Account]
    target_allocation: Dict[AssetClass, float] = {}
    rebalancing_threshold: float = 0.05  # 5% drift threshold
    last_rebalance_date: Optional[datetime] = None
    
    # Risk metrics
    volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    
    # Performance
    ytd_return: Optional[float] = None
    one_year_return: Optional[float] = None
    three_year_return: Optional[float] = None


class FinancialPlan(BaseModel):
    client_id: str
    created_date: datetime
    retirement_plan: Dict[str, Any] = {}
    tax_strategy: Dict[str, Any] = {}
    estate_plan: Dict[str, Any] = {}
    insurance_needs: Dict[str, Any] = {}
    debt_strategy: Dict[str, Any] = {}
    
    # Recommendations
    portfolio_recommendations: List[str] = []
    planning_recommendations: List[str] = []
    risk_alerts: List[str] = []


class AgentSignal(BaseModel):
    agent_name: str
    signal: str  # "bullish", "bearish", "neutral", "recommend", "avoid"
    confidence: float  # 0-100
    reasoning: str
    recommendations: List[str] = []
    risk_factors: List[str] = []


class PortfolioRecommendation(BaseModel):
    action: str  # "buy", "sell", "hold", "rebalance", "reallocate"
    symbol: Optional[str] = None
    quantity: Optional[float] = None
    account_type: Optional[AccountType] = None
    reasoning: str
    priority: str = "medium"  # "high", "medium", "low"
    expected_impact: str = "neutral"  # "positive", "negative", "neutral"


class WealthManagementOutput(BaseModel):
    client_id: str
    analysis_date: datetime
    agent_signals: Dict[str, AgentSignal]
    portfolio_recommendations: List[PortfolioRecommendation]
    risk_assessment: Dict[str, Any]
    financial_plan_updates: Dict[str, Any]
    compliance_checks: List[str] = [] 