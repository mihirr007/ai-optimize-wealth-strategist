from graph.state import WealthAgentState, show_agent_reasoning
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
import json
from typing_extensions import Literal
from data.models import ClientProfile, Portfolio, RiskTolerance, AgentSignal
from utils.llm import call_llm
from utils.progress import progress
from data.market_data_service import market_data_service


class RiskProfileSignal(BaseModel):
    signal: Literal["conservative", "moderate", "aggressive"]
    confidence: float
    reasoning: str
    risk_score: float  # 0-100 scale
    recommended_asset_allocation: dict
    risk_factors: list[str]


def risk_profiler_agent(state: WealthAgentState, agent_id: str = "risk_profiler_agent"):
    """Analyzes client risk profile and recommends appropriate asset allocation with real-time market data."""
    data = state["data"]
    client_profile = data["client_profile"]
    portfolio = data["portfolio"]

    progress.update_status(agent_id, client_profile.client_id, "Analyzing risk profile")

    # Get real-time market data for portfolio holdings
    symbols = []
    for account in portfolio.accounts:
        for holding in account.holdings:
            symbols.append(holding.symbol)
    
    market_data = {}
    if symbols:
        progress.update_status(agent_id, client_profile.client_id, "Fetching real-time market data")
        market_data = market_data_service.get_comprehensive_market_data(symbols)

    # Analyze client characteristics
    risk_analysis = analyze_client_risk_factors(client_profile, market_data)
    
    progress.update_status(agent_id, client_profile.client_id, "Calculating risk score")
    
    # Calculate comprehensive risk score with market context
    risk_score = calculate_risk_score(client_profile, risk_analysis, market_data)
    
    progress.update_status(agent_id, client_profile.client_id, "Generating asset allocation")
    
    # Generate recommended asset allocation
    asset_allocation = generate_asset_allocation(risk_score, client_profile, market_data)
    
    progress.update_status(agent_id, client_profile.client_id, "Finalizing recommendations")
    
    # Generate final risk profile signal
    risk_signal = generate_risk_profile_signal(
        client_profile=client_profile,
        portfolio=portfolio,
        analysis_data={
            "client_profile": client_profile.model_dump(),
            "risk_score": risk_score,
            "asset_allocation": asset_allocation,
            "risk_analysis": risk_analysis,
            "market_data": market_data
        },
        state=state,
        agent_id=agent_id
    )

    # Create the risk profiler message
    message = HumanMessage(
        content=json.dumps(risk_signal.model_dump()),
        name=agent_id,
    )

    # Store the signal in agent_signals for other agents to access
    agent_signals = data.get("agent_signals", {})
    agent_signals[agent_id] = AgentSignal(
        agent_name=agent_id,
        signal=risk_signal.signal,
        confidence=risk_signal.confidence,
        reasoning=risk_signal.reasoning,
        recommendations=[],  # Risk profiler doesn't have recommendations field
        risk_factors=risk_signal.risk_factors
    )

    # Print the decision if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(risk_signal.model_dump(), "Risk Profiler")

    progress.update_status(agent_id, client_profile.client_id, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": {
            **state["data"],
            "agent_signals": agent_signals
        },
    }


def analyze_client_risk_factors(client_profile: ClientProfile, market_data: dict = None) -> dict:
    """Analyze various risk factors for the client with market context"""
    risk_factors = {
        "age_factor": analyze_age_risk(client_profile.age),
        "time_horizon_factor": analyze_time_horizon_risk(client_profile.time_horizon),
        "income_stability": analyze_income_stability(client_profile.income),
        "debt_burden": analyze_debt_burden(client_profile.mortgage_balance, client_profile.other_debt, client_profile.income),
        "dependents_factor": analyze_dependents_risk(client_profile.dependents),
        "insurance_coverage": analyze_insurance_coverage(client_profile.life_insurance_coverage, client_profile.disability_insurance),
        "emergency_fund": analyze_emergency_fund(client_profile.emergency_fund_target, client_profile.income)
    }
    
    # Add market-based risk factors if market data is available
    if market_data and "summary" in market_data:
        market_summary = market_data["summary"]
        risk_factors["market_volatility"] = {
            "risk_level": market_summary.get("volatility_level", "moderate"),
            "score": 30 if market_summary.get("volatility_level") == "high" else 10,
            "reasoning": f"Market volatility: {market_summary.get('volatility_level', 'moderate')}"
        }
        risk_factors["market_trend"] = {
            "risk_level": market_summary.get("trend", "sideways"),
            "score": 20 if market_summary.get("trend") == "bearish" else 10,
            "reasoning": f"Market trend: {market_summary.get('trend', 'sideways')}"
        }
    
    return risk_factors


def analyze_age_risk(age: int) -> dict:
    """Analyze risk based on age"""
    if age < 30:
        return {"risk_level": "low", "score": 20, "reasoning": "Young age allows for higher risk tolerance"}
    elif age < 50:
        return {"risk_level": "medium", "score": 50, "reasoning": "Middle age - moderate risk tolerance"}
    elif age < 65:
        return {"risk_level": "medium_high", "score": 70, "reasoning": "Approaching retirement - reduced risk tolerance"}
    else:
        return {"risk_level": "high", "score": 80, "reasoning": "Retirement age - conservative approach recommended"}


def analyze_time_horizon_risk(time_horizon: int) -> dict:
    """Analyze risk based on investment time horizon"""
    if time_horizon > 20:
        return {"risk_level": "low", "score": 20, "reasoning": "Long time horizon allows for higher risk tolerance"}
    elif time_horizon > 10:
        return {"risk_level": "medium", "score": 50, "reasoning": "Medium time horizon - moderate risk tolerance"}
    elif time_horizon > 5:
        return {"risk_level": "medium_high", "score": 70, "reasoning": "Shorter time horizon - reduced risk tolerance"}
    else:
        return {"risk_level": "high", "score": 85, "reasoning": "Short time horizon - conservative approach needed"}


def analyze_income_stability(income: float) -> dict:
    """Analyze income stability risk"""
    if income > 200000:
        return {"risk_level": "low", "score": 20, "reasoning": "High income provides financial stability"}
    elif income > 100000:
        return {"risk_level": "medium", "score": 50, "reasoning": "Good income level"}
    elif income > 60000:
        return {"risk_level": "medium_high", "score": 70, "reasoning": "Moderate income - some risk concerns"}
    else:
        return {"risk_level": "high", "score": 80, "reasoning": "Lower income - higher risk sensitivity"}


def analyze_debt_burden(mortgage: float, other_debt: float, income: float) -> dict:
    """Analyze debt burden risk"""
    total_debt = mortgage + other_debt
    debt_to_income = total_debt / income if income > 0 else 0
    
    if debt_to_income < 0.3:
        return {"risk_level": "low", "score": 20, "reasoning": "Low debt burden"}
    elif debt_to_income < 0.5:
        return {"risk_level": "medium", "score": 50, "reasoning": "Moderate debt burden"}
    elif debt_to_income < 0.7:
        return {"risk_level": "medium_high", "score": 70, "reasoning": "High debt burden - increased risk"}
    else:
        return {"risk_level": "high", "score": 85, "reasoning": "Very high debt burden - conservative approach needed"}


def analyze_dependents_risk(dependents: int) -> dict:
    """Analyze risk based on number of dependents"""
    if dependents == 0:
        return {"risk_level": "low", "score": 20, "reasoning": "No dependents - higher risk tolerance"}
    elif dependents <= 2:
        return {"risk_level": "medium", "score": 50, "reasoning": "Few dependents - moderate risk tolerance"}
    else:
        return {"risk_level": "high", "score": 75, "reasoning": "Multiple dependents - conservative approach recommended"}


def analyze_insurance_coverage(life_coverage: float, disability_insurance: bool) -> dict:
    """Analyze insurance coverage risk"""
    if life_coverage > 1000000 and disability_insurance:
        return {"risk_level": "low", "score": 20, "reasoning": "Comprehensive insurance coverage"}
    elif life_coverage > 500000:
        return {"risk_level": "medium", "score": 50, "reasoning": "Adequate insurance coverage"}
    else:
        return {"risk_level": "high", "score": 75, "reasoning": "Insufficient insurance coverage - higher risk"}


def analyze_emergency_fund(target: float, income: float) -> dict:
    """Analyze emergency fund adequacy"""
    months_coverage = target / (income / 12) if income > 0 else 0
    
    if months_coverage >= 6:
        return {"risk_level": "low", "score": 20, "reasoning": "Adequate emergency fund"}
    elif months_coverage >= 3:
        return {"risk_level": "medium", "score": 50, "reasoning": "Moderate emergency fund"}
    else:
        return {"risk_level": "high", "score": 75, "reasoning": "Insufficient emergency fund - higher risk"}


def calculate_risk_score(client_profile: ClientProfile, risk_analysis: dict, market_data: dict = None) -> float:
    """Calculate comprehensive risk score (0-100) including market data"""
    # Weighted average of risk factors
    weights = {
        "age_factor": 0.12,
        "time_horizon_factor": 0.18,
        "income_stability": 0.12,
        "debt_burden": 0.18,
        "dependents_factor": 0.08,
        "insurance_coverage": 0.08,
        "emergency_fund": 0.08,
        "market_volatility": 0.08,
        "market_trend": 0.08
    }
    
    total_score = 0
    for factor, weight in weights.items():
        if factor in risk_analysis:
            total_score += risk_analysis[factor]["score"] * weight
    
    # Additional market-based adjustments
    if market_data and "summary" in market_data:
        market_summary = market_data["summary"]
        
        # Adjust for market volatility
        if market_summary.get("volatility_level") == "high":
            total_score += 10  # Increase risk score in high volatility
        elif market_summary.get("volatility_level") == "low":
            total_score -= 5   # Decrease risk score in low volatility
        
        # Adjust for market trend
        if market_summary.get("trend") == "bearish":
            total_score += 8   # Increase risk score in bearish market
        elif market_summary.get("trend") == "bullish":
            total_score -= 3   # Slight decrease in bullish market
    
    return min(100, max(0, total_score))


def generate_asset_allocation(risk_score: float, client_profile: ClientProfile, market_data: dict = None) -> dict:
    """Generate recommended asset allocation based on risk score and market conditions"""
    if risk_score < 30:
        # Conservative allocation
        allocation = {
            "canadian_equity": 0.20,
            "us_equity": 0.15,
            "international_equity": 0.10,
            "fixed_income": 0.45,
            "cash": 0.10,
            "alternatives": 0.00
        }
    elif risk_score < 60:
        # Moderate allocation
        allocation = {
            "canadian_equity": 0.30,
            "us_equity": 0.25,
            "international_equity": 0.15,
            "fixed_income": 0.25,
            "cash": 0.05,
            "alternatives": 0.00
        }
    else:
        # Aggressive allocation
        allocation = {
            "canadian_equity": 0.25,
            "us_equity": 0.35,
            "international_equity": 0.20,
            "fixed_income": 0.15,
            "cash": 0.05,
            "alternatives": 0.00
        }
    
    # Adjust allocation based on market conditions
    if market_data and "summary" in market_data:
        market_summary = market_data["summary"]
        
        # Increase cash allocation in high volatility
        if market_summary.get("volatility_level") == "high":
            allocation["cash"] = min(0.20, allocation["cash"] + 0.05)
            allocation["fixed_income"] = max(0.10, allocation["fixed_income"] + 0.05)
            # Reduce equity allocations proportionally
            total_equity = allocation["canadian_equity"] + allocation["us_equity"] + allocation["international_equity"]
            if total_equity > 0:
                reduction = 0.10
                allocation["canadian_equity"] = max(0, allocation["canadian_equity"] - (reduction * allocation["canadian_equity"] / total_equity))
                allocation["us_equity"] = max(0, allocation["us_equity"] - (reduction * allocation["us_equity"] / total_equity))
                allocation["international_equity"] = max(0, allocation["international_equity"] - (reduction * allocation["international_equity"] / total_equity))
        
        # Adjust for market trend
        if market_summary.get("trend") == "bearish":
            allocation["cash"] = min(0.15, allocation["cash"] + 0.03)
            allocation["fixed_income"] = min(0.50, allocation["fixed_income"] + 0.05)
        elif market_summary.get("trend") == "bullish":
            allocation["cash"] = max(0.02, allocation["cash"] - 0.02)
            allocation["us_equity"] = min(0.40, allocation["us_equity"] + 0.03)
    
    return allocation


def generate_risk_profile_signal(
    client_profile: ClientProfile,
    portfolio: Portfolio,
    analysis_data: dict,
    state: WealthAgentState,
    agent_id: str = "risk_profiler_agent",
) -> RiskProfileSignal:
    """Generate risk profile signal using LLM analysis."""
    
    # Create prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Risk Profiler Agent specializing in comprehensive risk assessment for wealth management clients.

Your expertise includes:
- Personal risk tolerance analysis (age, income, dependents, time horizon)
- Financial risk factors (debt levels, emergency funds, insurance coverage)
- Portfolio risk assessment (asset allocation, diversification, volatility)
- Canadian financial context (RRSP, TFSA, provincial considerations)

Provide detailed, actionable risk insights that help create personalized investment strategies.""",
            ),
            (
                "human",
                """Analyze this client's risk profile:

CLIENT AND PORTFOLIO DATA:
{analysis_data}

Please provide your risk assessment in exactly this JSON format:
{{
  "signal": "conservative" | "moderate" | "aggressive",
  "confidence": float between 0 and 100,
  "reasoning": "string with your detailed risk analysis",
  "risk_score": float between 0 and 100,
  "recommended_asset_allocation": {{
    "equity": float between 0 and 1,
    "fixed_income": float between 0 and 1,
    "cash": float between 0 and 1
  }},
  "risk_factors": ["list", "of", "key", "risk", "factors"]
}}

In your reasoning, be specific about:
1. Overall risk tolerance assessment
2. Key risk factors and their impact
3. Portfolio risk analysis
4. Recommendations for risk management
5. Suggested asset allocation changes

Focus on Canadian financial context and practical wealth management strategies.""",
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2)})

    # Default fallback signal in case parsing fails
    def create_default_risk_signal():
        return RiskProfileSignal(
            signal="moderate",
            confidence=50.0,
            reasoning="Error in risk analysis, defaulting to moderate",
            recommended_asset_allocation={"equity": 0.6, "fixed_income": 0.3, "cash": 0.1},
            risk_factors=["Analysis error"]
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=RiskProfileSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_risk_signal,
    ) 