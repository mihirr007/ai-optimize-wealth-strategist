from graph.state import WealthAgentState, show_agent_reasoning
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
import json
from typing_extensions import Literal
from data.models import ClientProfile, Portfolio, AssetClass, AgentSignal
from utils.llm import call_llm_with_model
from utils.progress import progress


class CanadianCoreSignal(BaseModel):
    signal: Literal["increase", "maintain", "decrease"]
    confidence: float
    reasoning: str
    recommendations: list[str] = []
    risk_factors: list[str] = []
    canadian_exposure_score: float = 0.0
    tsx_allocation_recommendation: float = 0.0
    sector_analysis: dict = {}


def canadian_core_agent(state: WealthAgentState, agent_id: str = "canadian_core_agent"):
    """Analyzes Canadian market exposure and provides TSX-focused recommendations with real-time market data"""
    data = state["data"]
    client_profile = data["client_profile"]
    portfolio = data["portfolio"]

    progress.update_status(agent_id, client_profile.client_id, "Analyzing Canadian market exposure")

    # Get real-time market data for portfolio holdings
    symbols = []
    for account in portfolio.accounts:
        for holding in account.holdings:
            symbols.append(holding.symbol)
    
    market_data = {}
    if symbols:
        progress.update_status(agent_id, client_profile.client_id, "Fetching real-time market data")
        from data.market_data_service import market_data_service
        market_data = market_data_service.get_comprehensive_market_data(symbols)
        
        # Print market data summary for this agent
        print(f"📊 [{agent_id.upper()}] Market Data Summary:")
        print(f"   📈 Symbols analyzed: {symbols}")
        print(f"   💰 Price data sources: {list(market_data.keys()) if market_data else 'None'}")
        if market_data:
            for source, data in market_data.items():
                if isinstance(data, dict) and 'error' not in data:
                    if 'price' in data:
                        print(f"   📊 {source}: ${data['price']:.2f}")
                    elif 'data' in data and isinstance(data['data'], dict):
                        for symbol, symbol_data in data['data'].items():
                            if isinstance(symbol_data, dict) and 'price' in symbol_data:
                                print(f"   📊 {source} {symbol}: ${symbol_data['price']:.2f}")

    # Analyze current Canadian exposure
    canadian_analysis = analyze_canadian_exposure(portfolio)
    
    progress.update_status(agent_id, client_profile.client_id, "Evaluating TSX allocation")
    
    # Evaluate optimal TSX allocation
    tsx_allocation = evaluate_tsx_allocation(client_profile, portfolio, canadian_analysis)
    
    progress.update_status(agent_id, client_profile.client_id, "Analyzing sector exposure")
    
    # Analyze Canadian sector exposure
    sector_analysis = analyze_canadian_sectors(portfolio)
    
    progress.update_status(agent_id, client_profile.client_id, "Generating Canadian strategy")
    
    # Generate Canadian investment strategy
    canadian_strategy = generate_canadian_strategy(client_profile, portfolio, canadian_analysis, tsx_allocation, sector_analysis)
    
    progress.update_status(agent_id, client_profile.client_id, "Finalizing recommendations")
    
    # Generate final Canadian core signal
    canadian_signal = generate_canadian_core_signal(
        client_profile=client_profile,
        portfolio=portfolio,
        canadian_analysis=canadian_analysis,
        tsx_allocation=tsx_allocation,
        sector_analysis=sector_analysis,
        canadian_strategy=canadian_strategy,
        state=state,
        agent_id=agent_id
    )

    # Create the Canadian core message
    message = HumanMessage(
        content=json.dumps(canadian_signal.model_dump()),
        name=agent_id,
    )

    # Store the signal in agent_signals for other agents to access
    agent_signals = data.get("agent_signals", {})
    agent_signals[agent_id] = AgentSignal(
        agent_name=agent_id,
        signal=canadian_signal.signal,
        confidence=canadian_signal.confidence,
        reasoning=canadian_signal.reasoning,
        recommendations=canadian_signal.recommendations,
        risk_factors=canadian_signal.risk_factors
    )

    # Print the decision if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(canadian_signal.model_dump(), "Canadian Core Agent")

    progress.update_status(agent_id, client_profile.client_id, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": {
            **state["data"],
            "agent_signals": agent_signals
        },
    }


def analyze_canadian_exposure(portfolio: Portfolio) -> dict:
    """Analyze current Canadian market exposure"""
    
    total_value = portfolio.total_value
    canadian_equity_value = 0
    canadian_fixed_income_value = 0
    canadian_holdings = []
    
    for account in portfolio.accounts:
        for holding in account.holdings:
            if holding.country == "Canada" or holding.symbol.endswith(".TO"):
                canadian_holdings.append(holding)
                if holding.asset_class == AssetClass.CANADIAN_EQUITY:
                    canadian_equity_value += holding.market_value
                elif holding.asset_class == AssetClass.FIXED_INCOME:
                    canadian_fixed_income_value += holding.market_value
    
    # Calculate Canadian exposure percentages
    canadian_equity_exposure = (canadian_equity_value / total_value) * 100 if total_value > 0 else 0
    canadian_fixed_income_exposure = (canadian_fixed_income_value / total_value) * 100 if total_value > 0 else 0
    total_canadian_exposure = canadian_equity_exposure + canadian_fixed_income_exposure
    
    # Calculate Canadian exposure score (0-100)
    # Optimal Canadian equity exposure is typically 20-30% for Canadian investors
    optimal_canadian_equity = 25  # 25% is often recommended
    exposure_score = max(0, 100 - abs(canadian_equity_exposure - optimal_canadian_equity) * 2)
    
    return {
        "total_canadian_exposure": total_canadian_exposure,
        "canadian_equity_exposure": canadian_equity_exposure,
        "canadian_fixed_income_exposure": canadian_fixed_income_exposure,
        "canadian_holdings_count": len(canadian_holdings),
        "exposure_score": exposure_score,
        "optimal_canadian_equity": optimal_canadian_equity,
        "canadian_holdings": [h.symbol for h in canadian_holdings]
    }


def evaluate_tsx_allocation(client_profile: ClientProfile, portfolio: Portfolio, canadian_analysis: dict) -> dict:
    """Evaluate optimal TSX allocation based on client profile"""
    
    # Base TSX allocation recommendations by age
    if client_profile.age < 30:
        base_tsx_allocation = 30  # Higher equity exposure for young investors
    elif client_profile.age < 50:
        base_tsx_allocation = 25  # Moderate equity exposure
    elif client_profile.age < 65:
        base_tsx_allocation = 20  # Reduced equity exposure approaching retirement
    else:
        base_tsx_allocation = 15  # Conservative equity exposure in retirement
    
    # Adjust based on risk tolerance
    risk_adjustment = {
        "conservative": -5,
        "moderate": 0,
        "aggressive": 5
    }
    
    adjusted_tsx_allocation = base_tsx_allocation + risk_adjustment.get(client_profile.risk_tolerance.value, 0)
    adjusted_tsx_allocation = max(10, min(40, adjusted_tsx_allocation))  # Clamp between 10-40%
    
    # Calculate current vs recommended allocation
    current_tsx_allocation = canadian_analysis["canadian_equity_exposure"]
    allocation_gap = adjusted_tsx_allocation - current_tsx_allocation
    
    return {
        "recommended_tsx_allocation": adjusted_tsx_allocation,
        "current_tsx_allocation": current_tsx_allocation,
        "allocation_gap": allocation_gap,
        "allocation_action": "increase" if allocation_gap > 5 else "decrease" if allocation_gap < -5 else "maintain",
        "reasoning": f"Age {client_profile.age} with {client_profile.risk_tolerance.value} risk tolerance"
    }


def analyze_canadian_sectors(portfolio: Portfolio) -> dict:
    """Analyze Canadian sector exposure"""
    
    sector_exposure = {}
    canadian_holdings = []
    
    # Collect Canadian holdings
    for account in portfolio.accounts:
        for holding in account.holdings:
            if holding.country == "Canada" or holding.symbol.endswith(".TO"):
                canadian_holdings.append(holding)
    
    # Analyze sector exposure (simplified - in practice would use real sector data)
    for holding in canadian_holdings:
        sector = holding.sector or "Unknown"
        if sector not in sector_exposure:
            sector_exposure[sector] = 0
        sector_exposure[sector] += holding.market_value
    
    # Calculate sector percentages
    total_canadian_value = sum(sector_exposure.values())
    sector_percentages = {}
    for sector, value in sector_exposure.items():
        sector_percentages[sector] = (value / total_canadian_value) * 100 if total_canadian_value > 0 else 0
    
    # Identify concentration risks
    concentration_risks = []
    for sector, percentage in sector_percentages.items():
        if percentage > 30:  # More than 30% in one sector
            concentration_risks.append(f"High concentration in {sector} ({percentage:.1f}%)")
    
    return {
        "sector_exposure": sector_exposure,
        "sector_percentages": sector_percentages,
        "concentration_risks": concentration_risks,
        "diversification_score": max(0, 100 - len(concentration_risks) * 20)
    }


def generate_canadian_strategy(client_profile: ClientProfile, portfolio: Portfolio,
                              canadian_analysis: dict, tsx_allocation: dict,
                              sector_analysis: dict) -> dict:
    """Generate Canadian investment strategy"""
    
    strategy = {
        "allocation_targets": {
            "canadian_equity": tsx_allocation["recommended_tsx_allocation"],
            "canadian_fixed_income": 15,  # Base Canadian bond allocation
            "international_equity": 60 - tsx_allocation["recommended_tsx_allocation"]  # Remainder
        },
        "recommended_etfs": {
            "canadian_equity": ["XIC.TO", "VCN.TO", "ZCN.TO"],  # TSX Composite ETFs
            "canadian_fixed_income": ["XBB.TO", "VAB.TO", "ZAG.TO"],  # Canadian Bond ETFs
            "canadian_dividend": ["XDV.TO", "VDY.TO", "ZDV.TO"]  # Canadian Dividend ETFs
        },
        "sector_recommendations": {
            "overweight": ["Financials", "Energy", "Materials"],  # Traditional Canadian strengths
            "underweight": ["Technology", "Healthcare"],  # Areas where Canada is less competitive
            "avoid": []  # Sectors to avoid
        },
        "specific_actions": []
    }
    
    # Generate specific actions based on analysis
    if tsx_allocation["allocation_gap"] > 5:
        strategy["specific_actions"].append(
            f"Increase Canadian equity exposure by {tsx_allocation['allocation_gap']:.1f}%"
        )
    elif tsx_allocation["allocation_gap"] < -5:
        strategy["specific_actions"].append(
            f"Reduce Canadian equity exposure by {abs(tsx_allocation['allocation_gap']):.1f}%"
        )
    
    # Add sector diversification recommendations
    if sector_analysis["concentration_risks"]:
        strategy["specific_actions"].extend([
            "Diversify sector exposure to reduce concentration risk",
            "Consider broad market ETFs for better sector diversification"
        ])
    
    return strategy


def generate_canadian_core_signal(
    client_profile: ClientProfile,
    portfolio: Portfolio,
    canadian_analysis: dict,
    tsx_allocation: dict,
    sector_analysis: dict,
    canadian_strategy: dict,
    state: WealthAgentState,
    agent_id: str = "canadian_core_agent"
) -> CanadianCoreSignal:
    """Generate final Canadian core signal using LLM reasoning"""
    
    # Prepare data for LLM analysis
    analysis_data = {
        "client_profile": client_profile.model_dump(),
        "portfolio": portfolio.model_dump(),
        "canadian_analysis": canadian_analysis,
        "tsx_allocation": tsx_allocation,
        "sector_analysis": sector_analysis,
        "canadian_strategy": canadian_strategy
    }
    
    # Create prompt for LLM analysis
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Canadian Core Agent specializing in Canadian wealth management. 
        Analyze the client's Canadian market exposure and provide TSX-focused recommendations.
        
        Your role is to:
        1. Assess current Canadian market exposure
        2. Recommend optimal TSX allocation based on client profile
        3. Analyze sector diversification within Canadian holdings
        4. Suggest Canadian-specific investment strategies
        
        Respond with a JSON object containing:
        - signal: "increase", "maintain", or "decrease" Canadian exposure
        - confidence: float between 0-100
        - reasoning: detailed explanation
        - recommendations: list of specific actions
        - risk_factors: list of Canadian market risks
        - canadian_exposure_score: calculated exposure score (0-100)
        - tsx_allocation_recommendation: recommended TSX allocation percentage
        - sector_analysis: analysis of Canadian sector exposure"""),
        ("human", "Analyze this client's Canadian market exposure: {analysis_data}")
    ])
    
    # Call LLM for analysis
    llm_response = call_llm_with_model(
        prompt=prompt,
        analysis_data=json.dumps(analysis_data, indent=2),
        model_name=state["metadata"]["model_name"],
        model_provider=state["metadata"]["model_provider"]
    )
    
    try:
        # Parse LLM response
        response_data = json.loads(llm_response)
        return CanadianCoreSignal(**response_data)
    except (json.JSONDecodeError, KeyError):
        # Fallback to calculated values if LLM fails
        signal = tsx_allocation["allocation_action"]
        exposure_score = canadian_analysis["exposure_score"]
        
        return CanadianCoreSignal(
            signal=signal,
            confidence=85.0,
            reasoning=f"Canadian exposure score: {exposure_score:.1f}%, TSX allocation gap: {tsx_allocation['allocation_gap']:.1f}%",
            recommendations=[
                f"Target {tsx_allocation['recommended_tsx_allocation']:.1f}% TSX allocation",
                "Consider Canadian dividend ETFs for income",
                "Diversify across Canadian sectors",
                "Maintain international exposure for global diversification"
            ],
            risk_factors=[
                "Canadian market concentration in financials and energy",
                "Currency risk for Canadian dollar exposure",
                "Commodity price volatility affecting Canadian markets",
                "Interest rate sensitivity of Canadian financial sector"
            ],
            canadian_exposure_score=exposure_score,
            tsx_allocation_recommendation=tsx_allocation["recommended_tsx_allocation"],
            sector_analysis=sector_analysis
        ) 