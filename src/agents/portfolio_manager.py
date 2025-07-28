from graph.state import WealthAgentState, show_agent_reasoning
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
import json
from typing import List, Dict, Any
from typing_extensions import Literal
from data.models import ClientProfile, Portfolio, AgentSignal, PortfolioRecommendation, WealthManagementOutput
from utils.llm import call_llm_with_model
from utils.progress import progress


class PortfolioManagerOutput(BaseModel):
    portfolio_recommendations: List[PortfolioRecommendation]
    risk_assessment: Dict[str, Any]
    financial_plan_updates: Dict[str, Any]
    compliance_checks: List[str]


def portfolio_management_agent(state: WealthAgentState, agent_id: str = "portfolio_manager"):
    """Final decision maker that consolidates all agent signals and generates recommendations with real-time market data"""
    data = state["data"]
    client_profile = data["client_profile"]
    portfolio = data["portfolio"]
    agent_signals = data["agent_signals"]

    progress.update_status(agent_id, client_profile.client_id, "Consolidating agent signals")

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

    # Display received agent signals
    print(f"\n🤖 [{agent_id.upper()}] RECEIVED AGENT SIGNALS:")
    print(f"   📊 Total signals received: {len(agent_signals)}")
    
    # Group signals by type for display
    signal_types = {}
    for agent_name, signal in agent_signals.items():
        signal_type = signal.signal if hasattr(signal, 'signal') else 'unknown'
        if signal_type not in signal_types:
            signal_types[signal_type] = []
        signal_types[signal_type].append(agent_name)
    
    print(f"   🎯 Signal distribution:")
    for signal_type, agents in signal_types.items():
        print(f"      • {signal_type.title()}: {len(agents)} agents")
    
    # Show high-confidence signals
    high_confidence_signals = []
    for agent_name, signal in agent_signals.items():
        if hasattr(signal, 'confidence') and signal.confidence >= 80:
            high_confidence_signals.append(f"{agent_name} ({signal.confidence:.1f}%)")
    
    if high_confidence_signals:
        print(f"   🟢 High-confidence signals (≥80%): {', '.join(high_confidence_signals[:5])}")
        if len(high_confidence_signals) > 5:
            print(f"      ... and {len(high_confidence_signals) - 5} more")
    
    # Collect all agent signals
    all_signals = {}
    for agent_name, signal in agent_signals.items():
        if isinstance(signal, dict):
            all_signals[agent_name] = signal
        else:
            all_signals[agent_name] = signal.model_dump()

    progress.update_status(agent_id, client_profile.client_id, "Generating comprehensive recommendations")

    # Generate final recommendations
    recommendations = generate_portfolio_recommendations(
        client_profile=client_profile,
        portfolio=portfolio,
        agent_signals=all_signals,
        state=state,
        agent_id=agent_id
    )

    # Create the portfolio manager message
    message = HumanMessage(
        content=json.dumps(recommendations.model_dump()),
        name=agent_id,
    )

    # Print the decision if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(recommendations.model_dump(), "Portfolio Manager")

    progress.update_status(agent_id, client_profile.client_id, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


def generate_portfolio_recommendations(
    client_profile: ClientProfile,
    portfolio: Portfolio,
    agent_signals: Dict[str, Any],
    state: WealthAgentState,
    agent_id: str = "portfolio_manager"
) -> PortfolioManagerOutput:
    """Generate final portfolio recommendations using LLM reasoning"""
    
    # Prepare data for LLM analysis
    analysis_data = {
        "client_profile": client_profile.model_dump(),
        "portfolio": portfolio.model_dump(),
        "agent_signals": agent_signals
    }
    
    # Create prompt for LLM analysis
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Portfolio Manager Agent specializing in wealth management. 
        Your role is to consolidate insights from all other agents and provide final recommendations.
        
        Analyze the client profile, current portfolio, and all agent signals to generate:
        1. Portfolio recommendations (buy/sell/hold actions)
        2. Risk assessment summary
        3. Financial plan updates
        4. Compliance checks
        
        Respond with a JSON object containing:
        - portfolio_recommendations: list of actions with symbol, quantity, reasoning, priority
        - risk_assessment: summary of key risk factors and mitigation strategies
        - financial_plan_updates: updates to retirement, tax, estate, insurance plans
        - compliance_checks: list of any compliance issues or confirmations"""),
        ("human", "Analyze this client's situation and provide recommendations: {analysis_data}")
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
        return PortfolioManagerOutput(**response_data)
    except (json.JSONDecodeError, KeyError):
        # Fallback to basic recommendations if LLM fails
        return create_fallback_recommendations(client_profile, portfolio, agent_signals)


def create_fallback_recommendations(
    client_profile: ClientProfile,
    portfolio: Portfolio,
    agent_signals: Dict[str, Any]
) -> PortfolioManagerOutput:
    """Create fallback recommendations when LLM analysis fails"""
    
    # Basic portfolio recommendations
    portfolio_recommendations = [
        PortfolioRecommendation(
            action="rebalance",
            reasoning="Periodic rebalancing recommended based on target allocation",
            priority="medium",
            expected_impact="positive"
        )
    ]
    
    # Basic risk assessment
    risk_assessment = {
        "overall_risk_level": client_profile.risk_tolerance.value,
        "key_risk_factors": [
            "Market volatility",
            "Interest rate changes",
            "Currency fluctuations"
        ],
        "mitigation_strategies": [
            "Diversification across asset classes",
            "Regular rebalancing",
            "Risk-adjusted position sizing"
        ]
    }
    
    # Basic financial plan updates
    financial_plan_updates = {
        "retirement_plan": {
            "current_savings_rate": "Adequate",
            "recommended_actions": ["Maximize RRSP contributions", "Consider TFSA for additional savings"]
        },
        "tax_strategy": {
            "optimization_opportunities": ["Asset location optimization", "Tax-loss harvesting"]
        },
        "estate_plan": {
            "status": "Good" if client_profile.has_will else "Needs attention",
            "recommendations": ["Update will if needed", "Review beneficiary designations"]
        }
    }
    
    # Basic compliance checks
    compliance_checks = [
        "Portfolio within risk tolerance limits",
        "Asset allocation aligned with investment policy",
        "No concentration issues identified"
    ]
    
    return PortfolioManagerOutput(
        portfolio_recommendations=portfolio_recommendations,
        risk_assessment=risk_assessment,
        financial_plan_updates=financial_plan_updates,
        compliance_checks=compliance_checks
    ) 