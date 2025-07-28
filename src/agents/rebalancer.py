from graph.state import WealthAgentState, show_agent_reasoning
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
import json
from typing_extensions import Literal
from data.models import ClientProfile, Portfolio, AssetClass, AgentSignal
from utils.llm import call_llm_with_model
from utils.progress import progress


class RebalancerSignal(BaseModel):
    signal: Literal["rebalance", "monitor", "no_action"]
    confidence: float
    reasoning: str
    recommendations: list[str] = []
    risk_factors: list[str] = []
    drift_score: float = 0.0
    rebalancing_trades: list[dict] = []
    expected_impact: str = "neutral"


def rebalancer_agent(state: WealthAgentState, agent_id: str = "rebalancer_agent"):
    """Analyzes portfolio drift and recommends rebalancing actions with real-time market data"""
    data = state["data"]
    client_profile = data["client_profile"]
    portfolio = data["portfolio"]

    progress.update_status(agent_id, client_profile.client_id, "Analyzing portfolio drift")

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

    # Analyze current portfolio allocation vs targets
    allocation_analysis = analyze_portfolio_allocation(portfolio)
    
    progress.update_status(agent_id, client_profile.client_id, "Calculating drift metrics")
    
    # Calculate portfolio drift
    drift_analysis = calculate_portfolio_drift(portfolio, allocation_analysis)
    
    progress.update_status(agent_id, client_profile.client_id, "Generating rebalancing recommendations")
    
    # Generate rebalancing recommendations
    rebalancing_recommendations = generate_rebalancing_recommendations(portfolio, allocation_analysis, drift_analysis)
    
    progress.update_status(agent_id, client_profile.client_id, "Assessing rebalancing impact")
    
    # Assess impact of rebalancing
    impact_analysis = assess_rebalancing_impact(client_profile, portfolio, rebalancing_recommendations)
    
    progress.update_status(agent_id, client_profile.client_id, "Finalizing rebalancing strategy")
    
    # Generate final rebalancing signal
    rebalancing_signal = generate_rebalancing_signal(
        client_profile=client_profile,
        portfolio=portfolio,
        allocation_analysis=allocation_analysis,
        drift_analysis=drift_analysis,
        rebalancing_recommendations=rebalancing_recommendations,
        impact_analysis=impact_analysis,
        state=state,
        agent_id=agent_id
    )

    # Create the rebalancer message
    message = HumanMessage(
        content=json.dumps(rebalancing_signal.model_dump()),
        name=agent_id,
    )

    # Store the signal in agent_signals for other agents to access
    agent_signals = data.get("agent_signals", {})
    agent_signals[agent_id] = AgentSignal(
        agent_name=agent_id,
        signal=rebalancing_signal.signal,
        confidence=rebalancing_signal.confidence,
        reasoning=rebalancing_signal.reasoning,
        recommendations=rebalancing_signal.recommendations,
        risk_factors=rebalancing_signal.risk_factors
    )

    # Print the decision if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(rebalancing_signal.model_dump(), "Rebalancer Agent")

    progress.update_status(agent_id, client_profile.client_id, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": {
            **state["data"],
            "agent_signals": agent_signals
        },
    }


def analyze_portfolio_allocation(portfolio: Portfolio) -> dict:
    """Analyze current portfolio allocation across asset classes"""
    
    total_value = portfolio.total_value
    current_allocation = {}
    
    # Calculate current allocation by asset class
    for account in portfolio.accounts:
        for holding in account.holdings:
            asset_class = holding.asset_class.value
            if asset_class not in current_allocation:
                current_allocation[asset_class] = 0
            current_allocation[asset_class] += holding.market_value
    
    # Convert to percentages
    allocation_percentages = {}
    for asset_class, value in current_allocation.items():
        allocation_percentages[asset_class] = (value / total_value) * 100 if total_value > 0 else 0
    
    # Get target allocation (use portfolio targets or default)
    target_allocation = portfolio.target_allocation
    if not target_allocation:
        # Default target allocation based on typical Canadian portfolios
        target_allocation = {
            AssetClass.CANADIAN_EQUITY: 25,
            AssetClass.US_EQUITY: 25,
            AssetClass.INTERNATIONAL_EQUITY: 15,
            AssetClass.FIXED_INCOME: 30,
            AssetClass.CASH: 5
        }
    
    # Convert target allocation to percentages
    target_percentages = {}
    for asset_class, target in target_allocation.items():
        target_percentages[asset_class.value] = target
    
    return {
        "current_allocation": allocation_percentages,
        "target_allocation": target_percentages,
        "total_value": total_value
    }


def calculate_portfolio_drift(portfolio: Portfolio, allocation_analysis: dict) -> dict:
    """Calculate portfolio drift from target allocation"""
    
    current_allocation = allocation_analysis["current_allocation"]
    target_allocation = allocation_analysis["target_allocation"]
    
    drift_metrics = {}
    total_drift = 0
    significant_drifts = []
    
    # Calculate drift for each asset class
    for asset_class in set(current_allocation.keys()) | set(target_allocation.keys()):
        current_pct = current_allocation.get(asset_class, 0)
        target_pct = target_allocation.get(asset_class, 0)
        drift = current_pct - target_pct
        drift_metrics[asset_class] = {
            "current": current_pct,
            "target": target_pct,
            "drift": drift,
            "drift_absolute": abs(drift)
        }
        
        total_drift += abs(drift)
        
        # Flag significant drifts (>5% from target)
        if abs(drift) > 5:
            significant_drifts.append({
                "asset_class": asset_class,
                "drift": drift,
                "severity": "high" if abs(drift) > 10 else "medium"
            })
    
    # Calculate overall drift score (0-100, higher = more drift)
    drift_score = min(100, total_drift * 2)  # Scale drift to 0-100
    
    # Determine if rebalancing is needed
    rebalancing_threshold = portfolio.rebalancing_threshold or 0.05  # 5% default
    needs_rebalancing = total_drift > (rebalancing_threshold * 100)
    
    return {
        "drift_metrics": drift_metrics,
        "total_drift": total_drift,
        "drift_score": drift_score,
        "significant_drifts": significant_drifts,
        "needs_rebalancing": needs_rebalancing,
        "rebalancing_threshold": rebalancing_threshold * 100
    }


def generate_rebalancing_recommendations(portfolio: Portfolio, allocation_analysis: dict, 
                                       drift_analysis: dict) -> dict:
    """Generate specific rebalancing recommendations"""
    
    current_allocation = allocation_analysis["current_allocation"]
    target_allocation = allocation_analysis["target_allocation"]
    total_value = allocation_analysis["total_value"]
    
    rebalancing_trades = []
    priority_actions = []
    
    # Generate trades for each asset class
    for asset_class in set(current_allocation.keys()) | set(target_allocation.keys()):
        current_pct = current_allocation.get(asset_class, 0)
        target_pct = target_allocation.get(asset_class, 0)
        drift = current_pct - target_pct
        
        # Only recommend trades for significant drifts (>2%)
        if abs(drift) > 2:
            # Calculate trade amount
            trade_amount = (drift / 100) * total_value
            
            if drift > 0:
                # Overweight - need to sell
                action = "sell"
                priority = "high" if abs(drift) > 10 else "medium"
            else:
                # Underweight - need to buy
                action = "buy"
                priority = "high" if abs(drift) > 10 else "medium"
            
            rebalancing_trades.append({
                "asset_class": asset_class,
                "action": action,
                "amount": abs(trade_amount),
                "percentage": abs(drift),
                "priority": priority,
                "reasoning": f"Rebalance {asset_class} from {current_pct:.1f}% to {target_pct:.1f}%"
            })
            
            priority_actions.append(f"{action.title()} {asset_class} by {abs(drift):.1f}%")
    
    # Sort trades by priority and amount
    rebalancing_trades.sort(key=lambda x: (x["priority"] == "high", x["amount"]), reverse=True)
    
    return {
        "rebalancing_trades": rebalancing_trades,
        "priority_actions": priority_actions,
        "total_trade_value": sum(trade["amount"] for trade in rebalancing_trades),
        "estimated_transaction_costs": sum(trade["amount"] for trade in rebalancing_trades) * 0.001  # 0.1% estimate
    }


def assess_rebalancing_impact(client_profile: ClientProfile, portfolio: Portfolio, 
                             rebalancing_recommendations: dict) -> dict:
    """Assess the impact of rebalancing recommendations"""
    
    total_trade_value = rebalancing_recommendations["total_trade_value"]
    portfolio_value = portfolio.total_value
    
    # Calculate impact metrics
    trade_impact = (total_trade_value / portfolio_value) * 100 if portfolio_value > 0 else 0
    
    # Assess risk impact
    risk_impact = "neutral"
    if trade_impact > 10:
        risk_impact = "high"
    elif trade_impact > 5:
        risk_impact = "medium"
    
    # Assess tax impact (simplified)
    tax_impact = "low"
    if total_trade_value > 50000:  # Large trades may trigger capital gains
        tax_impact = "medium"
    if total_trade_value > 100000:
        tax_impact = "high"
    
    # Assess cost impact
    transaction_costs = rebalancing_recommendations["estimated_transaction_costs"]
    cost_impact = (transaction_costs / portfolio_value) * 100 if portfolio_value > 0 else 0
    
    return {
        "trade_impact": trade_impact,
        "risk_impact": risk_impact,
        "tax_impact": tax_impact,
        "cost_impact": cost_impact,
        "overall_impact": "positive" if trade_impact < 5 else "neutral" if trade_impact < 10 else "negative"
    }


def generate_rebalancing_signal(
    client_profile: ClientProfile,
    portfolio: Portfolio,
    allocation_analysis: dict,
    drift_analysis: dict,
    rebalancing_recommendations: dict,
    impact_analysis: dict,
    state: WealthAgentState,
    agent_id: str = "rebalancer_agent"
) -> RebalancerSignal:
    """Generate final rebalancing signal using LLM reasoning"""
    
    # Prepare data for LLM analysis
    analysis_data = {
        "client_profile": client_profile.model_dump(),
        "portfolio": portfolio.model_dump(),
        "allocation_analysis": allocation_analysis,
        "drift_analysis": drift_analysis,
        "rebalancing_recommendations": rebalancing_recommendations,
        "impact_analysis": impact_analysis
    }
    
    # Create prompt for LLM analysis
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Rebalancer Agent specializing in portfolio rebalancing. 
        Analyze the portfolio drift and provide rebalancing recommendations.
        
        Your role is to:
        1. Assess current portfolio drift from target allocation
        2. Recommend specific rebalancing trades
        3. Evaluate the impact and costs of rebalancing
        4. Determine optimal timing for rebalancing
        
        Respond with a JSON object containing:
        - signal: "rebalance", "monitor", or "no_action"
        - confidence: float between 0-100
        - reasoning: detailed explanation
        - recommendations: list of specific actions
        - risk_factors: list of rebalancing risks
        - drift_score: calculated drift score (0-100)
        - rebalancing_trades: list of specific trades to execute
        - expected_impact: "positive", "neutral", or "negative"""),
        ("human", "Analyze this portfolio's rebalancing needs: {analysis_data}")
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
        return RebalancerSignal(**response_data)
    except (json.JSONDecodeError, KeyError):
        # Fallback to calculated values if LLM fails
        drift_score = drift_analysis["drift_score"]
        needs_rebalancing = drift_analysis["needs_rebalancing"]
        
        if needs_rebalancing and drift_score > 20:
            signal = "rebalance"
        elif needs_rebalancing:
            signal = "monitor"
        else:
            signal = "no_action"
        
        return RebalancerSignal(
            signal=signal,
            confidence=85.0,
            reasoning=f"Portfolio drift score: {drift_score:.1f}%, {'Rebalancing needed' if needs_rebalancing else 'Within acceptable range'}",
            recommendations=rebalancing_recommendations["priority_actions"],
            risk_factors=[
                "Transaction costs reducing net returns",
                "Tax implications of selling positions",
                "Market timing risk during rebalancing",
                "Potential for overtrading"
            ],
            drift_score=drift_score,
            rebalancing_trades=rebalancing_recommendations["rebalancing_trades"],
            expected_impact=impact_analysis["overall_impact"]
        ) 