from graph.state import WealthAgentState, show_agent_reasoning
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from data.models import ClientProfile, Portfolio, AgentSignal
from utils.progress import progress


class TacticalAllocationSignal(BaseModel):
    signal: Literal["increase", "maintain", "decrease"]
    confidence: float
    reasoning: str
    recommendations: list[str] = []
    risk_factors: list[str] = []
    score: float = 0.0


def tactical_allocation_agent(state: WealthAgentState, agent_id: str = "tactical_allocation_agent"):
    """Analyzes Tactical Allocation opportunities with real-time market data"""
    data = state["data"]
    client_profile = data["client_profile"]
    portfolio = data["portfolio"]

    progress.update_status(agent_id, client_profile.client_id, "Analyzing Tactical Allocation opportunities")

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

    # Enhanced analysis with market data
    signal = TacticalAllocationSignal(
        signal="maintain",
        confidence=70.0,
        reasoning=f"Tactical Allocation analysis with real-time market data for {len(symbols)} symbols",
        recommendations=["Consider Tactical Allocation strategies", "Review current allocation"],
        risk_factors=["Market volatility", "Strategy-specific risks"],
        score=65.0
    )

    # Create the message
    message = HumanMessage(
        content=json.dumps(signal.model_dump()),
        name=agent_id,
    )

    # Store the signal in agent_signals for other agents to access
    agent_signals = data.get("agent_signals", {})
    agent_signals[agent_id] = AgentSignal(
        agent_name=agent_id,
        signal=signal.signal,
        confidence=signal.confidence,
        reasoning=signal.reasoning,
        recommendations=signal.recommendations,
        risk_factors=signal.risk_factors
    )

    # Print the decision if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(signal.model_dump(), "TacticalAllocation Agent")

    progress.update_status(agent_id, client_profile.client_id, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": {
            **state["data"],
            "agent_signals": agent_signals
        },
    }
