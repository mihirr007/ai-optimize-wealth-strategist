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
    reasoning_parts = []
    recommendations = []
    risk_factors = []
    confidence = 70.0
    score = 65.0
    signal_type = "maintain"
    
    # Analyze sentiment for tactical decisions
    if market_data and 'news_sentiment' in market_data:
        sentiment = market_data['news_sentiment']
        positive_pct = sentiment.get('positive', 0)
        negative_pct = sentiment.get('negative', 0)
        overall_sentiment = sentiment.get('overall_sentiment', 'Neutral')
        
        reasoning_parts.append(f"Market sentiment: {positive_pct:.1f}% positive, {negative_pct:.1f}% negative")
        
        if positive_pct > 65:
            signal_type = "increase"
            confidence += 15
            score += 20
            recommendations.append("Strong positive sentiment suggests tactical overweight to equities")
        elif negative_pct > 50:
            signal_type = "decrease"
            confidence += 10
            score -= 15
            recommendations.append("High negative sentiment suggests tactical defensive positioning")
            risk_factors.append("Negative sentiment may indicate market correction")
    
    # Analyze economic indicators for tactical timing
    if market_data and 'economic_indicators' in market_data:
        indicators = market_data['economic_indicators']
        if indicators:
            reasoning_parts.append(f"Economic indicators: {len(indicators)} available")
            
            # Check inflation and interest rates
            if 'cpi' in indicators and 'federal_funds_rate' in indicators:
                cpi = indicators['cpi'].get('value', 0)
                fed_rate = indicators['federal_funds_rate'].get('value', 0)
                
                reasoning_parts.append(f"CPI: {cpi:.1f}, Fed Rate: {fed_rate:.2f}%")
                
                if cpi > 300 and fed_rate > 5:
                    recommendations.append("High inflation and rates suggest defensive tactical allocation")
                    risk_factors.append("Stagflation risk may impact equity returns")
                elif cpi < 250 and fed_rate < 3:
                    recommendations.append("Low inflation and rates favorable for tactical equity overweight")
    
    # Analyze sector performance for tactical rotation
    if market_data and 'sector_performance' in market_data:
        sectors = market_data['sector_performance']
        if sectors:
            reasoning_parts.append(f"Sector performance: {len(sectors)} sectors analyzed")
            
            # Find momentum sectors
            momentum_sectors = [(sector, perf) for sector, perf in sectors.items() if perf > 0.5]
            weak_sectors = [(sector, perf) for sector, perf in sectors.items() if perf < -0.5]
            
            if momentum_sectors:
                best_sector = max(momentum_sectors, key=lambda x: x[1])
                reasoning_parts.append(f"Strong momentum: {best_sector[0]} ({best_sector[1]:+.2f}%)")
                recommendations.append(f"Tactical overweight to {best_sector[0]} sector showing momentum")
            
            if weak_sectors:
                worst_sector = min(weak_sectors, key=lambda x: x[1])
                reasoning_parts.append(f"Weak performance: {worst_sector[0]} ({worst_sector[1]:+.2f}%)")
                recommendations.append(f"Tactical underweight to {worst_sector[0]} sector showing weakness")
            
            # Calculate sector dispersion
            sector_values = list(sectors.values())
            if len(sector_values) > 1:
                dispersion = max(sector_values) - min(sector_values)
                reasoning_parts.append(f"Sector dispersion: {dispersion:.2f}%")
                
                if dispersion > 2.0:
                    recommendations.append("High sector dispersion suggests tactical sector rotation opportunities")
                    confidence += 5
                elif dispersion < 0.5:
                    recommendations.append("Low sector dispersion suggests broad market moves")
    
    # Analyze technical data for tactical timing
    if market_data and 'technical_data' in market_data:
        tech_data = market_data['technical_data']
        if tech_data:
            reasoning_parts.append(f"Technical indicators available for {len(tech_data)} symbols")
            
            # Check for oversold/overbought conditions
            oversold_count = 0
            overbought_count = 0
            
            for symbol, data in tech_data.items():
                if data and 'rsi' in data:
                    rsi = data['rsi']
                    if rsi < 30:
                        oversold_count += 1
                    elif rsi > 70:
                        overbought_count += 1
            
            if oversold_count > 0:
                reasoning_parts.append(f"{oversold_count} symbols showing oversold conditions")
                if oversold_count > len(tech_data) * 0.3:
                    recommendations.append("Multiple oversold conditions suggest tactical buying opportunity")
                    signal_type = "increase"
                    confidence += 10
            
            if overbought_count > 0:
                reasoning_parts.append(f"{overbought_count} symbols showing overbought conditions")
                if overbought_count > len(tech_data) * 0.3:
                    recommendations.append("Multiple overbought conditions suggest tactical profit-taking")
                    signal_type = "decrease"
                    confidence += 10
    
    # Combine all reasoning
    reasoning = f"Tactical Allocation analysis for {len(symbols)} symbols. " + ". ".join(reasoning_parts)
    
    # Adjust signal based on analysis
    if not recommendations:
        recommendations = ["Monitor market conditions", "Maintain current tactical allocation"]
    if not risk_factors:
        risk_factors = ["Market volatility", "Tactical timing risk"]
    
    signal = TacticalAllocationSignal(
        signal=signal_type,
        confidence=min(confidence, 95.0),
        reasoning=reasoning,
        recommendations=recommendations,
        risk_factors=risk_factors,
        score=max(min(score, 100.0), 0.0)
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
