from graph.state import WealthAgentState, show_agent_reasoning
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from data.models import ClientProfile, Portfolio, AgentSignal
from utils.progress import progress


class SentimentMarketContextSignal(BaseModel):
    signal: Literal["increase", "maintain", "decrease"]
    confidence: float
    reasoning: str
    recommendations: list[str] = []
    risk_factors: list[str] = []
    score: float = 0.0


def sentiment_market_context_agent(state: WealthAgentState, agent_id: str = "sentiment_market_context_agent"):
    """Analyzes Sentiment & Market Context opportunities with real-time market data"""
    data = state["data"]
    client_profile = data["client_profile"]
    portfolio = data["portfolio"]

    progress.update_status(agent_id, client_profile.client_id, "Analyzing Sentiment & Market Context opportunities")

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
    
    # Analyze sentiment data
    if market_data and 'news_sentiment' in market_data:
        sentiment = market_data['news_sentiment']
        positive_pct = sentiment.get('positive', 0)
        negative_pct = sentiment.get('negative', 0)
        overall_sentiment = sentiment.get('overall_sentiment', 'Neutral')
        
        reasoning_parts.append(f"Market sentiment: {positive_pct:.1f}% positive, {negative_pct:.1f}% negative")
        
        if positive_pct > 60:
            signal_type = "increase"
            confidence += 10
            score += 15
            recommendations.append("Consider increasing equity exposure given positive market sentiment")
        elif negative_pct > 40:
            signal_type = "decrease"
            confidence += 5
            score -= 10
            recommendations.append("Consider defensive positioning given negative market sentiment")
            risk_factors.append("High negative sentiment may indicate market stress")
    
    # Analyze economic indicators
    if market_data and 'economic_indicators' in market_data:
        indicators = market_data['economic_indicators']
        if indicators:
            reasoning_parts.append(f"Economic indicators available: {len(indicators)} indicators")
            
            # Check specific indicators
            if 'unemployment_rate' in indicators:
                unemployment = indicators['unemployment_rate']
                if 'value' in unemployment:
                    unemployment_rate = unemployment['value']
                    reasoning_parts.append(f"Unemployment rate: {unemployment_rate}%")
                    if unemployment_rate < 4:
                        recommendations.append("Low unemployment suggests strong economy - favorable for equities")
                    elif unemployment_rate > 6:
                        risk_factors.append("High unemployment may indicate economic weakness")
            
            if 'federal_funds_rate' in indicators:
                fed_rate = indicators['federal_funds_rate']
                if 'value' in fed_rate:
                    rate = fed_rate['value']
                    reasoning_parts.append(f"Federal funds rate: {rate}%")
                    if rate > 5:
                        recommendations.append("High interest rates may favor defensive positioning")
    
    # Analyze sector performance
    if market_data and 'sector_performance' in market_data:
        sectors = market_data['sector_performance']
        if sectors:
            reasoning_parts.append(f"Sector performance data available: {len(sectors)} sectors")
            
            # Find best and worst performing sectors
            if sectors:
                best_sector = max(sectors.items(), key=lambda x: x[1])
                worst_sector = min(sectors.items(), key=lambda x: x[1])
                
                reasoning_parts.append(f"Best sector: {best_sector[0]} ({best_sector[1]:+.2f}%)")
                reasoning_parts.append(f"Worst sector: {worst_sector[0]} ({worst_sector[1]:+.2f}%)")
                
                # Sector rotation recommendations
                if best_sector[1] > 1.0:
                    recommendations.append(f"Consider overweighting {best_sector[0]} sector showing strong momentum")
                if worst_sector[1] < -1.0:
                    recommendations.append(f"Consider underweighting {worst_sector[0]} sector showing weakness")
                
                # Calculate average sector performance
                avg_performance = sum(sectors.values()) / len(sectors)
                reasoning_parts.append(f"Average sector performance: {avg_performance:+.2f}%")
                
                if avg_performance > 0.5:
                    confidence += 5
                    score += 10
                elif avg_performance < -0.5:
                    confidence += 5
                    score -= 10
                    risk_factors.append("Broad sector weakness may indicate market downturn")
    
    # Combine all reasoning
    reasoning = f"Sentiment & Market Context analysis for {len(symbols)} symbols. " + ". ".join(reasoning_parts)
    
    # Adjust signal based on analysis
    if not recommendations:
        recommendations = ["Monitor market conditions", "Review current allocation"]
    if not risk_factors:
        risk_factors = ["Market volatility", "Strategy-specific risks"]
    
    signal = SentimentMarketContextSignal(
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
        show_agent_reasoning(signal.model_dump(), "SentimentMarketContext Agent")

    progress.update_status(agent_id, client_profile.client_id, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": {
            **state["data"],
            "agent_signals": agent_signals
        },
    }
