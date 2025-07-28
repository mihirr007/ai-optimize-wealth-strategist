from graph.state import WealthAgentState, show_agent_reasoning
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from data.models import ClientProfile, Portfolio, AgentSignal
from utils.progress import progress


class GlobalMacroSignal(BaseModel):
    signal: Literal["increase", "maintain", "decrease"]
    confidence: float
    reasoning: str
    recommendations: list[str] = []
    risk_factors: list[str] = []
    score: float = 0.0


def global_macro_agent(state: WealthAgentState, agent_id: str = "global_macro_agent"):
    """Analyzes Global Macro opportunities with real-time market data"""
    data = state["data"]
    client_profile = data["client_profile"]
    portfolio = data["portfolio"]

    progress.update_status(agent_id, client_profile.client_id, "Analyzing Global Macro opportunities")

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
    
    # Analyze economic indicators for macro trends
    if market_data and 'economic_indicators' in market_data:
        indicators = market_data['economic_indicators']
        if indicators:
            reasoning_parts.append(f"Economic indicators analyzed: {len(indicators)} indicators")
            
            # Analyze unemployment rate
            if 'unemployment_rate' in indicators:
                unemployment = indicators['unemployment_rate']
                if 'value' in unemployment:
                    unemployment_rate = unemployment['value']
                    reasoning_parts.append(f"Unemployment rate: {unemployment_rate}%")
                    
                    if unemployment_rate < 4:
                        recommendations.append("Low unemployment suggests strong labor market - favorable for consumer spending")
                        score += 10
                    elif unemployment_rate > 6:
                        recommendations.append("High unemployment suggests economic weakness - consider defensive positioning")
                        risk_factors.append("High unemployment may indicate recession risk")
                        score -= 15
            
            # Analyze inflation (CPI)
            if 'cpi' in indicators:
                cpi = indicators['cpi']
                if 'value' in cpi:
                    cpi_value = cpi['value']
                    reasoning_parts.append(f"CPI: {cpi_value:.1f}")
                    
                    if cpi_value > 320:
                        recommendations.append("High inflation suggests defensive positioning and inflation hedges")
                        risk_factors.append("High inflation may erode real returns")
                        score -= 10
                    elif cpi_value < 280:
                        recommendations.append("Low inflation environment favorable for growth assets")
                        score += 10
            
            # Analyze Federal Funds Rate
            if 'federal_funds_rate' in indicators:
                fed_rate = indicators['federal_funds_rate']
                if 'value' in fed_rate:
                    rate = fed_rate['value']
                    reasoning_parts.append(f"Federal Funds Rate: {rate:.2f}%")
                    
                    if rate > 5:
                        recommendations.append("High interest rates favor defensive positioning and bond allocations")
                        risk_factors.append("High rates may pressure equity valuations")
                        score -= 10
                    elif rate < 2:
                        recommendations.append("Low interest rates favorable for growth and risk assets")
                        score += 15
            
            # Analyze 10-Year Treasury Rate
            if 'treasury_10y' in indicators:
                treasury = indicators['treasury_10y']
                if 'value' in treasury:
                    treasury_rate = treasury['value']
                    reasoning_parts.append(f"10-Year Treasury: {treasury_rate:.2f}%")
                    
                    # Yield curve analysis (simplified)
                    if 'federal_funds_rate' in indicators:
                        fed_rate = indicators['federal_funds_rate'].get('value', 0)
                        spread = treasury_rate - fed_rate
                        reasoning_parts.append(f"Yield spread: {spread:+.2f}%")
                        
                        if spread < 0:
                            recommendations.append("Inverted yield curve suggests recession risk - defensive positioning")
                            risk_factors.append("Yield curve inversion historically precedes recessions")
                            signal_type = "decrease"
                            score -= 20
                        elif spread > 2:
                            recommendations.append("Steep yield curve suggests economic growth - favorable for equities")
                            score += 15
            
            # Analyze GDP growth
            if 'gdp' in indicators:
                gdp = indicators['gdp']
                if 'value' in gdp:
                    gdp_value = gdp['value']
                    reasoning_parts.append(f"GDP: ${gdp_value:.1f}B")
                    
                    # Calculate GDP growth rate (simplified)
                    if 'previous_value' in gdp:
                        prev_gdp = gdp['previous_value']
                        if prev_gdp > 0:
                            gdp_growth = ((gdp_value - prev_gdp) / prev_gdp) * 100
                            reasoning_parts.append(f"GDP growth: {gdp_growth:+.1f}%")
                            
                            if gdp_growth > 2:
                                recommendations.append("Strong GDP growth suggests favorable macro environment")
                                score += 15
                            elif gdp_growth < 0:
                                recommendations.append("Negative GDP growth suggests recession - defensive positioning")
                                risk_factors.append("Negative GDP growth indicates economic contraction")
                                signal_type = "decrease"
                                score -= 20
    
    # Analyze market sentiment for macro context
    if market_data and 'news_sentiment' in market_data:
        sentiment = market_data['news_sentiment']
        positive_pct = sentiment.get('positive', 0)
        negative_pct = sentiment.get('negative', 0)
        
        reasoning_parts.append(f"Market sentiment: {positive_pct:.1f}% positive, {negative_pct:.1f}% negative")
        
        if negative_pct > 60:
            risk_factors.append("Extreme negative sentiment may indicate macro stress")
            score -= 10
        elif positive_pct > 70:
            recommendations.append("Strong positive sentiment suggests favorable macro backdrop")
            score += 10
    
    # Analyze sector performance for macro trends
    if market_data and 'sector_performance' in market_data:
        sectors = market_data['sector_performance']
        if sectors:
            reasoning_parts.append(f"Sector performance: {len(sectors)} sectors")
            
            # Cyclical vs defensive sector analysis
            cyclical_sectors = ['Consumer Discretionary', 'Industrial', 'Materials', 'Energy']
            defensive_sectors = ['Consumer Staples', 'Healthcare', 'Utilities']
            
            cyclical_performance = sum(sectors.get(sector, 0) for sector in cyclical_sectors if sector in sectors)
            defensive_performance = sum(sectors.get(sector, 0) for sector in defensive_sectors if sector in sectors)
            
            reasoning_parts.append(f"Cyclical sectors avg: {cyclical_performance/len(cyclical_sectors):+.2f}%")
            reasoning_parts.append(f"Defensive sectors avg: {defensive_performance/len(defensive_sectors):+.2f}%")
            
            if cyclical_performance > defensive_performance:
                recommendations.append("Cyclical sectors outperforming suggests economic growth")
                score += 10
            else:
                recommendations.append("Defensive sectors outperforming suggests economic caution")
                risk_factors.append("Defensive sector leadership may indicate economic weakness")
                score -= 10
    
    # Combine all reasoning
    reasoning = f"Global Macro analysis for {len(symbols)} symbols. " + ". ".join(reasoning_parts)
    
    # Adjust signal based on analysis
    if not recommendations:
        recommendations = ["Monitor economic indicators", "Maintain current macro allocation"]
    if not risk_factors:
        risk_factors = ["Economic uncertainty", "Macro volatility"]
    
    signal = GlobalMacroSignal(
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
        show_agent_reasoning(signal.model_dump(), "GlobalMacro Agent")

    progress.update_status(agent_id, client_profile.client_id, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": {
            **state["data"],
            "agent_signals": agent_signals
        },
    }
