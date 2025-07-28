from graph.state import WealthAgentState, show_agent_reasoning
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
import json
from typing_extensions import Literal
from data.models import ClientProfile, Portfolio, AssetClass, AgentSignal
from utils.llm import call_llm
from utils.progress import progress


class ESGSignal(BaseModel):
    signal: Literal["enhance", "maintain", "review"]
    confidence: float
    reasoning: str
    recommendations: list[str] = []
    risk_factors: list[str] = []
    esg_score: float = 0.0
    sustainability_rating: str = "neutral"
    esg_analysis: dict = {}


def esg_agent(state: WealthAgentState, agent_id: str = "esg_agent"):
    """Analyzes portfolio from environmental, social, and governance perspective with real-time market data"""
    data = state["data"]
    client_profile = data["client_profile"]
    portfolio = data["portfolio"]

    progress.update_status(agent_id, client_profile.client_id, "Analyzing ESG exposure")

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

    # Analyze current ESG characteristics
    esg_analysis = analyze_esg_characteristics(portfolio)
    
    progress.update_status(agent_id, client_profile.client_id, "Calculating ESG scores")
    
    # Calculate portfolio ESG scores
    esg_scores = calculate_esg_scores(portfolio, esg_analysis)
    
    progress.update_status(agent_id, client_profile.client_id, "Identifying ESG opportunities")
    
    # Identify ESG improvement opportunities
    esg_opportunities = identify_esg_opportunities(portfolio, esg_analysis, esg_scores)
    
    progress.update_status(agent_id, client_profile.client_id, "Generating sustainable strategy")
    
    # Generate sustainable investment strategy
    sustainable_strategy = generate_sustainable_strategy(client_profile, portfolio, esg_analysis, esg_scores, esg_opportunities)
    
    progress.update_status(agent_id, client_profile.client_id, "Finalizing ESG recommendations")
    
    # Generate final ESG signal
    esg_signal = generate_esg_signal(
        client_profile=client_profile,
        portfolio=portfolio,
        esg_analysis=esg_analysis,
        esg_scores=esg_scores,
        esg_opportunities=esg_opportunities,
        sustainable_strategy=sustainable_strategy,
        state=state,
        agent_id=agent_id
    )

    # Create the ESG message
    message = HumanMessage(
        content=json.dumps(esg_signal.model_dump()),
        name=agent_id,
    )

    # Store the signal in agent_signals for other agents to access
    agent_signals = data.get("agent_signals", {})
    agent_signals[agent_id] = AgentSignal(
        agent_name=agent_id,
        signal=esg_signal.signal,
        confidence=esg_signal.confidence,
        reasoning=esg_signal.reasoning,
        recommendations=esg_signal.recommendations,
        risk_factors=esg_signal.risk_factors
    )

    # Print the decision if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(esg_signal.model_dump(), "ESG Agent")

    progress.update_status(agent_id, client_profile.client_id, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": {
            **state["data"],
            "agent_signals": agent_signals
        },
    }


def analyze_esg_characteristics(portfolio: Portfolio) -> dict:
    """Analyze current ESG characteristics of the portfolio"""
    
    total_value = portfolio.total_value
    esg_holdings = []
    non_esg_holdings = []
    sector_esg_analysis = {}
    
    # Define ESG-friendly and ESG-risky sectors (simplified)
    esg_friendly_sectors = [
        "Renewable Energy", "Technology", "Healthcare", "Consumer Staples"
    ]
    
    esg_risky_sectors = [
        "Oil & Gas", "Mining", "Tobacco", "Weapons", "Gambling"
    ]
    
    for account in portfolio.accounts:
        for holding in account.holdings:
            # Convert holding to dict for JSON serialization
            holding_dict = {
                "symbol": holding.symbol,
                "name": holding.name,
                "market_value": holding.market_value,
                "esg_score": holding.esg_score,
                "sector": holding.sector,
                "asset_class": holding.asset_class.value if holding.asset_class else None
            }
            
            # Check if holding has ESG score
            if holding.esg_score is not None:
                esg_holdings.append(holding_dict)
            else:
                non_esg_holdings.append(holding_dict)
            
            # Analyze by sector
            sector = holding.sector or "Unknown"
            if sector not in sector_esg_analysis:
                sector_esg_analysis[sector] = {
                    "holdings": [],
                    "total_value": 0,
                    "esg_friendly": sector in esg_friendly_sectors,
                    "esg_risky": sector in esg_risky_sectors
                }
            
            sector_esg_analysis[sector]["holdings"].append(holding_dict)
            sector_esg_analysis[sector]["total_value"] += holding.market_value
    
    # Calculate ESG exposure percentages
    esg_covered_value = sum(h["market_value"] for h in esg_holdings)
    esg_coverage = (esg_covered_value / total_value) * 100 if total_value > 0 else 0
    
    # Calculate sector-based ESG risk
    risky_sector_exposure = 0
    friendly_sector_exposure = 0
    
    for sector, data in sector_esg_analysis.items():
        sector_pct = (data["total_value"] / total_value) * 100 if total_value > 0 else 0
        if data["esg_risky"]:
            risky_sector_exposure += sector_pct
        elif data["esg_friendly"]:
            friendly_sector_exposure += sector_pct
    
    return {
        "esg_holdings": esg_holdings,
        "non_esg_holdings": non_esg_holdings,
        "esg_coverage": esg_coverage,
        "sector_esg_analysis": sector_esg_analysis,
        "risky_sector_exposure": risky_sector_exposure,
        "friendly_sector_exposure": friendly_sector_exposure,
        "total_value": total_value
    }


def calculate_esg_scores(portfolio: Portfolio, esg_analysis: dict) -> dict:
    """Calculate portfolio ESG scores"""
    
    # Calculate weighted average ESG score
    total_value = esg_analysis["total_value"]
    weighted_esg_score = 0
    total_weighted_value = 0
    
    for holding in esg_analysis["esg_holdings"]:
        if holding["esg_score"] is not None:
            weighted_esg_score += holding["market_value"] * holding["esg_score"]
            total_weighted_value += holding["market_value"]
    
    # Calculate average ESG score
    average_esg_score = (weighted_esg_score / total_weighted_value) if total_weighted_value > 0 else 0
    
    # Calculate ESG score based on sector exposure
    sector_esg_score = 50  # Base score
    sector_esg_score += esg_analysis["friendly_sector_exposure"] * 0.5  # Bonus for friendly sectors
    sector_esg_score -= esg_analysis["risky_sector_exposure"] * 0.5     # Penalty for risky sectors
    
    # Calculate overall ESG score (weighted average)
    overall_esg_score = (average_esg_score * 0.7) + (sector_esg_score * 0.3)
    
    # Determine sustainability rating
    if overall_esg_score >= 80:
        sustainability_rating = "excellent"
    elif overall_esg_score >= 70:
        sustainability_rating = "good"
    elif overall_esg_score >= 50:
        sustainability_rating = "neutral"
    elif overall_esg_score >= 30:
        sustainability_rating = "poor"
    else:
        sustainability_rating = "very_poor"
    
    return {
        "average_esg_score": average_esg_score,
        "sector_esg_score": sector_esg_score,
        "overall_esg_score": overall_esg_score,
        "sustainability_rating": sustainability_rating,
        "esg_coverage": esg_analysis["esg_coverage"]
    }


def identify_esg_opportunities(portfolio: Portfolio, esg_analysis: dict, esg_scores: dict) -> dict:
    """Identify ESG improvement opportunities"""
    
    opportunities = {
        "high_priority": [],
        "medium_priority": [],
        "low_priority": [],
        "esg_etf_recommendations": []
    }
    
    # Check for holdings in ESG-risky sectors
    for sector, data in esg_analysis["sector_esg_analysis"].items():
        if data["esg_risky"] and data["total_value"] > 0:
            opportunities["high_priority"].append({
                "action": "consider_replacement",
                "sector": sector,
                "value": data["total_value"],
                "reason": f"High ESG risk sector: {sector}"
            })
    
    # Check for holdings without ESG scores
    if esg_analysis["non_esg_holdings"]:
        opportunities["medium_priority"].append({
            "action": "enhance_esg_data",
            "holdings_count": len(esg_analysis["non_esg_holdings"]),
            "reason": "Holdings without ESG scores"
        })
    
    # Recommend ESG ETFs for better diversification
    if esg_scores["esg_coverage"] < 50:
        opportunities["esg_etf_recommendations"] = [
            "XESG.TO - iShares ESG Aware MSCI Canada Index ETF",
            "VESG.TO - Vanguard ESG Canadian All Cap Index ETF",
            "ZESG.TO - BMO ESG Leaders Index ETF"
        ]
    
    return opportunities


def generate_sustainable_strategy(client_profile: ClientProfile, portfolio: Portfolio,
                                esg_analysis: dict, esg_scores: dict,
                                esg_opportunities: dict) -> dict:
    """Generate sustainable investment strategy"""
    
    strategy = {
        "current_status": {
            "esg_score": esg_scores["overall_esg_score"],
            "sustainability_rating": esg_scores["sustainability_rating"],
            "esg_coverage": esg_scores["esg_coverage"]
        },
        "targets": {
            "target_esg_score": 75,  # Target ESG score
            "target_coverage": 80,   # Target ESG data coverage
            "max_risky_exposure": 10  # Maximum exposure to ESG-risky sectors
        },
        "strategies": {
            "immediate_actions": [],
            "medium_term_actions": [],
            "long_term_actions": []
        },
        "esg_focus_areas": {
            "environmental": ["Climate Change", "Resource Efficiency", "Pollution Prevention"],
            "social": ["Labor Rights", "Community Relations", "Data Privacy"],
            "governance": ["Board Diversity", "Executive Compensation", "Business Ethics"]
        }
    }
    
    # Generate specific strategies based on current ESG score
    if esg_scores["overall_esg_score"] < 50:
        strategy["strategies"]["immediate_actions"].extend([
            "Replace holdings in ESG-risky sectors",
            "Increase allocation to ESG-focused ETFs",
            "Review and enhance ESG data collection"
        ])
    elif esg_scores["overall_esg_score"] < 70:
        strategy["strategies"]["medium_term_actions"].extend([
            "Gradually increase ESG score targets",
            "Consider ESG integration in all new investments",
            "Engage with companies on ESG issues"
        ])
    else:
        strategy["strategies"]["long_term_actions"].extend([
            "Maintain high ESG standards",
            "Consider impact investing opportunities",
            "Lead ESG initiatives in portfolio"
        ])
    
    return strategy


def generate_esg_signal(
    client_profile: ClientProfile,
    portfolio: Portfolio,
    esg_analysis: dict,
    esg_scores: dict,
    esg_opportunities: dict,
    sustainable_strategy: dict,
    state: WealthAgentState,
    agent_id: str = "esg_agent"
) -> ESGSignal:
    """Generate final ESG signal using LLM reasoning"""
    
    # Prepare data for LLM analysis
    analysis_data = {
        "client_profile": client_profile.model_dump(),
        "portfolio": portfolio.model_dump(),
        "esg_analysis": esg_analysis,
        "esg_scores": esg_scores,
        "esg_opportunities": esg_opportunities,
        "sustainable_strategy": sustainable_strategy
    }
    
    # Create prompt template
    template = ChatPromptTemplate.from_messages([
        ("system", """You are an ESG Agent specializing in sustainable investing. 
        Analyze the portfolio from environmental, social, and governance perspectives.
        
        Your role is to:
        1. Assess current ESG characteristics and scores
        2. Identify ESG improvement opportunities
        3. Recommend sustainable investment strategies
        4. Evaluate ESG risks and opportunities
        
        Provide detailed, actionable ESG insights for wealth management.""",
        ),
        ("human", """Analyze this portfolio's ESG characteristics:

PORTFOLIO AND ESG DATA:
{analysis_data}

Please provide your ESG assessment in exactly this JSON format:
{{
  "signal": "enhance" | "maintain" | "review",
  "confidence": float between 0 and 100,
  "reasoning": "string with your detailed ESG analysis",
  "recommendations": ["list", "of", "specific", "recommendations"],
  "risk_factors": ["list", "of", "ESG", "risk", "factors"],
  "esg_score": float between 0 and 100,
  "sustainability_rating": "excellent" | "good" | "neutral" | "poor" | "very_poor",
  "esg_analysis": {{"detailed": "ESG analysis results"}}
}}

In your reasoning, be specific about:
1. Current ESG performance assessment
2. Key ESG risks and opportunities
3. Recommended sustainable strategies
4. Impact on long-term wealth management

Focus on practical ESG integration for Canadian investors.""")
    ])
    
    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2)})
    
    # Default fallback signal in case parsing fails
    def create_default_esg_signal():
        esg_score = esg_scores["overall_esg_score"]
        sustainability_rating = esg_scores["sustainability_rating"]
        
        if esg_score < 50:
            signal = "enhance"
        elif esg_score < 70:
            signal = "review"
        else:
            signal = "maintain"
        
        return ESGSignal(
            signal=signal,
            confidence=85.0,
            reasoning=f"ESG score: {esg_score:.1f}/100, Sustainability rating: {sustainability_rating}",
            recommendations=["Review ESG data coverage", "Consider ESG-focused investments"],
            risk_factors=["ESG regulatory changes", "Climate change impact"],
            esg_score=esg_score,
            sustainability_rating=sustainability_rating,
            esg_analysis=esg_analysis
        )
    
    return call_llm(
        prompt=prompt,
        pydantic_model=ESGSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_esg_signal,
    ) 