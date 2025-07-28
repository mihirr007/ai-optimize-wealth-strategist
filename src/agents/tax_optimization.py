from graph.state import WealthAgentState, show_agent_reasoning
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
import json
from typing_extensions import Literal
from data.models import ClientProfile, Portfolio, AccountType, AssetClass, AgentSignal
from utils.llm import call_llm_with_model
from utils.progress import progress


class TaxOptimizationSignal(BaseModel):
    signal: Literal["optimize", "maintain", "review"]
    confidence: float
    reasoning: str
    recommendations: list[str] = []
    risk_factors: list[str] = []
    tax_savings_estimate: float = 0.0
    asset_location_suggestions: dict = {}


def tax_optimization_agent(state: WealthAgentState, agent_id: str = "tax_optimization_agent"):
    """Analyzes tax optimization opportunities for Canadian accounts with real-time market data"""
    data = state["data"]
    client_profile = data["client_profile"]
    portfolio = data["portfolio"]

    progress.update_status(agent_id, client_profile.client_id, "Analyzing tax optimization opportunities")

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

    # Analyze current tax situation
    tax_analysis = analyze_tax_situation(client_profile, portfolio)
    
    progress.update_status(agent_id, client_profile.client_id, "Calculating tax savings potential")
    
    # Calculate potential tax savings
    tax_savings = calculate_tax_savings(client_profile, portfolio, tax_analysis)
    
    progress.update_status(agent_id, client_profile.client_id, "Generating optimization recommendations")
    
    # Generate asset location recommendations
    asset_location = generate_asset_location_recommendations(client_profile, portfolio)
    
    progress.update_status(agent_id, client_profile.client_id, "Finalizing tax strategy")
    
    # Generate final tax optimization signal
    tax_signal = generate_tax_optimization_signal(
        client_profile=client_profile,
        portfolio=portfolio,
        tax_analysis=tax_analysis,
        tax_savings=tax_savings,
        asset_location=asset_location,
        state=state,
        agent_id=agent_id
    )

    # Create the tax optimization message
    message = HumanMessage(
        content=json.dumps(tax_signal.model_dump()),
        name=agent_id,
    )

    # Store the signal in agent_signals for other agents to access
    agent_signals = data.get("agent_signals", {})
    agent_signals[agent_id] = AgentSignal(
        agent_name=agent_id,
        signal=tax_signal.signal,
        confidence=tax_signal.confidence,
        reasoning=tax_signal.reasoning,
        recommendations=tax_signal.recommendations,
        risk_factors=tax_signal.risk_factors
    )

    # Print the decision if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(tax_signal.model_dump(), "Tax Optimization Agent")

    progress.update_status(agent_id, client_profile.client_id, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": {
            **state["data"],
            "agent_signals": agent_signals
        },
    }


def analyze_tax_situation(client_profile: ClientProfile, portfolio: Portfolio) -> dict:
    """Analyze current tax situation and opportunities"""
    
    # Calculate current tax burden
    current_tax_burden = calculate_current_tax_burden(client_profile, portfolio)
    
    # Analyze contribution room utilization
    contribution_analysis = analyze_contribution_rooms(portfolio)
    
    # Analyze asset location efficiency
    location_efficiency = analyze_asset_location_efficiency(portfolio)
    
    # Identify tax-loss harvesting opportunities
    tax_loss_opportunities = identify_tax_loss_opportunities(portfolio)
    
    return {
        "current_tax_burden": current_tax_burden,
        "contribution_analysis": contribution_analysis,
        "location_efficiency": location_efficiency,
        "tax_loss_opportunities": tax_loss_opportunities,
        "province": client_profile.province,
        "tax_bracket": client_profile.tax_bracket
    }


def calculate_current_tax_burden(client_profile: ClientProfile, portfolio: Portfolio) -> dict:
    """Calculate current tax burden across accounts"""
    total_value = portfolio.total_value
    non_reg_value = 0
    rrsp_value = 0
    tfsa_value = 0
    
    for account in portfolio.accounts:
        if account.account_type == AccountType.NON_REGISTERED:
            non_reg_value += account.balance
        elif account.account_type == AccountType.RRSP:
            rrsp_value += account.balance
        elif account.account_type == AccountType.TFSA:
            tfsa_value += account.balance
    
    # Estimate annual tax on non-registered investments (assuming 2% dividend yield)
    estimated_dividends = non_reg_value * 0.02
    estimated_tax = estimated_dividends * client_profile.tax_bracket
    
    return {
        "total_portfolio_value": total_value,
        "non_registered_value": non_reg_value,
        "rrsp_value": rrsp_value,
        "tfsa_value": tfsa_value,
        "estimated_annual_tax": estimated_tax,
        "tax_efficiency_score": (rrsp_value + tfsa_value) / total_value if total_value > 0 else 0
    }


def analyze_contribution_rooms(portfolio: Portfolio) -> dict:
    """Analyze RRSP and TFSA contribution room utilization"""
    rrsp_room = 0
    tfsa_room = 0
    
    for account in portfolio.accounts:
        if account.account_type == AccountType.RRSP and account.contribution_room:
            rrsp_room += account.contribution_room
        elif account.account_type == AccountType.TFSA and account.contribution_room:
            tfsa_room += account.contribution_room
    
    return {
        "rrsp_contribution_room": rrsp_room,
        "tfsa_contribution_room": tfsa_room,
        "total_available_room": rrsp_room + tfsa_room,
        "priority": "high" if rrsp_room > 10000 or tfsa_room > 5000 else "medium"
    }


def analyze_asset_location_efficiency(portfolio: Portfolio) -> dict:
    """Analyze how efficiently assets are located across accounts"""
    location_score = 0
    issues = []
    
    for account in portfolio.accounts:
        for holding in account.holdings:
            # Check if high-dividend stocks are in non-registered accounts
            if (account.account_type == AccountType.NON_REGISTERED and 
                holding.asset_class in [AssetClass.CANADIAN_EQUITY, AssetClass.US_EQUITY]):
                issues.append(f"High-dividend {holding.symbol} in non-registered account")
                location_score -= 1
            
            # Check if bonds are in registered accounts (good)
            elif (account.account_type in [AccountType.RRSP, AccountType.TFSA] and 
                  holding.asset_class == AssetClass.FIXED_INCOME):
                location_score += 1
    
    return {
        "location_score": location_score,
        "issues": issues,
        "efficiency": "good" if location_score >= 0 else "needs_improvement"
    }


def identify_tax_loss_opportunities(portfolio: Portfolio) -> list:
    """Identify potential tax-loss harvesting opportunities"""
    opportunities = []
    
    for account in portfolio.accounts:
        if account.account_type == AccountType.NON_REGISTERED:
            for holding in account.holdings:
                # Check for unrealized losses (simplified calculation)
                if holding.market_value < holding.cost_basis:
                    loss_amount = holding.cost_basis - holding.market_value
                    opportunities.append({
                        "symbol": holding.symbol,
                        "account": account.account_type.value,
                        "unrealized_loss": loss_amount,
                        "potential_tax_savings": loss_amount * 0.5  # 50% inclusion rate
                    })
    
    return opportunities


def calculate_tax_savings(client_profile: ClientProfile, portfolio: Portfolio, tax_analysis: dict) -> float:
    """Calculate potential annual tax savings from optimization"""
    potential_savings = 0
    
    # Tax savings from RRSP contributions
    rrsp_room = tax_analysis["contribution_analysis"]["rrsp_contribution_room"]
    if rrsp_room > 0:
        # Assume max contribution of $30,000 or available room
        max_contribution = min(30000, rrsp_room)
        potential_savings += max_contribution * client_profile.tax_bracket
    
    # Tax savings from TFSA contributions
    tfsa_room = tax_analysis["contribution_analysis"]["tfsa_contribution_room"]
    if tfsa_room > 0:
        # TFSA saves tax on investment income
        potential_savings += tfsa_room * 0.02 * client_profile.tax_bracket  # 2% yield assumption
    
    # Tax savings from tax-loss harvesting
    for opportunity in tax_analysis["tax_loss_opportunities"]:
        potential_savings += opportunity["potential_tax_savings"] * client_profile.tax_bracket
    
    return potential_savings


def generate_asset_location_recommendations(client_profile: ClientProfile, portfolio: Portfolio) -> dict:
    """Generate specific asset location recommendations"""
    recommendations = {
        "rrsp_priorities": [
            "Fixed income investments (bonds, GICs)",
            "High-yield dividend stocks",
            "US equities (withholding tax recovery)",
            "International equities"
        ],
        "tfsa_priorities": [
            "Canadian dividend stocks",
            "Growth stocks with low dividends",
            "Alternative investments",
            "Cash and short-term investments"
        ],
        "non_registered_priorities": [
            "Canadian dividend stocks (eligible dividends)",
            "Growth stocks with low dividends",
            "Index ETFs with low turnover"
        ],
        "moves_to_consider": []
    }
    
    # Generate specific moves based on current holdings
    for account in portfolio.accounts:
        for holding in account.holdings:
            if (account.account_type == AccountType.NON_REGISTERED and 
                holding.asset_class == AssetClass.FIXED_INCOME):
                recommendations["moves_to_consider"].append(
                    f"Move {holding.symbol} from non-registered to RRSP/TFSA"
                )
    
    return recommendations


def generate_tax_optimization_signal(
    client_profile: ClientProfile,
    portfolio: Portfolio,
    tax_analysis: dict,
    tax_savings: float,
    asset_location: dict,
    state: WealthAgentState,
    agent_id: str = "tax_optimization_agent"
) -> TaxOptimizationSignal:
    """Generate final tax optimization signal using LLM reasoning"""
    
    # Prepare data for LLM analysis
    analysis_data = {
        "client_profile": client_profile.model_dump(),
        "portfolio": portfolio.model_dump(),
        "tax_analysis": tax_analysis,
        "potential_tax_savings": tax_savings,
        "asset_location_recommendations": asset_location
    }
    
    # Create prompt for LLM analysis
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Tax Optimization Agent specializing in Canadian wealth management. 
        Analyze the client's tax situation and provide optimization recommendations.
        
        Your role is to:
        1. Identify tax optimization opportunities
        2. Recommend asset location strategies
        3. Suggest contribution strategies
        4. Identify tax-loss harvesting opportunities
        
        Respond with a JSON object containing:
        - signal: "optimize", "maintain", or "review"
        - confidence: float between 0-100
        - reasoning: detailed explanation
        - recommendations: list of specific actions
        - risk_factors: list of tax-related risks
        - tax_savings_estimate: estimated annual tax savings
        - asset_location_suggestions: specific moves to consider"""),
        ("human", "Analyze this client's tax optimization opportunities: {analysis_data}")
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
        return TaxOptimizationSignal(**response_data)
    except (json.JSONDecodeError, KeyError):
        # Fallback to calculated values if LLM fails
        signal = "optimize" if tax_savings > 1000 else "maintain" if tax_savings > 100 else "review"
        return TaxOptimizationSignal(
            signal=signal,
            confidence=85.0,
            reasoning=f"Potential annual tax savings of ${tax_savings:,.2f} identified",
            recommendations=[
                "Maximize RRSP contributions",
                "Utilize TFSA contribution room",
                "Optimize asset location across accounts",
                "Consider tax-loss harvesting opportunities"
            ],
            risk_factors=[
                "Tax law changes",
                "Contribution limit changes",
                "Market volatility affecting tax-loss harvesting"
            ],
            tax_savings_estimate=tax_savings,
            asset_location_suggestions=asset_location
        ) 