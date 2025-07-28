from graph.state import WealthAgentState, show_agent_reasoning
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
import json
from typing_extensions import Literal
from data.models import ClientProfile, Portfolio, AgentSignal
from utils.llm import call_llm_with_model
from utils.progress import progress
import math


class RetirementPlannerSignal(BaseModel):
    signal: Literal["on_track", "needs_attention", "critical"]
    confidence: float
    reasoning: str
    recommendations: list[str] = []
    risk_factors: list[str] = []
    retirement_readiness_score: float = 0.0
    required_savings_rate: float = 0.0
    projected_retirement_income: float = 0.0
    retirement_plan: dict = {}


def retirement_planner_agent(state: WealthAgentState, agent_id: str = "retirement_planner_agent"):
    """Analyzes retirement readiness and provides planning recommendations with real-time market data"""
    data = state["data"]
    client_profile = data["client_profile"]
    portfolio = data["portfolio"]

    progress.update_status(agent_id, client_profile.client_id, "Analyzing retirement readiness")

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

    # Calculate retirement readiness
    retirement_analysis = analyze_retirement_readiness(client_profile, portfolio)
    
    progress.update_status(agent_id, client_profile.client_id, "Running retirement simulations")
    
    # Run retirement income projections
    income_projections = project_retirement_income(client_profile, portfolio)
    
    progress.update_status(agent_id, client_profile.client_id, "Calculating required savings")
    
    # Calculate required savings rate
    savings_analysis = calculate_required_savings(client_profile, portfolio, income_projections)
    
    progress.update_status(agent_id, client_profile.client_id, "Generating retirement plan")
    
    # Generate comprehensive retirement plan
    retirement_plan = generate_retirement_plan(client_profile, portfolio, retirement_analysis, income_projections, savings_analysis)
    
    progress.update_status(agent_id, client_profile.client_id, "Finalizing recommendations")
    
    # Generate final retirement planning signal
    retirement_signal = generate_retirement_planning_signal(
        client_profile=client_profile,
        portfolio=portfolio,
        retirement_analysis=retirement_analysis,
        income_projections=income_projections,
        savings_analysis=savings_analysis,
        retirement_plan=retirement_plan,
        state=state,
        agent_id=agent_id
    )

    # Create the retirement planner message
    message = HumanMessage(
        content=json.dumps(retirement_signal.model_dump()),
        name=agent_id,
    )

    # Store the signal in agent_signals for other agents to access
    agent_signals = data.get("agent_signals", {})
    agent_signals[agent_id] = AgentSignal(
        agent_name=agent_id,
        signal=retirement_signal.signal,
        confidence=retirement_signal.confidence,
        reasoning=retirement_signal.reasoning,
        recommendations=retirement_signal.recommendations,
        risk_factors=retirement_signal.risk_factors
    )

    # Print the decision if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(retirement_signal.model_dump(), "Retirement Planner Agent")

    progress.update_status(agent_id, client_profile.client_id, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": {
            **state["data"],
            "agent_signals": agent_signals
        },
    }


def analyze_retirement_readiness(client_profile: ClientProfile, portfolio: Portfolio) -> dict:
    """Analyze current retirement readiness"""
    
    # Calculate years to retirement
    years_to_retirement = client_profile.retirement_age - client_profile.age
    
    # Calculate current retirement savings
    current_savings = portfolio.total_value
    
    # Calculate required retirement savings (simplified 4% rule)
    required_savings = client_profile.retirement_income_target * 25  # 4% withdrawal rate
    
    # Calculate retirement readiness score (0-100)
    readiness_score = min(100, (current_savings / required_savings) * 100) if required_savings > 0 else 0
    
    # Calculate annual savings needed
    annual_savings_needed = (required_savings - current_savings) / years_to_retirement if years_to_retirement > 0 else 0
    
    # Calculate current savings rate
    current_savings_rate = (annual_savings_needed / client_profile.income) * 100 if client_profile.income > 0 else 0
    
    return {
        "years_to_retirement": years_to_retirement,
        "current_savings": current_savings,
        "required_savings": required_savings,
        "readiness_score": readiness_score,
        "annual_savings_needed": annual_savings_needed,
        "current_savings_rate": current_savings_rate,
        "savings_gap": required_savings - current_savings
    }


def project_retirement_income(client_profile: ClientProfile, portfolio: Portfolio) -> dict:
    """Project retirement income using simplified Monte Carlo simulation"""
    
    # Assumptions for projections
    inflation_rate = 0.025  # 2.5% annual inflation
    investment_return = 0.06  # 6% annual return
    retirement_years = 30  # Assume 30 years in retirement
    
    # Calculate future value of current savings
    years_to_retirement = client_profile.retirement_age - client_profile.age
    future_savings = portfolio.total_value * (1 + investment_return) ** years_to_retirement
    
    # Calculate annual retirement income (4% rule)
    annual_retirement_income = future_savings * 0.04
    
    # Adjust for inflation
    inflation_adjusted_income = annual_retirement_income / (1 + inflation_rate) ** years_to_retirement
    
    # Calculate income replacement ratio
    income_replacement_ratio = (inflation_adjusted_income / client_profile.income) * 100 if client_profile.income > 0 else 0
    
    # Run Monte Carlo simulation (simplified)
    simulation_results = run_monte_carlo_simulation(
        current_savings=portfolio.total_value,
        years_to_retirement=years_to_retirement,
        annual_contribution=client_profile.income * 0.15,  # Assume 15% savings rate
        target_income=client_profile.retirement_income_target
    )
    
    return {
        "future_savings": future_savings,
        "annual_retirement_income": annual_retirement_income,
        "inflation_adjusted_income": inflation_adjusted_income,
        "income_replacement_ratio": income_replacement_ratio,
        "monte_carlo_success_rate": simulation_results["success_rate"],
        "confidence_intervals": simulation_results["confidence_intervals"]
    }


def run_monte_carlo_simulation(current_savings: float, years_to_retirement: int, 
                              annual_contribution: float, target_income: float) -> dict:
    """Run simplified Monte Carlo simulation for retirement planning"""
    
    # Simplified Monte Carlo with 1000 scenarios
    scenarios = 1000
    success_count = 0
    
    # Store final portfolio values for confidence intervals
    final_values = []
    
    for _ in range(scenarios):
        # Simulate investment returns (normal distribution around 6% with 15% volatility)
        portfolio_value = current_savings
        
        for year in range(years_to_retirement):
            # Random annual return between -9% and +21% (6% ± 15%)
            annual_return = 0.06 + (0.15 * (hash(str(_) + str(year)) % 100 - 50) / 50)
            portfolio_value = portfolio_value * (1 + annual_return) + annual_contribution
        
        final_values.append(portfolio_value)
        
        # Check if portfolio can sustain target income (4% rule)
        sustainable_income = portfolio_value * 0.04
        if sustainable_income >= target_income:
            success_count += 1
    
    # Calculate confidence intervals
    final_values.sort()
    confidence_25 = final_values[int(0.25 * scenarios)]
    confidence_50 = final_values[int(0.50 * scenarios)]
    confidence_75 = final_values[int(0.75 * scenarios)]
    
    return {
        "success_rate": (success_count / scenarios) * 100,
        "confidence_intervals": {
            "25th_percentile": confidence_25,
            "median": confidence_50,
            "75th_percentile": confidence_75
        }
    }


def calculate_required_savings(client_profile: ClientProfile, portfolio: Portfolio, 
                              income_projections: dict) -> dict:
    """Calculate required savings rate to meet retirement goals"""
    
    years_to_retirement = client_profile.retirement_age - client_profile.age
    target_income = client_profile.retirement_income_target
    current_savings = portfolio.total_value
    
    # Calculate required future savings (simplified)
    required_future_savings = target_income * 25  # 4% rule
    additional_savings_needed = required_future_savings - current_savings
    
    # Calculate required annual savings
    if years_to_retirement > 0:
        # Using future value of annuity formula
        # FV = PMT * ((1 + r)^n - 1) / r
        # PMT = FV * r / ((1 + r)^n - 1)
        r = 0.06  # 6% annual return
        n = years_to_retirement
        required_annual_savings = additional_savings_needed * r / ((1 + r) ** n - 1)
    else:
        required_annual_savings = 0
    
    # Calculate as percentage of income
    required_savings_rate = (required_annual_savings / client_profile.income) * 100 if client_profile.income > 0 else 0
    
    return {
        "required_annual_savings": required_annual_savings,
        "required_savings_rate": required_savings_rate,
        "additional_savings_needed": additional_savings_needed,
        "feasibility": "feasible" if required_savings_rate <= 30 else "challenging" if required_savings_rate <= 50 else "difficult"
    }


def generate_retirement_plan(client_profile: ClientProfile, portfolio: Portfolio,
                            retirement_analysis: dict, income_projections: dict,
                            savings_analysis: dict) -> dict:
    """Generate comprehensive retirement plan"""
    
    plan = {
        "current_status": {
            "readiness_score": retirement_analysis["readiness_score"],
            "years_to_retirement": retirement_analysis["years_to_retirement"],
            "current_savings": retirement_analysis["current_savings"]
        },
        "targets": {
            "required_savings": retirement_analysis["required_savings"],
            "target_income": client_profile.retirement_income_target,
            "required_savings_rate": savings_analysis["required_savings_rate"]
        },
        "projections": {
            "projected_income": income_projections["inflation_adjusted_income"],
            "income_replacement_ratio": income_projections["income_replacement_ratio"],
            "success_probability": income_projections["monte_carlo_success_rate"]
        },
        "strategies": {
            "immediate_actions": [],
            "medium_term_actions": [],
            "long_term_actions": []
        }
    }
    
    # Generate specific strategies based on analysis
    if retirement_analysis["readiness_score"] < 50:
        plan["strategies"]["immediate_actions"].extend([
            "Increase savings rate immediately",
            "Maximize RRSP contributions",
            "Consider working longer or part-time in retirement"
        ])
    elif retirement_analysis["readiness_score"] < 75:
        plan["strategies"]["medium_term_actions"].extend([
            "Optimize investment allocation for retirement",
            "Consider additional income sources",
            "Review retirement age flexibility"
        ])
    else:
        plan["strategies"]["long_term_actions"].extend([
            "Maintain current savings rate",
            "Consider early retirement options",
            "Plan for legacy and estate transfer"
        ])
    
    return plan


def generate_retirement_planning_signal(
    client_profile: ClientProfile,
    portfolio: Portfolio,
    retirement_analysis: dict,
    income_projections: dict,
    savings_analysis: dict,
    retirement_plan: dict,
    state: WealthAgentState,
    agent_id: str = "retirement_planner_agent"
) -> RetirementPlannerSignal:
    """Generate final retirement planning signal using LLM reasoning"""
    
    # Prepare data for LLM analysis
    analysis_data = {
        "client_profile": client_profile.model_dump(),
        "portfolio": portfolio.model_dump(),
        "retirement_analysis": retirement_analysis,
        "income_projections": income_projections,
        "savings_analysis": savings_analysis,
        "retirement_plan": retirement_plan
    }
    
    # Create prompt for LLM analysis
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Retirement Planner Agent specializing in Canadian wealth management. 
        Analyze the client's retirement readiness and provide comprehensive planning recommendations.
        
        Your role is to:
        1. Assess retirement readiness and identify gaps
        2. Recommend savings strategies and investment approaches
        3. Suggest retirement age and income strategies
        4. Identify risks and mitigation strategies
        
        Respond with a JSON object containing:
        - signal: "on_track", "needs_attention", or "critical"
        - confidence: float between 0-100
        - reasoning: detailed explanation
        - recommendations: list of specific actions
        - risk_factors: list of retirement-related risks
        - retirement_readiness_score: calculated readiness score (0-100)
        - required_savings_rate: percentage of income needed to save
        - projected_retirement_income: estimated annual retirement income
        - retirement_plan: comprehensive retirement strategy"""),
        ("human", "Analyze this client's retirement planning needs: {analysis_data}")
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
        return RetirementPlannerSignal(**response_data)
    except (json.JSONDecodeError, KeyError):
        # Fallback to calculated values if LLM fails
        readiness_score = retirement_analysis["readiness_score"]
        if readiness_score >= 75:
            signal = "on_track"
        elif readiness_score >= 50:
            signal = "needs_attention"
        else:
            signal = "critical"
        
        return RetirementPlannerSignal(
            signal=signal,
            confidence=85.0,
            reasoning=f"Retirement readiness score: {readiness_score:.1f}%",
            recommendations=[
                f"Save {savings_analysis['required_savings_rate']:.1f}% of income annually",
                "Maximize RRSP and TFSA contributions",
                "Consider working longer or reducing retirement income needs",
                "Review investment allocation for retirement goals"
            ],
            risk_factors=[
                "Market volatility affecting retirement savings",
                "Inflation eroding purchasing power",
                "Longevity risk (living longer than expected)",
                "Healthcare costs in retirement"
            ],
            retirement_readiness_score=readiness_score,
            required_savings_rate=savings_analysis["required_savings_rate"],
            projected_retirement_income=income_projections["inflation_adjusted_income"],
            retirement_plan=retirement_plan
        ) 