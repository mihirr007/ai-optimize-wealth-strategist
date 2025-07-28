# Define the order and mapping of wealth management agents
ANALYST_ORDER = [
    ("Passive Indexing Agent", "passive_indexing_agent"),
    ("Dividend Growth Agent", "dividend_growth_agent"),
    ("ESG Agent", "esg_agent"),
    ("Factor Investing Agent", "factor_investing_agent"),
    ("Global Macro Agent", "global_macro_agent"),
    ("Tactical Allocation Agent", "tactical_allocation_agent"),
    ("Canadian Core Agent", "canadian_core_agent"),
    ("Risk Profiler Agent", "risk_profiler_agent"),
    ("Tax Optimization Agent", "tax_optimization_agent"),
    ("Estate Planning Agent", "estate_planning_agent"),
    ("Retirement Planner Agent", "retirement_planner_agent"),
    ("Insurance Planning Agent", "insurance_planning_agent"),
    ("Debt Strategy Agent", "debt_strategy_agent"),
    ("Portfolio Auditor Agent", "portfolio_auditor_agent"),
    ("Rebalancer Agent", "rebalancer_agent"),
    ("Sentiment & Market Context Agent", "sentiment_market_context_agent"),
    ("Portfolio Manager Agent", "portfolio_manager"),
]


def get_analyst_nodes():
    """Get mapping of analyst keys to their functions"""
    # Import agent functions here to avoid circular imports
    from agents.passive_indexing import passive_indexing_agent
    from agents.dividend_growth import dividend_growth_agent
    from agents.esg import esg_agent
    from agents.factor_investing import factor_investing_agent
    from agents.global_macro import global_macro_agent
    from agents.tactical_allocation import tactical_allocation_agent
    from agents.canadian_core import canadian_core_agent
    from agents.risk_profiler import risk_profiler_agent
    from agents.tax_optimization import tax_optimization_agent
    from agents.estate_planning import estate_planning_agent
    from agents.retirement_planner import retirement_planner_agent
    from agents.insurance_planning import insurance_planning_agent
    from agents.debt_strategy import debt_strategy_agent
    from agents.portfolio_auditor import portfolio_auditor_agent
    from agents.rebalancer import rebalancer_agent
    from agents.sentiment_market_context import sentiment_market_context_agent
    from agents.portfolio_manager import portfolio_management_agent
    
    return {
        "passive_indexing_agent": passive_indexing_agent,
        "dividend_growth_agent": dividend_growth_agent,
        "esg_agent": esg_agent,
        "factor_investing_agent": factor_investing_agent,
        "global_macro_agent": global_macro_agent,
        "tactical_allocation_agent": tactical_allocation_agent,
        "canadian_core_agent": canadian_core_agent,
        "risk_profiler_agent": risk_profiler_agent,
        "tax_optimization_agent": tax_optimization_agent,
        "estate_planning_agent": estate_planning_agent,
        "retirement_planner_agent": retirement_planner_agent,
        "insurance_planning_agent": insurance_planning_agent,
        "debt_strategy_agent": debt_strategy_agent,
        "portfolio_auditor_agent": portfolio_auditor_agent,
        "rebalancer_agent": rebalancer_agent,
        "sentiment_market_context_agent": sentiment_market_context_agent,
        "portfolio_manager": portfolio_management_agent,
    }


def get_agents_list():
    """Get list of all available agents"""
    return [display for display, _ in ANALYST_ORDER] 