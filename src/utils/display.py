from tabulate import tabulate
from colorama import Fore, Style
from data.models import WealthManagementOutput, AgentSignal, PortfolioRecommendation


def print_wealth_management_output(result):
    """Print wealth management analysis results in a formatted way"""
    if not result or "recommendations" not in result:
        print(f"{Fore.RED}No results to display{Style.RESET_ALL}")
        return

    recommendations = result["recommendations"]
    agent_signals = result.get("agent_signals", {})

    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}AI WEALTH STRATEGIST ANALYSIS RESULTS{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

    # Print agent signals summary
    if agent_signals:
        print(f"\n{Fore.YELLOW}AGENT ANALYSIS SUMMARY:{Style.RESET_ALL}")
        print_agent_signals_table(agent_signals)

    # Print portfolio recommendations
    if "portfolio_recommendations" in recommendations:
        print(f"\n{Fore.YELLOW}PORTFOLIO RECOMMENDATIONS:{Style.RESET_ALL}")
        print_portfolio_recommendations(recommendations["portfolio_recommendations"])

    # Print risk assessment
    if "risk_assessment" in recommendations:
        print(f"\n{Fore.YELLOW}RISK ASSESSMENT:{Style.RESET_ALL}")
        print_risk_assessment(recommendations["risk_assessment"])

    # Print financial plan updates
    if "financial_plan_updates" in recommendations:
        print(f"\n{Fore.YELLOW}FINANCIAL PLAN UPDATES:{Style.RESET_ALL}")
        print_financial_plan_updates(recommendations["financial_plan_updates"])

    # Print compliance checks
    if "compliance_checks" in recommendations:
        print(f"\n{Fore.YELLOW}COMPLIANCE CHECKS:{Style.RESET_ALL}")
        print_compliance_checks(recommendations["compliance_checks"])

    print(f"\n{Fore.GREEN}Analysis completed successfully!{Style.RESET_ALL}")


def print_agent_signals_table(agent_signals):
    """Print agent signals in a formatted table"""
    if not agent_signals:
        print("No agent signals available")
        return

    table_data = []
    for agent_name, signal in agent_signals.items():
        if isinstance(signal, dict):
            # Handle dictionary format
            signal_text = signal.get("signal", "N/A")
            confidence = signal.get("confidence", 0)
            reasoning = signal.get("reasoning", "")[:100] + "..." if len(signal.get("reasoning", "")) > 100 else signal.get("reasoning", "")
        else:
            # Handle AgentSignal object
            signal_text = signal.signal
            confidence = signal.confidence
            reasoning = signal.reasoning[:100] + "..." if len(signal.reasoning) > 100 else signal.reasoning

        table_data.append([
            agent_name.replace("_", " ").title(),
            signal_text.upper(),
            f"{confidence:.1f}%",
            reasoning
        ])

    headers = ["Agent", "Signal", "Confidence", "Reasoning"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def print_portfolio_recommendations(recommendations):
    """Print portfolio recommendations in a formatted table"""
    if not recommendations:
        print("No portfolio recommendations")
        return

    table_data = []
    for rec in recommendations:
        if isinstance(rec, dict):
            # Handle dictionary format
            action = rec.get("action", "N/A")
            symbol = rec.get("symbol", "N/A")
            quantity = rec.get("quantity", "N/A")
            reasoning = rec.get("reasoning", "")[:80] + "..." if len(rec.get("reasoning", "")) > 80 else rec.get("reasoning", "")
            priority = rec.get("priority", "medium")
        else:
            # Handle PortfolioRecommendation object
            action = rec.action
            symbol = rec.symbol or "N/A"
            quantity = rec.quantity or "N/A"
            reasoning = rec.reasoning[:80] + "..." if len(rec.reasoning) > 80 else rec.reasoning
            priority = rec.priority

        table_data.append([
            action.upper(),
            symbol,
            quantity,
            priority.title(),
            reasoning
        ])

    headers = ["Action", "Symbol", "Quantity", "Priority", "Reasoning"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def print_risk_assessment(risk_assessment):
    """Print risk assessment results"""
    if not risk_assessment:
        print("No risk assessment available")
        return

    if isinstance(risk_assessment, dict):
        for key, value in risk_assessment.items():
            if isinstance(value, dict):
                print(f"\n{Fore.BLUE}{key.replace('_', ' ').title()}:{Style.RESET_ALL}")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key.replace('_', ' ').title()}: {sub_value}")
            else:
                print(f"{Fore.BLUE}{key.replace('_', ' ').title()}:{Style.RESET_ALL} {value}")
    else:
        print(risk_assessment)


def print_financial_plan_updates(plan_updates):
    """Print financial plan updates"""
    if not plan_updates:
        print("No financial plan updates")
        return

    if isinstance(plan_updates, dict):
        for key, value in plan_updates.items():
            print(f"\n{Fore.BLUE}{key.replace('_', ' ').title()}:{Style.RESET_ALL}")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key.replace('_', ' ').title()}: {sub_value}")
            elif isinstance(value, list):
                for item in value:
                    print(f"  • {item}")
            else:
                print(f"  {value}")
    else:
        print(plan_updates)


def print_compliance_checks(compliance_checks):
    """Print compliance check results"""
    if not compliance_checks:
        print("All compliance checks passed")
        return

    print(f"{Fore.RED}Compliance Issues Found:{Style.RESET_ALL}")
    for check in compliance_checks:
        print(f"  • {check}")


def print_client_profile(client_profile):
    """Print client profile information"""
    print(f"\n{Fore.CYAN}CLIENT PROFILE:{Style.RESET_ALL}")
    print(f"Name: {client_profile.name}")
    print(f"Age: {client_profile.age}")
    print(f"Risk Tolerance: {client_profile.risk_tolerance.value.title()}")
    print(f"Time Horizon: {client_profile.time_horizon} years")
    print(f"Income: ${client_profile.income:,.2f}")
    print(f"Province: {client_profile.province}")
    print(f"Retirement Age: {client_profile.retirement_age}")
    print(f"Retirement Income Target: ${client_profile.retirement_income_target:,.2f}")


def print_portfolio_summary(portfolio):
    """Print portfolio summary"""
    print(f"\n{Fore.CYAN}PORTFOLIO SUMMARY:{Style.RESET_ALL}")
    print(f"Total Value: ${portfolio.total_value:,.2f}")
    print(f"Number of Accounts: {len(portfolio.accounts)}")
    
    for account in portfolio.accounts:
        print(f"\n{account.account_type.value.upper()}: ${account.balance:,.2f}")
        if account.contribution_room:
            print(f"  Contribution Room: ${account.contribution_room:,.2f}")
        
        for holding in account.holdings:
            print(f"  • {holding.symbol}: {holding.quantity} shares @ ${holding.market_value:,.2f}") 