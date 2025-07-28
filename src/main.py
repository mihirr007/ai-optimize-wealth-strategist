import sys
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
import questionary
from graph.state import WealthAgentState, show_agent_reasoning
from utils.display import print_wealth_management_output
from utils.analysts import ANALYST_ORDER, get_analyst_nodes
from utils.progress import progress
from llm.models import LLM_ORDER, OLLAMA_LLM_ORDER, get_model_info, ModelProvider
from utils.ollama import ensure_ollama_and_model
from data.models import ClientProfile, Portfolio
from utils.visualize import save_graph_as_png
from data.market_data_service import MarketDataService
import argparse
import json
from datetime import datetime

# Load environment variables from .env file
load_dotenv()


def display_client_and_portfolio_info(client_profile: ClientProfile, portfolio: Portfolio):
    """Display comprehensive client and portfolio information."""
    print("=" * 80)
    print("👤 CLIENT PROFILE & PORTFOLIO ANALYSIS")
    print("=" * 80)
    
    # Client Profile Details
    print("\n📋 CLIENT PROFILE:")
    print(f"   👤 Name: {client_profile.name}")
    print(f"   🎂 Age: {client_profile.age} years old")
    print(f"   💰 Annual Income: ${client_profile.income:,.2f}")
    print(f"   🏠 Mortgage Balance: ${client_profile.mortgage_balance:,.2f}")
    print(f"   💳 Other Debt: ${client_profile.other_debt:,.2f}")
    print(f"   👨‍👩‍👧‍👦 Dependents: {client_profile.dependents}")
    print(f"   🎯 Time Horizon: {client_profile.time_horizon} years")
    print(f"   🛡️ Life Insurance: ${client_profile.life_insurance_coverage:,.2f}")
    print(f"   🏥 Disability Insurance: {'Yes' if client_profile.disability_insurance else 'No'}")
    print(f"   💰 Emergency Fund Target: ${client_profile.emergency_fund_target:,.2f}")
    print(f"   ⚖️ Risk Tolerance: {client_profile.risk_tolerance.value}")
    
    # Portfolio Summary
    print(f"\n💼 PORTFOLIO SUMMARY:")
    print(f"   📊 Total Accounts: {len(portfolio.accounts)}")
    print(f"   📈 Total Holdings: {sum(len(account.holdings) for account in portfolio.accounts)}")
    print(f"   💰 Total Market Value: ${sum(holding.market_value for account in portfolio.accounts for holding in account.holdings):,.2f}")
    
    # Account Details
    for i, account in enumerate(portfolio.accounts, 1):
        print(f"\n   📁 Account {i}: {account.account_type.value}")
        print(f"      💰 Market Value: ${sum(holding.market_value for holding in account.holdings):,.2f}")
        print(f"      📈 Holdings: {len(account.holdings)}")
        
        for holding in account.holdings:
            print(f"         • {holding.symbol} ({holding.name})")
            print(f"           💰 Market Value: ${holding.market_value:,.2f}")
            print(f"           📊 Cost Basis: ${holding.cost_basis:,.2f}")
            print(f"           🏢 Sector: {holding.sector}")
            print(f"           🌍 Country: {holding.country}")
            print(f"           📊 Asset Class: {holding.asset_class.value}")
    
    print("\n" + "=" * 80)


def display_comprehensive_market_data(symbols: list[str]):
    """Display comprehensive market data fetched from all sources."""
    print("\n📊 COMPREHENSIVE MARKET DATA ANALYSIS")
    print("=" * 80)
    
    if not symbols:
        print("⚠️  No symbols found in portfolio for market data analysis")
        return
    
    # Check if markets are likely closed
    from datetime import datetime
    now = datetime.now()
    is_weekend = now.weekday() >= 5  # Saturday = 5, Sunday = 6
    is_market_hours = 9 <= now.hour <= 16  # Rough market hours
    
    if is_weekend:
        print("📅 MARKETS CLOSED (Weekend)")
        print("   💡 Using historical data and limited real-time sources")
        print("   📈 For full real-time data, run during market hours (Mon-Fri, 9:30 AM - 4:00 PM ET)")
        print()
    elif not is_market_hours:
        print("📅 MARKETS CLOSED (Outside Market Hours)")
        print("   💡 Some data may be limited or delayed")
        print()
    
    print(f"🔍 Fetching market data for {len(symbols)} symbols...")
    print(f"📈 Symbols: {', '.join(symbols)}")
    
    try:
        from data.market_data_service import market_data_service
        market_data = market_data_service.get_comprehensive_market_data(symbols)
        
        print(f"\n✅ Market data retrieved successfully!")
        print(f"📊 Data sources: {len(market_data.get('available_sources', []))}")
        print(f"❌ Error sources: {len(market_data.get('error_sources', []))}")
        
        # Display detailed price data for each symbol
        if 'price_data' in market_data:
            print(f"\n💰 {'HISTORICAL' if is_weekend else 'REAL-TIME'} PRICE DATA:")
            for symbol, data in market_data['price_data'].items():
                if data and 'price' in data:
                    print(f"   📈 {symbol}: ${data['price']:.2f}")
                    if 'change' in data:
                        change_symbol = "📈" if data['change'] >= 0 else "📉"
                        print(f"      {change_symbol} Change: {data['change']:+.2f} ({data.get('change_percent', 0):+.2f}%)")
                    if 'volume' in data:
                        print(f"      📊 Volume: {data['volume']:,}")
                    if 'market_cap' in data:
                        print(f"      💰 Market Cap: ${data['market_cap']:,.0f}")
                    if 'pe_ratio' in data:
                        print(f"      📊 P/E Ratio: {data['pe_ratio']:.2f}")
        
        # Display ticker-specific news headlines and articles
        if 'ticker_news' in market_data:
            print(f"\n📰 {'LAST WEEK' if is_weekend else 'TICKER-SPECIFIC'} FINANCIAL NEWS:")
            if is_weekend:
                print("   📅 Using last week's data (markets closed on weekend)")
            ticker_news = market_data['ticker_news']
            if ticker_news:
                for ticker, articles in ticker_news.items():
                    if articles:
                        print(f"   📈 {ticker} News ({len(articles)} articles):")
                        for i, article in enumerate(articles[:6], 1):  # Show top 6 articles per ticker
                            title = article.get('title', 'No title')
                            if title == 'No title' and is_weekend:
                                print(f"      {i}. [Weekend - Limited News Coverage]")
                            else:
                                print(f"      {i}. {title}")
                                if article.get('description'):
                                    desc = article['description'][:80] + "..." if len(article['description']) > 80 else article['description']
                                    print(f"         📝 {desc}")
                                if article.get('published_at'):
                                    print(f"         📅 {article['published_at']}")
                        print()
            else:
                print("   📰 No ticker-specific news articles available")
        
        # Display news sentiment with details
        if 'news_sentiment' in market_data:
            print(f"\n📊 MARKET SENTIMENT ANALYSIS:")
            sentiment = market_data['news_sentiment']
            if is_weekend and all(v == 0 for v in [sentiment.get('positive', 0), sentiment.get('neutral', 0), sentiment.get('negative', 0)]):
                print("   📅 Weekend - Using last week's sentiment data")
            else:
                print(f"   😊 Positive: {sentiment.get('positive', 0)}%")
                print(f"   😐 Neutral: {sentiment.get('neutral', 0)}%")
                print(f"   😞 Negative: {sentiment.get('negative', 0)}%")
                print(f"   📊 Overall Sentiment: {sentiment.get('overall_sentiment', 'Neutral')}")
            
            # Show sentiment trends if available
            if 'sentiment_trend' in sentiment:
                print(f"   📈 Trend: {sentiment['sentiment_trend']}")
        
        # Display economic indicators with descriptions
        if 'economic_indicators' in market_data:
            print(f"\n📊 ECONOMIC INDICATORS:")
            indicators = market_data['economic_indicators']
            if indicators:
                if is_weekend:
                    print("   📅 Weekend - Using last week's economic data")
                for indicator, value in indicators.items():
                    # Add descriptions for common indicators
                    descriptions = {
                        'CPI': 'Consumer Price Index (Inflation)',
                        'UNRATE': 'Unemployment Rate',
                        'GDP': 'Gross Domestic Product',
                        'FEDFUNDS': 'Federal Funds Rate',
                        'DGS10': '10-Year Treasury Rate'
                    }
                    desc = descriptions.get(indicator, indicator)
                    print(f"   📈 {desc}: {value}")
            else:
                print("   📊 No economic indicators available")
        
        # Display market volatility and technical indicators
        if 'technical_data' in market_data:
            print(f"\n📈 TECHNICAL ANALYSIS:")
            tech_data = market_data['technical_data']
            for symbol, data in tech_data.items():
                if data:
                    print(f"   📊 {symbol} Technical Indicators:")
                    if 'rsi' in data:
                        print(f"      📊 RSI: {data['rsi']:.2f}")
                    if 'macd' in data:
                        print(f"      📊 MACD: {data['macd']:.2f}")
                    if 'sma_50' in data:
                        print(f"      📊 50-day SMA: ${data['sma_50']:.2f}")
                    if 'sma_200' in data:
                        print(f"      📊 200-day SMA: ${data['sma_200']:.2f}")
        
        # Display sector performance
        if 'sector_performance' in market_data:
            print(f"\n🏢 SECTOR PERFORMANCE:")
            if is_weekend:
                print("   📅 Weekend - Using last week's sector performance data")
            sectors = market_data['sector_performance']
            for sector, performance in sectors.items():
                change_symbol = "📈" if performance >= 0 else "📉"
                print(f"   {change_symbol} {sector}: {performance:+.2f}%")
        
        print(f"\n✅ Market data ready for agent analysis!")
        print(f"📊 Total data points analyzed: {len(market_data.get('price_data', {}))}")
        
        if is_weekend:
            print(f"\n💡 TIP: For full real-time data, run this analysis during market hours:")
            print(f"   📅 Monday-Friday, 9:30 AM - 4:00 PM Eastern Time")
            print(f"   🌍 Markets: NYSE, NASDAQ, TSX")
        
    except Exception as e:
        print(f"❌ Error fetching market data: {str(e)}")
        print("⚠️  Continuing with limited market data...")


def save_analysis_to_files(client_profile: ClientProfile, portfolio: Portfolio, agent_signals: dict, final_recommendations: dict, market_data: dict = None):
    """Save complete analysis to JSON and detailed Markdown files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_dir = "analysis_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for saving
    analysis_data = {
        "timestamp": timestamp,
        "client_profile": client_profile.model_dump(),
        "portfolio": portfolio.model_dump(),
        "agent_signals": {name: signal.model_dump() for name, signal in agent_signals.items()},
        "final_recommendations": final_recommendations,
        "market_data": market_data,
        "summary": {
            "total_agents": len(agent_signals),
            "avg_confidence": sum(s.confidence for s in agent_signals.values()) / len(agent_signals) if agent_signals else 0,
            "signal_distribution": {}
        }
    }
    
    # Calculate signal distribution
    signal_groups = {}
    for agent_name, signal in agent_signals.items():
        signal_type = signal.signal
        if signal_type not in signal_groups:
            signal_groups[signal_type] = []
        signal_groups[signal_type].append(agent_name)
    
    analysis_data["summary"]["signal_distribution"] = {
        signal_type: {
            "count": len(agents),
            "percentage": (len(agents) / len(agent_signals)) * 100,
            "agents": agents
        }
        for signal_type, agents in signal_groups.items()
    }
    
    # Save as JSON (complete data preservation)
    json_filename = f"{output_dir}/wealth_analysis_{timestamp}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False, default=str)
    
    # Save as detailed Markdown
    md_filename = f"{output_dir}/wealth_analysis_{timestamp}.md"
    with open(md_filename, 'w', encoding='utf-8') as f:
        f.write(generate_detailed_markdown_report(analysis_data))
    
    print(f"\n💾 ANALYSIS SAVED TO FILES:")
    print(f"   📄 Complete Data (JSON): {json_filename}")
    print(f"   📝 Detailed Report (MD): {md_filename}")
    
    return {
        "json_file": json_filename,
        "markdown_file": md_filename
    }


def generate_detailed_markdown_report(data: dict) -> str:
    """Generate a detailed markdown report with comprehensive analysis and market data"""
    md = []
    
    # Header
    md.append("# 🤖 AI Wealth Management Analysis Report")
    md.append(f"**Generated:** {data['timestamp']}")
    md.append("")
    
    # Executive Summary
    md.append("## 📋 Executive Summary")
    portfolio = data['portfolio']
    total_value = sum(holding['market_value'] for account in portfolio['accounts'] for holding in account['holdings'])
    summary = data['summary']
    
    md.append(f"**Portfolio Value:** ${total_value:,.2f}")
    md.append(f"**Total Agents Analyzed:** {summary['total_agents']}")
    md.append(f"**Average Confidence:** {summary['avg_confidence']:.1f}%")
    
    # Signal summary
    if summary['signal_distribution']:
        most_common = max(summary['signal_distribution'].items(), key=lambda x: x[1]['count'])
        md.append(f"**Primary Signal:** {most_common[0].title()} ({most_common[1]['count']} agents)")
    
    md.append("")
    
    # Client Profile
    md.append("## 👤 Client Profile")
    client = data['client_profile']
    md.append(f"- **Name:** {client['name']}")
    md.append(f"- **Age:** {client['age']} years")
    md.append(f"- **Income:** ${client['income']:,.2f}")
    md.append(f"- **Risk Tolerance:** {client['risk_tolerance']}")
    md.append(f"- **Time Horizon:** {client['time_horizon']} years")
    md.append(f"- **Tax Bracket:** {client['tax_bracket']:.1%}")
    md.append(f"- **Province:** {client['province']}")
    md.append(f"- **Marital Status:** {client['marital_status']}")
    md.append(f"- **Dependents:** {client['dependents']}")
    md.append(f"- **Retirement Age:** {client['retirement_age']} years")
    md.append(f"- **Retirement Income Target:** ${client['retirement_income_target']:,.2f}")
    md.append(f"- **Emergency Fund Target:** ${client['emergency_fund_target']:,.2f}")
    md.append(f"- **Mortgage Balance:** ${client['mortgage_balance']:,.2f}")
    md.append(f"- **Other Debt:** ${client['other_debt']:,.2f}")
    md.append(f"- **Life Insurance Coverage:** ${client['life_insurance_coverage']:,.2f}")
    md.append(f"- **Disability Insurance:** {'Yes' if client['disability_insurance'] else 'No'}")
    md.append(f"- **Estate Value:** ${client['estate_value']:,.2f}")
    md.append(f"- **Has Will:** {'Yes' if client['has_will'] else 'No'}")
    md.append(f"- **Has Power of Attorney:** {'Yes' if client['has_power_of_attorney'] else 'No'}")
    md.append("")
    
    # Portfolio Analysis
    md.append("## 💼 Portfolio Analysis")
    md.append(f"- **Total Value:** ${total_value:,.2f}")
    md.append(f"- **Accounts:** {len(portfolio['accounts'])}")
    md.append(f"- **Holdings:** {sum(len(account['holdings']) for account in portfolio['accounts'])}")
    md.append(f"- **Rebalancing Threshold:** {portfolio.get('rebalancing_threshold', 0.05):.1%}")
    md.append("")
    
    # Account Details
    md.append("### 📁 Account Details")
    for i, account in enumerate(portfolio['accounts'], 1):
        account_value = sum(holding['market_value'] for holding in account['holdings'])
        md.append(f"#### Account {i}: {account['account_type']}")
        md.append(f"- **Account Number:** {account['account_number']}")
        md.append(f"- **Balance:** ${account['balance']:,.2f}")
        md.append(f"- **Market Value:** ${account_value:,.2f}")
        md.append(f"- **Holdings:** {len(account['holdings'])}")
        
        if account.get('contribution_room'):
            md.append(f"- **Contribution Room:** ${account['contribution_room']:,.2f}")
        
        if account.get('withdrawal_restrictions'):
            md.append(f"- **Withdrawal Restrictions:** {account['withdrawal_restrictions']}")
        
        md.append("")
        
        # Holdings in this account
        md.append("**Holdings:**")
        for holding in account['holdings']:
            md.append(f"- **{holding['symbol']}** ({holding['name']})")
            md.append(f"  - Quantity: {holding['quantity']}")
            md.append(f"  - Market Value: ${holding['market_value']:,.2f}")
            md.append(f"  - Cost Basis: ${holding['cost_basis']:,.2f}")
            md.append(f"  - Asset Class: {holding['asset_class']}")
            md.append(f"  - Sector: {holding.get('sector', 'N/A')}")
            md.append(f"  - Country: {holding.get('country', 'N/A')}")
            if holding.get('esg_score'):
                md.append(f"  - ESG Score: {holding['esg_score']}")
            md.append("")
    
    # Market Data Analysis
    if data.get('market_data'):
        md.append("## 📊 Market Data Analysis")
        market_data = data['market_data']
        
        # Price Data
        if market_data.get('price_data'):
            md.append("### 💰 Real-Time Price Data")
            for symbol, price_info in market_data['price_data'].items():
                if isinstance(price_info, dict) and 'price' in price_info:
                    md.append(f"**{symbol}:**")
                    md.append(f"- Current Price: ${price_info['price']:.2f}")
                    if 'change' in price_info:
                        md.append(f"- Change: {price_info['change']}")
                    if 'volume' in price_info:
                        md.append(f"- Volume: {price_info['volume']:,}")
                    if 'market_cap' in price_info:
                        md.append(f"- Market Cap: ${price_info['market_cap']:,.0f}")
                    md.append("")
        
        # News Analysis
        if market_data.get('ticker_news'):
            md.append("### 📰 Company-Specific News")
            md.append(f"**Total News Articles:** {len(market_data['ticker_news'])}")
            md.append("")
            
            # Group news by ticker
            news_by_ticker = {}
            for article in market_data['ticker_news']:
                ticker = article.get('related_ticker', 'Unknown')
                if ticker not in news_by_ticker:
                    news_by_ticker[ticker] = []
                news_by_ticker[ticker].append(article)
            
            for ticker, articles in news_by_ticker.items():
                md.append(f"**{ticker} News ({len(articles)} articles):**")
                for i, article in enumerate(articles[:3], 1):  # Show top 3 per ticker
                    md.append(f"{i}. **{article.get('title', 'No title')}**")
                    md.append(f"   - Source: {article.get('source', {}).get('name', 'Unknown')}")
                    md.append(f"   - Published: {article.get('publishedAt', 'Unknown')}")
                    md.append(f"   - URL: {article.get('url', 'N/A')}")
                md.append("")
        
        # Sentiment Analysis
        if market_data.get('sentiment_data'):
            md.append("### 😊 Market Sentiment Analysis")
            sentiment = market_data['sentiment_data']
            md.append(f"- **Overall Sentiment:** {sentiment.get('overall_sentiment', 'N/A')}")
            md.append(f"- **Sentiment Score:** {sentiment.get('sentiment_score', 'N/A')}")
            if sentiment.get('sentiment_breakdown'):
                md.append("- **Sentiment Breakdown:**")
                for category, score in sentiment['sentiment_breakdown'].items():
                    md.append(f"  - {category}: {score}")
            md.append("")
        
        # Technical Analysis
        if market_data.get('technical_data'):
            md.append("### 📈 Technical Analysis")
            technical = market_data['technical_data']
            for symbol, tech_data in technical.items():
                if isinstance(tech_data, dict):
                    md.append(f"**{symbol} Technical Indicators:**")
                    for indicator, value in tech_data.items():
                        if isinstance(value, (int, float)):
                            md.append(f"- {indicator}: {value:.2f}")
                        else:
                            md.append(f"- {indicator}: {value}")
                    md.append("")
        
        # Economic Indicators
        if market_data.get('economic_indicators'):
            md.append("### 🏛️ Economic Indicators")
            economic = market_data['economic_indicators']
            for indicator, data in economic.items():
                if isinstance(data, dict):
                    md.append(f"**{indicator}:**")
                    for key, value in data.items():
                        md.append(f"- {key}: {value}")
                    md.append("")
    
    # Agent Analysis
    md.append("## 🤖 AI Agent Analysis")
    md.append(f"- **Total Agents:** {summary['total_agents']}")
    md.append(f"- **Average Confidence:** {summary['avg_confidence']:.1f}%")
    md.append("")
    
    # Signal Distribution
    md.append("### 📊 Signal Distribution")
    for signal_type, info in summary['signal_distribution'].items():
        md.append(f"- **{signal_type.title()}:** {info['count']} agents ({info['percentage']:.1f}%)")
        md.append(f"  - Agents: {', '.join(info['agents'])}")
    md.append("")
    
    # Detailed Agent Analysis
    md.append("### 📈 Individual Agent Analysis")
    md.append("")
    
    # Group agents by signal type
    signal_groups = {}
    for agent_name, signal in data['agent_signals'].items():
        signal_type = signal['signal']
        if signal_type not in signal_groups:
            signal_groups[signal_type] = []
        signal_groups[signal_type].append((agent_name, signal))
    
    for signal_type, agents in signal_groups.items():
        md.append(f"#### {signal_type.upper()} SIGNALS ({len(agents)} agents)")
        md.append("")
        
        for agent_name, signal in agents:
            display_name = agent_name.replace('_', ' ').title()
            confidence_icon = "🟢" if signal['confidence'] >= 80 else "🟡" if signal['confidence'] >= 60 else "🔴"
            
            md.append(f"**{confidence_icon} {display_name}**")
            md.append(f"- **Signal:** {signal['signal']}")
            md.append(f"- **Confidence:** {signal['confidence']:.1f}%")
            md.append(f"- **Reasoning:** {signal['reasoning']}")
            
            if signal.get('recommendations'):
                md.append("- **Recommendations:**")
                for rec in signal['recommendations']:
                    md.append(f"  - {rec}")
            
            if signal.get('risk_factors'):
                md.append("- **Risk Factors:**")
                for risk in signal['risk_factors']:
                    md.append(f"  - {risk}")
            
            # Agent-specific metrics
            if signal.get('score'):
                md.append(f"- **Score:** {signal['score']:.1f}")
            if signal.get('risk_score'):
                md.append(f"- **Risk Score:** {signal['risk_score']:.1f}")
            if signal.get('esg_score'):
                md.append(f"- **ESG Score:** {signal['esg_score']:.1f}")
            if signal.get('sustainability_rating'):
                md.append(f"- **Sustainability Rating:** {signal['sustainability_rating']}")
            if signal.get('canadian_exposure_score'):
                md.append(f"- **Canadian Exposure Score:** {signal['canadian_exposure_score']:.1f}")
            if signal.get('retirement_readiness_score'):
                md.append(f"- **Retirement Readiness Score:** {signal['retirement_readiness_score']:.1f}")
            if signal.get('drift_score'):
                md.append(f"- **Drift Score:** {signal['drift_score']:.1f}")
            if signal.get('tax_savings_estimate'):
                md.append(f"- **Tax Savings Estimate:** ${signal['tax_savings_estimate']:,.2f}")
            
            md.append("")
    
    # Final Recommendations
    md.append("## 🎯 Final Recommendations")
    md.append("### Portfolio Manager Analysis")
    md.append("")
    
    if data.get('final_recommendations'):
        recommendations = data['final_recommendations']
        if isinstance(recommendations, dict):
            for key, value in recommendations.items():
                if isinstance(value, list):
                    md.append(f"#### {key.title()}")
                    for item in value:
                        if isinstance(item, dict):
                            for sub_key, sub_value in item.items():
                                md.append(f"- **{sub_key.title()}:** {sub_value}")
                        else:
                            md.append(f"- {item}")
                    md.append("")
                elif isinstance(value, dict):
                    md.append(f"#### {key.title()}")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, list):
                            md.append(f"**{sub_key.title()}:**")
                            for item in sub_value:
                                md.append(f"- {item}")
                        else:
                            md.append(f"**{sub_key.title()}:** {sub_value}")
                    md.append("")
                else:
                    md.append(f"**{key.title()}:** {value}")
                    md.append("")
        else:
            md.append(f"{recommendations}")
            md.append("")
    else:
        md.append("*No final recommendations available*")
        md.append("")
    
    # Risk Assessment
    md.append("## ⚠️ Risk Assessment")
    if data.get('final_recommendations') and isinstance(data['final_recommendations'], dict):
        risk_assessment = data['final_recommendations'].get('risk_assessment', {})
        if isinstance(risk_assessment, dict):
            md.append(f"**Overall Risk Level:** {risk_assessment.get('overall_risk_level', 'N/A')}")
            md.append("")
            
            if risk_assessment.get('key_risk_factors'):
                md.append("**Key Risk Factors:**")
                for risk in risk_assessment['key_risk_factors']:
                    md.append(f"- {risk}")
                md.append("")
            
            if risk_assessment.get('mitigation_strategies'):
                md.append("**Mitigation Strategies:**")
                for strategy in risk_assessment['mitigation_strategies']:
                    md.append(f"- {strategy}")
                md.append("")
    
    # Financial Plan Updates
    md.append("## 📋 Financial Plan Updates")
    if data.get('final_recommendations') and isinstance(data['final_recommendations'], dict):
        plan_updates = data['final_recommendations'].get('financial_plan_updates', {})
        if isinstance(plan_updates, dict):
            for plan_type, plan_data in plan_updates.items():
                md.append(f"### {plan_type.replace('_', ' ').title()}")
                if isinstance(plan_data, dict):
                    for key, value in plan_data.items():
                        if isinstance(value, list):
                            md.append(f"**{key.replace('_', ' ').title()}:**")
                            for item in value:
                                md.append(f"- {item}")
                        else:
                            md.append(f"**{key.replace('_', ' ').title()}:** {value}")
                md.append("")
    
    # Analysis Summary
    md.append("## 📊 Analysis Summary")
    md.append("### Key Insights")
    md.append("")
    
    # Calculate insights
    high_confidence_agents = [name for name, signal in data['agent_signals'].items() if signal['confidence'] >= 80]
    low_confidence_agents = [name for name, signal in data['agent_signals'].items() if signal['confidence'] < 60]
    
    if high_confidence_agents:
        md.append(f"- **High Confidence Agents ({len(high_confidence_agents)}):** {', '.join(high_confidence_agents)}")
    
    if low_confidence_agents:
        md.append(f"- **Low Confidence Agents ({len(low_confidence_agents)}):** {', '.join(low_confidence_agents)}")
    
    if summary['signal_distribution']:
        most_common_signal = max(summary['signal_distribution'].items(), key=lambda x: x[1]['count'])
        md.append(f"- **Most Common Signal:** {most_common_signal[0].title()} ({most_common_signal[1]['count']} agents)")
    
    md.append("")
    md.append("### Next Steps")
    md.append("1. Review all agent recommendations carefully")
    md.append("2. Consider implementing high-confidence recommendations first")
    md.append("3. Address any risk factors identified by agents")
    md.append("4. Schedule a follow-up review in 3-6 months")
    md.append("5. Consult with a financial advisor for personalized guidance")
    md.append("")
    
    # Footer
    md.append("---")
    md.append(f"*Report generated by AI Wealth Strategist on {data['timestamp']}*")
    md.append("*This analysis is for informational purposes only and should not be considered as financial advice.*")
    
    return "\n".join(md)





def display_agent_signals(agent_signals: dict):
    """Display all agent signals in a formatted way"""
    print(f"\n{'='*60}")
    print(f"🤖 COMPREHENSIVE AGENT ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    if not agent_signals:
        print("⚠️  No agent signals found in state")
        return
    
    # Group agents by signal type
    signal_groups = {}
    for agent_name, signal in agent_signals.items():
        signal_type = signal.signal
        if signal_type not in signal_groups:
            signal_groups[signal_type] = []
        signal_groups[signal_type].append((agent_name, signal))
    
    # Display by signal groups
    for signal_type, agents in signal_groups.items():
        print(f"\n📊 {signal_type.upper()} SIGNALS ({len(agents)} agents):")
        print(f"{'─' * 50}")
        
        for agent_name, signal in agents:
            # Format agent name for display
            display_name = agent_name.replace('_', ' ').title()
            
            # Confidence indicator
            if signal.confidence >= 80:
                confidence_icon = "🟢"
            elif signal.confidence >= 60:
                confidence_icon = "🟡"
            else:
                confidence_icon = "🔴"
            
            print(f"\n{confidence_icon} {display_name}")
            print(f"   📈 Signal: {signal.signal}")
            print(f"   🎯 Confidence: {signal.confidence:.1f}%")
            print(f"   💭 Reasoning: {signal.reasoning[:100]}{'...' if len(signal.reasoning) > 100 else ''}")
            
            if signal.recommendations:
                print(f"   💡 Recommendations:")
                for i, rec in enumerate(signal.recommendations[:3], 1):  # Show first 3
                    print(f"      {i}. {rec}")
                if len(signal.recommendations) > 3:
                    print(f"      ... and {len(signal.recommendations) - 3} more")
            
            if signal.risk_factors:
                print(f"   ⚠️  Risk Factors:")
                for i, risk in enumerate(signal.risk_factors[:3], 1):  # Show first 3
                    print(f"      {i}. {risk}")
                if len(signal.risk_factors) > 3:
                    print(f"      ... and {len(signal.risk_factors) - 3} more")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print(f"📊 ANALYSIS SUMMARY:")
    print(f"{'='*60}")
    
    total_agents = len(agent_signals)
    avg_confidence = sum(s.confidence for s in agent_signals.values()) / total_agents if total_agents > 0 else 0
    
    print(f"🤖 Total Agents Analyzed: {total_agents}")
    print(f"📈 Average Confidence: {avg_confidence:.1f}%")
    print(f"🎯 Signal Distribution:")
    
    for signal_type, agents in signal_groups.items():
        percentage = (len(agents) / total_agents) * 100
        print(f"   • {signal_type.title()}: {len(agents)} agents ({percentage:.1f}%)")
    
    print(f"\n💡 All agent signals are now available to the Portfolio Manager for final analysis!")
    print(f"{'='*60}")


def parse_wealth_management_response(response):
    """Parses a JSON string and returns a dictionary."""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        return None
    except TypeError as e:
        print(f"Invalid response type (expected string, got {type(response).__name__}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
        return None


def run_wealth_management(
    client_profile: ClientProfile,
    portfolio: Portfolio,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "llama3.1:8b",
    model_provider: str = "Ollama",
):
    """Run the wealth management analysis with selected agents."""
    print(f"🤖 AI AGENT ANALYSIS IN PROGRESS...")
    print(f"   📊 AI Model: {model_name} ({model_provider})")
    print(f"   👤 Analyzing: {client_profile.name} (Age: {client_profile.age})")
    print(f"   💼 Portfolio: {len(portfolio.accounts)} accounts, {sum(len(account.holdings) for account in portfolio.accounts)} holdings")
    print(f"   🤖 Agents: {len(selected_analysts)} specialized AI agents")
    print(f"   🔍 Detailed Reasoning: {'Enabled' if show_reasoning else 'Disabled'}")
    print()
    print("🔄 Starting comprehensive analysis...")
    print()

    # Start progress tracking
    progress.start()

    try:
        # Create workflow with all agents
        print(f"🔧 Creating comprehensive AI workflow...")
        workflow = create_workflow(selected_analysts)
        agent = workflow.compile()
        print(f"✅ AI workflow compiled successfully!")
        print(f"🔄 Starting multi-agent analysis...")
        print()

        final_state = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Generate comprehensive wealth management recommendations based on the provided client and portfolio data.",
                    )
                ],
                "data": {
                    "client_profile": client_profile,
                    "portfolio": portfolio,
                    "agent_signals": {},
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name,
                    "model_provider": model_provider,
                },
            },
        )

        print(f"✅ Analysis completed successfully!")
        
        # Display all agent signals
        display_agent_signals(final_state["data"]["agent_signals"])
        
        # Parse final recommendations
        final_recommendations = parse_wealth_management_response(final_state["messages"][-1].content)
        
        # Save analysis to files
        saved_files = save_analysis_to_files(
            client_profile=client_profile,
            portfolio=portfolio,
            agent_signals=final_state["data"]["agent_signals"],
            final_recommendations=final_recommendations,
            market_data=market_data if 'market_data' in locals() else None
        )
        
        return {
            "recommendations": final_recommendations,
            "agent_signals": final_state["data"]["agent_signals"],
            "saved_files": saved_files
        }
    finally:
        # Stop progress tracking
        progress.stop()





def create_workflow(selected_analysts=None):
    """Create a custom workflow with selected analysts."""
    print(f"🔧 Creating custom workflow...")
    workflow = StateGraph(WealthAgentState)
    
    # Import start function from state module
    from graph.state import start
    
    # Add start node
    workflow.add_node("start", start)
    
    # Get all available analyst nodes
    analyst_nodes = get_analyst_nodes()
    
    # Add selected analyst nodes to the workflow
    if selected_analysts:
        for analyst_name in selected_analysts:
            if analyst_name in analyst_nodes:
                print(f"   ✅ Adding {analyst_name} to workflow")
                workflow.add_node(analyst_name, analyst_nodes[analyst_name])
            else:
                print(f"   ⚠️  Warning: {analyst_name} not found in available analysts")
    else:
        # Add all analyst nodes
        for analyst_name, analyst_func in analyst_nodes.items():
            print(f"   ✅ Adding {analyst_name} to workflow")
            workflow.add_node(analyst_name, analyst_func)
    
    # Set up the workflow edges
    if selected_analysts:
        # Connect selected analysts in sequence
        for i, analyst_name in enumerate(selected_analysts):
            if analyst_name in analyst_nodes:
                if i == 0:
                    workflow.add_edge("start", analyst_name)
                else:
                    workflow.add_edge(selected_analysts[i-1], analyst_name)
        
        # Connect the last analyst to END
        if selected_analysts:
            workflow.add_edge(selected_analysts[-1], END)
    else:
        # Connect all analysts in the default order using the key names
        analyst_keys = [key for _, key in ANALYST_ORDER]
        
        for i, analyst_key in enumerate(analyst_keys):
            if analyst_key in analyst_nodes:
                if i == 0:
                    workflow.add_edge("start", analyst_key)
                else:
                    workflow.add_edge(analyst_keys[i-1], analyst_key)
    
        # Connect the last analyst to END
        if analyst_keys:
            workflow.add_edge(analyst_keys[-1], END)
    
    # Set the entry point
    workflow.set_entry_point("start")
    
    print(f"✅ Custom workflow created with {len(selected_analysts) if selected_analysts else len(analyst_nodes)} agents")
    return workflow





def create_sample_client_profile() -> ClientProfile:
    """Create a sample client profile for testing."""
    from data.models import RiskTolerance, AccountType
    
    return ClientProfile(
        client_id="SAMPLE_001",
        name="John Smith",
        age=45,
        risk_tolerance=RiskTolerance.MODERATE,
        time_horizon=20,
        income=120000,
        tax_bracket=0.25,
        province="ON",
        marital_status="married",
        dependents=2,
        retirement_age=65,
        retirement_income_target=80000,
        emergency_fund_target=36000,
        mortgage_balance=400000,
        other_debt=15000,
        life_insurance_coverage=500000,
        disability_insurance=True,
        estate_value=1500000,
        has_will=True,
        has_power_of_attorney=True
    )


def create_sample_portfolio() -> Portfolio:
    """Create a sample portfolio for testing."""
    from data.models import Account, PortfolioHolding, AccountType, AssetClass
    
    # Create holdings
    holdings = [
        PortfolioHolding(
            symbol="AAPL",
            name="Apple Inc.",
            quantity=100,
            market_value=21376.0,
            cost_basis=15000.0,
            account_type=AccountType.RRSP,
            asset_class=AssetClass.US_EQUITY,
            sector="Technology",
            country="US"
        ),
        PortfolioHolding(
            symbol="MSFT",
            name="Microsoft Corporation",
            quantity=50,
            market_value=10688.0,
            cost_basis=10000.0,
            account_type=AccountType.RRSP,
            asset_class=AssetClass.US_EQUITY,
            sector="Technology",
            country="US"
        ),
        PortfolioHolding(
            symbol="GOOGL",
            name="Alphabet Inc.",
            quantity=25,
            market_value=62500.0,
            cost_basis=62500.0,
            account_type=AccountType.TFSA,
            asset_class=AssetClass.US_EQUITY,
            sector="Technology",
            country="US"
        )
    ]
    
    # Create accounts
    accounts = [
        Account(
            account_type=AccountType.RRSP,
            account_number="RRSP001",
            balance=32064.0,
            holdings=holdings[:2]  # AAPL and MSFT
        ),
        Account(
            account_type=AccountType.TFSA,
            account_number="TFSA001",
            balance=62500.0,
            holdings=holdings[2:]  # GOOGL
        )
    ]
    
    return Portfolio(
        client_id="SAMPLE_001",
        total_value=94564.0,
        accounts=accounts
    )


def test_market_data_integration():
    """Test the market data integration and show live details."""
    print(f"\n📊 TESTING MARKET DATA INTEGRATION")
    print("=" * 50)
    
    try:
        # Initialize market data service
        print(f"🔧 Initializing MarketDataService...")
        market_service = MarketDataService()
        print(f"✅ MarketDataService initialized with {len(market_service.agents)} agents")
        
        # Test individual agents
        print(f"\n🧪 TESTING INDIVIDUAL AGENTS:")
        print("-" * 30)
        
        test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        
        for agent_name, agent in market_service.agents.items():
            print(f"\n🔍 Testing {agent_name}...")
            try:
                if agent_name in ["yfinance", "polygon", "alpha_vantage", "marketstack", "twelve_data"]:
                    # Test with stock data
                    result = agent.get_stock_data("AAPL")
                    if "error" not in result:
                        price = result.get("current_price", "N/A")
                        print(f"   ✅ {agent_name}: AAPL price = ${price}")
                    else:
                        print(f"   ❌ {agent_name}: {result['error']}")
                elif agent_name in ["newsapi_us", "finnhub"]:
                    # Test with news data
                    result = agent.get_latest_news()
                    if "error" not in result:
                        articles = len(result.get("articles", []))
                        print(f"   ✅ {agent_name}: {articles} articles retrieved")
                    else:
                        print(f"   ❌ {agent_name}: {result['error']}")
                elif agent_name == "fred":
                    # Test with economic data
                    result = agent.get_economic_indicators()
                    if "error" not in result:
                        indicators = len(result.get("indicators", {}))
                        print(f"   ✅ {agent_name}: {indicators} indicators retrieved")
                    else:
                        print(f"   ❌ {agent_name}: {result['error']}")
            except Exception as e:
                print(f"   ❌ {agent_name}: Error - {str(e)}")
        
        # Test comprehensive data
        print(f"\n🔍 TESTING COMPREHENSIVE MARKET DATA:")
        print("-" * 40)
        
        print(f"🔄 Fetching comprehensive market data...")
        comprehensive_data = market_service.get_comprehensive_market_data(["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"])
        
        if "error" not in comprehensive_data:
            summary = comprehensive_data.get("summary", {})
            print(f"✅ Comprehensive data retrieved successfully!")
            print(f"   📊 Available sources: {summary.get('available_sources', [])}")
            print(f"   ❌ Error sources: {summary.get('error_sources', [])}")
            print(f"   📈 Total data points: {summary.get('data_points', 0)}")
            
            # Show detailed breakdown
            print(f"\n📋 DETAILED BREAKDOWN:")
            for source_name, source_data in comprehensive_data.items():
                if source_name == "summary":
                    continue
                if "error" in source_data:
                    print(f"   ❌ {source_name}: {source_data['error']}")
                else:
                    if "portfolio" in source_data:
                        portfolio = source_data["portfolio"]
                        print(f"   ✅ {source_name}: {len(portfolio)} symbols")
                    elif "articles" in source_data:
                        articles = source_data["articles"]
                        print(f"   ✅ {source_name}: {len(articles)} articles")
                    elif "indicators" in source_data:
                        indicators = source_data["indicators"]
                        print(f"   ✅ {source_name}: {len(indicators)} indicators")
                    else:
                        print(f"   ✅ {source_name}: Data available")
        else:
            print(f"❌ Comprehensive data error: {comprehensive_data['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Market data integration error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point for the wealth management system."""
    parser = argparse.ArgumentParser(description="AI Wealth Strategist - Comprehensive Wealth Management")
    parser.add_argument("--sample", action="store_true", help="Use sample data for testing")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama for local LLM inference")
    parser.add_argument("--show-reasoning", action="store_true", help="Show agent reasoning")
    parser.add_argument("--test-market-data", action="store_true", help="Test market data integration only")
    parser.add_argument("--agents", nargs="+", help="Specify which agents to use")
    
    args = parser.parse_args()

    print("=" * 80)
    print("🤖 AI WEALTH STRATEGIST - COMPREHENSIVE WEALTH MANAGEMENT")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test market data integration if requested
    if args.test_market_data:
        test_market_data_integration()
        return
    
    # Step 1: Load client and portfolio data
    print("📊 LOADING CLIENT & PORTFOLIO DATA...")
    if args.sample:
        print("📊 Using sample data for testing...")
        client_profile = create_sample_client_profile()
        portfolio = create_sample_portfolio()
    else:
        # TODO: Implement interactive data input
        print("📊 Using sample data (interactive input not implemented yet)...")
        client_profile = create_sample_client_profile()
        portfolio = create_sample_portfolio()
    
    # Display comprehensive client and portfolio information
    display_client_and_portfolio_info(client_profile, portfolio)
    
    # Step 2: Extract symbols and fetch comprehensive market data
    symbols = []
    for account in portfolio.accounts:
        for holding in account.holdings:
            symbols.append(holding.symbol)
    
    # Display comprehensive market data
    display_comprehensive_market_data(symbols)
    
    # Step 3: Use all agents by default (no selection needed)
    selected_analysts = [key for _, key in ANALYST_ORDER]
    print(f"\n🤖 USING ALL {len(selected_analysts)} WEALTH MANAGEMENT AGENTS")
    print("   This comprehensive analysis will provide complete financial planning insights.")
    print()

    # Step 4: Configure LLM model
    print("🦙 CONFIGURING AI MODEL...")
    if args.ollama:
        print("🦙 Using Ollama for local LLM inference.")
        
        # Show model selection options directly
        model_choice = questionary.select(
            "Select your Ollama model:",
            choices=[questionary.Choice(display, value=value) for display, value, _ in OLLAMA_LLM_ORDER],
            style=questionary.Style(
                [
                    ("selected", "fg:green bold"),
                    ("pointer", "fg:green bold"),
                    ("highlighted", "fg:green"),
                    ("answer", "fg:green bold"),
                ]
            ),
        ).ask()

        if not model_choice:
            print("\n\nInterrupt received. Exiting...")
            sys.exit(0)

        if model_choice == "-":
            model_name = questionary.text("Enter the custom model name:").ask()
            if not model_name:
                print("\n\nInterrupt received. Exiting...")
                sys.exit(0)
        else:
            model_name = model_choice
        
        model_provider = ModelProvider.OLLAMA.value
        
        # Ensure Ollama is installed, running, and the model is available
        print(f"🔧 Checking Ollama and model availability...")
        if not ensure_ollama_and_model(model_name):
            print(f"📥 Model {model_name} not found. Attempting to download...")
            print("⏳ This may take several minutes depending on model size...")
            
            # Try to download the model
            import subprocess
            try:
                result = subprocess.run(["ollama", "pull", model_name], 
                                      capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    print(f"✅ Model {model_name} downloaded successfully!")
                else:
                    print(f"❌ Failed to download model {model_name}: {result.stderr}")
                    print("❌ Cannot proceed without the selected model.")
                    sys.exit(1)
            except subprocess.TimeoutExpired:
                print("❌ Model download timed out. Please try again.")
                sys.exit(1)
            except Exception as e:
                print(f"❌ Error downloading model: {str(e)}")
                print("❌ Cannot proceed without the selected model.")
                sys.exit(1)
        
        print(f"✅ Using Ollama model: {model_name}")
    else:
        # Use default cloud model
        model_name = "gpt-4"
        model_provider = "OpenAI"
        print(f"✅ Using cloud model: {model_name} ({model_provider})")
    
    print()

    # Step 5: Run comprehensive wealth management analysis
    print("🚀 STARTING COMPREHENSIVE WEALTH MANAGEMENT ANALYSIS...")
    print("   This analysis will provide detailed insights from all specialized agents.")
    print("   Each agent will analyze your portfolio using real-time market data.")
    print()
    
    start_time = datetime.now()
    
    result = run_wealth_management(
        client_profile=client_profile,
        portfolio=portfolio,
        show_reasoning=args.show_reasoning,
        selected_analysts=selected_analysts,
        model_name=model_name,
        model_provider=model_provider,
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Step 6: Display comprehensive final report
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE WEALTH MANAGEMENT REPORT")
    print("=" * 80)
    print(f"⏱️  Analysis completed in {duration:.2f} seconds")
    print(f"🤖 Agents analyzed: {len(selected_analysts)}")
    print(f"📈 Market data sources: 8")
    print(f"💰 Portfolio value: ${sum(holding.market_value for account in portfolio.accounts for holding in account.holdings):,.2f}")
    print()

    # Display enhanced results
    print_wealth_management_output(result)
    
    # Show saved files information
    if 'saved_files' in result:
        print(f"\n💾 ANALYSIS SAVED TO FILES:")
        print(f"   📄 Complete Data (JSON): {result['saved_files']['json_file']}")
        print(f"   📝 Detailed Report (MD): {result['saved_files']['markdown_file']}")
        print(f"   📁 Files location: analysis_outputs/")
    
    # Generate and save visualizations
    try:
        save_graph_as_png("wealth_management_workflow.png")
        print("\n📊 Workflow visualization saved as 'wealth_management_workflow.png'")
        
        # Additional visualizations could be added here
        print("📈 Portfolio analysis charts and graphs available in the saved files")
        
    except Exception as e:
        print(f"\n⚠️  Could not save workflow visualization: {e}")
    
    print("\n" + "=" * 80)
    print("✅ COMPREHENSIVE WEALTH MANAGEMENT ANALYSIS COMPLETE")
    print("=" * 80)
    print("📋 Next Steps:")
    print("   • Review the detailed recommendations above")
    print("   • Check the saved analysis files for detailed reports")
    print("   • Consider implementing the suggested portfolio changes")
    print("   • Schedule a follow-up review in 3-6 months")
    print("   • Contact your financial advisor for personalized guidance")
    print("=" * 80)


# Import the app after all functions are defined
from graph.state import app

if __name__ == "__main__":
    main() 