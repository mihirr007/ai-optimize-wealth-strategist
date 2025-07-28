import matplotlib.pyplot as plt
import networkx as nx
from typing import Any


def save_graph_as_png(graph: Any, filename: str = "wealth_management_graph.png"):
    """Save the agent workflow graph as a PNG file"""
    try:
        # Create a NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes and edges based on the LangGraph structure
        # This is a simplified visualization - in practice you'd extract the actual graph structure
        
        # Add sample nodes for wealth management agents
        nodes = [
            "Start",
            "Passive Indexing",
            "Dividend Growth", 
            "ESG",
            "Factor Investing",
            "Global Macro",
            "Tactical Allocation",
            "Canadian Core",
            "Risk Profiler",
            "Tax Optimization",
            "Estate Planning",
            "Retirement Planner",
            "Insurance Planning",
            "Debt Strategy",
            "Portfolio Auditor",
            "Rebalancer",
            "Sentiment & Market Context",
            "Portfolio Manager",
            "End"
        ]
        
        # Add nodes
        for node in nodes:
            G.add_node(node)
        
        # Add edges (simplified workflow)
        edges = [
            ("Start", "Passive Indexing"),
            ("Start", "Dividend Growth"),
            ("Start", "ESG"),
            ("Start", "Factor Investing"),
            ("Start", "Global Macro"),
            ("Start", "Tactical Allocation"),
            ("Start", "Canadian Core"),
            ("Start", "Risk Profiler"),
            ("Start", "Tax Optimization"),
            ("Start", "Estate Planning"),
            ("Start", "Retirement Planner"),
            ("Start", "Insurance Planning"),
            ("Start", "Debt Strategy"),
            ("Start", "Portfolio Auditor"),
            ("Start", "Rebalancer"),
            ("Start", "Sentiment & Market Context"),
            ("Passive Indexing", "Portfolio Manager"),
            ("Dividend Growth", "Portfolio Manager"),
            ("ESG", "Portfolio Manager"),
            ("Factor Investing", "Portfolio Manager"),
            ("Global Macro", "Portfolio Manager"),
            ("Tactical Allocation", "Portfolio Manager"),
            ("Canadian Core", "Portfolio Manager"),
            ("Risk Profiler", "Portfolio Manager"),
            ("Tax Optimization", "Portfolio Manager"),
            ("Estate Planning", "Portfolio Manager"),
            ("Retirement Planner", "Portfolio Manager"),
            ("Insurance Planning", "Portfolio Manager"),
            ("Debt Strategy", "Portfolio Manager"),
            ("Portfolio Auditor", "Portfolio Manager"),
            ("Rebalancer", "Portfolio Manager"),
            ("Sentiment & Market Context", "Portfolio Manager"),
            ("Portfolio Manager", "End")
        ]
        
        # Add edges
        for edge in edges:
            G.add_edge(edge[0], edge[1])
        
        # Create the plot
        plt.figure(figsize=(16, 12))
        
        # Use hierarchical layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color='lightblue',
                              node_size=3000,
                              alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, 
                              edge_color='gray',
                              arrows=True,
                              arrowsize=20,
                              alpha=0.6)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, 
                               font_size=8,
                               font_weight='bold')
        
        plt.title("AI Wealth Strategist Agent Workflow", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graph saved as {filename}")
        
    except Exception as e:
        print(f"Error saving graph: {e}")
        # Create a simple text representation instead
        with open(filename.replace('.png', '.txt'), 'w') as f:
            f.write("AI Wealth Strategist Agent Workflow\n")
            f.write("=" * 40 + "\n\n")
            f.write("Start\n")
            f.write("├── Passive Indexing Agent\n")
            f.write("├── Dividend Growth Agent\n")
            f.write("├── ESG Agent\n")
            f.write("├── Factor Investing Agent\n")
            f.write("├── Global Macro Agent\n")
            f.write("├── Tactical Allocation Agent\n")
            f.write("├── Canadian Core Agent\n")
            f.write("├── Risk Profiler Agent\n")
            f.write("├── Tax Optimization Agent\n")
            f.write("├── Estate Planning Agent\n")
            f.write("├── Retirement Planner Agent\n")
            f.write("├── Insurance Planning Agent\n")
            f.write("├── Debt Strategy Agent\n")
            f.write("├── Portfolio Auditor Agent\n")
            f.write("├── Rebalancer Agent\n")
            f.write("└── Sentiment & Market Context Agent\n")
            f.write("    └── Portfolio Manager Agent\n")
            f.write("        └── End\n")
        
        print(f"Text representation saved as {filename.replace('.png', '.txt')}") 