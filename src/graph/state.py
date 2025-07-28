from typing_extensions import Annotated, Sequence, TypedDict
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
import json


def merge_dicts(a: dict[str, any], b: dict[str, any]) -> dict[str, any]:
    """Merge dictionaries, properly accumulating agent_signals"""
    result = {**a}
    
    # Special handling for agent_signals to accumulate them
    if "agent_signals" in a and "agent_signals" in b:
        # Merge agent_signals dictionaries
        result["agent_signals"] = {**a["agent_signals"], **b["agent_signals"]}
    elif "agent_signals" in b:
        # If only b has agent_signals, use it
        result["agent_signals"] = b["agent_signals"]
    
    # Merge all other keys normally
    for key, value in b.items():
        if key != "agent_signals":  # Already handled above
            result[key] = value
    
    return result


# Define agent state for wealth management
class WealthAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[dict[str, any], merge_dicts]
    metadata: Annotated[dict[str, any], merge_dicts]


def start(state: WealthAgentState):
    """Initialize the workflow with the input message."""
    print(f"🎯 START NODE: Initializing wealth management workflow...")
    return state


def show_agent_reasoning(output, agent_name):
    """Display agent reasoning in a formatted way"""
    print(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}")

    def convert_to_serializable(obj):
        if hasattr(obj, "to_dict"):  # Handle Pandas Series/DataFrame
            return obj.to_dict()
        elif hasattr(obj, "__dict__"):  # Handle custom objects
            return obj.__dict__
        elif isinstance(obj, (int, float, bool, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return str(obj)  # Fallback to string representation

    if isinstance(output, (dict, list)):
        # Convert the output to JSON-serializable format
        serializable_output = convert_to_serializable(output)
        print(json.dumps(serializable_output, indent=2))
    else:
        try:
            # Parse the string as JSON and pretty print it
            parsed_output = json.loads(output)
            print(json.dumps(parsed_output, indent=2))
        except json.JSONDecodeError:
            # Fallback to original string if not valid JSON
            print(output)

    print("=" * 48)


def show_current_agent_signals(state):
    """Display current state of agent signals"""
    if "data" in state and "agent_signals" in state["data"]:
        agent_signals = state["data"]["agent_signals"]
        if agent_signals:
            print(f"\n📊 CURRENT AGENT SIGNALS ({len(agent_signals)} agents):")
            print(f"{'─' * 50}")
            
            # Group by signal type
            signal_groups = {}
            for agent_name, signal in agent_signals.items():
                signal_type = signal.signal if hasattr(signal, 'signal') else 'unknown'
                if signal_type not in signal_groups:
                    signal_groups[signal_type] = []
                signal_groups[signal_type].append(agent_name)
            
            for signal_type, agents in signal_groups.items():
                print(f"   📈 {signal_type.title()}: {len(agents)} agents")
                for agent_name in agents[:3]:  # Show first 3
                    print(f"      • {agent_name}")
                if len(agents) > 3:
                    print(f"      ... and {len(agents) - 3} more")
            
            print(f"{'─' * 50}")


# Create the default app workflow
app = StateGraph(WealthAgentState)

# Add start node
app.add_node("start", start)

# Import and add all analyst nodes
from utils.analysts import get_analyst_nodes, ANALYST_ORDER

analyst_nodes = get_analyst_nodes()

# Add all analyst nodes to the default workflow
for analyst_name, analyst_func in analyst_nodes.items():
    app.add_node(analyst_name, analyst_func)

# Connect all analysts in the default order using the key names (second element of tuples)
analyst_keys = [key for _, key in ANALYST_ORDER]

for i, analyst_key in enumerate(analyst_keys):
    if analyst_key in analyst_nodes:
        if i == 0:
            app.add_edge("start", analyst_key)
        else:
            app.add_edge(analyst_keys[i-1], analyst_key)

# Connect the last analyst to END
if analyst_keys:
    app.add_edge(analyst_keys[-1], END) 