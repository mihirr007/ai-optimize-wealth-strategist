import os
import json
from typing import Any, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from llm.models import get_model, get_model_info, ModelProvider
from utils.progress import progress
from graph.state import WealthAgentState


def call_llm(
    prompt: any,
    pydantic_model: type[BaseModel],
    agent_name: str | None = None,
    state: WealthAgentState | None = None,
    max_retries: int = 3,
    default_factory=None,
) -> BaseModel:
    """
    Makes an LLM call with retry logic, handling both JSON supported and non-JSON supported models.

    Args:
        prompt: The prompt to send to the LLM
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates and model config extraction
        state: Optional state object to extract agent-specific model configuration
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure

    Returns:
        An instance of the specified Pydantic model
    """
    
    # Extract model configuration if state is provided and agent_name is available
    if state and agent_name:
        model_name, model_provider = get_agent_model_config(state, agent_name)
    else:
        # Use system defaults when no state or agent_name is provided
        model_name = "gpt-4o"
        model_provider = "OpenAI"

    model_info = get_model_info(model_name, model_provider)
    llm = get_model(model_name, model_provider)

    # For non-JSON support models, we can use structured output
    if not (model_info and not model_info.has_json_mode()):
        llm = llm.with_structured_output(
            pydantic_model,
            method="json_mode",
        )

    # Call the LLM with retries
    for attempt in range(max_retries):
        try:
            # Call the LLM
            result = llm.invoke(prompt)

            # For non-JSON support models, we need to extract and parse the JSON manually
            if model_info and not model_info.has_json_mode():
                parsed_result = extract_json_from_response(result.content)
                if parsed_result:
                    return pydantic_model(**parsed_result)
            else:
                return result

        except Exception as e:
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")

            if attempt == max_retries - 1:
                print(f"Error in LLM call after {max_retries} attempts: {e}")
                # Use default_factory if provided, otherwise create a basic default
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # This should never be reached due to the retry logic above
    return create_default_response(pydantic_model)


def create_default_response(model_class: type[BaseModel]) -> BaseModel:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None

    return model_class(**default_values)


def extract_json_from_response(content: str) -> dict | None:
    """Extracts JSON from markdown-formatted response."""
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7 :]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
    except Exception as e:
        print(f"Error extracting JSON from response: {e}")
    return None


def get_agent_model_config(state, agent_name):
    """
    Get model configuration for a specific agent from the state.
    Falls back to global model configuration if agent-specific config is not available.
    Always returns valid model_name and model_provider values.
    """
    # Fall back to global configuration (system defaults)
    model_name = state.get("metadata", {}).get("model_name") or "gpt-4o"
    model_provider = state.get("metadata", {}).get("model_provider") or "OpenAI"
    
    # Convert enum to string if necessary
    if hasattr(model_provider, 'value'):
        model_provider = model_provider.value
    
    return model_name, model_provider


def call_llm_with_model(prompt: ChatPromptTemplate, model_name: str, model_provider: str, **kwargs) -> str:
    """
    Call specific LLM model with prompt and return response.
    
    Args:
        prompt: LangChain prompt template
        model_name: Name of the model to use
        model_provider: Provider of the model
        **kwargs: Variables to format the prompt
        
    Returns:
        str: LLM response
    """
    # Get LLM instance
    provider_enum = ModelProvider(model_provider)
    llm = get_model(model_name, provider_enum)
    
    if llm is None:
        raise ValueError(f"Could not initialize LLM: {model_name} from {model_provider}")
    
    # Format prompt with variables
    formatted_prompt = prompt.format(**kwargs)
    
    # Call LLM
    response = llm.invoke(formatted_prompt)
    
    return response.content


def parse_json_response(response: str) -> dict:
    """
    Parse JSON response from LLM, handling common formatting issues.
    
    Args:
        response: Raw LLM response string
        
    Returns:
        dict: Parsed JSON object
    """
    # Try to extract JSON from response
    try:
        # Look for JSON block in markdown
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            json_str = response[start:end].strip()
        elif "```" in response:
            # Look for any code block
            start = response.find("```") + 3
            end = response.find("```", start)
            json_str = response[start:end].strip()
        else:
            # Try to parse the entire response as JSON
            json_str = response.strip()
        
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # If JSON parsing fails, try to extract JSON-like structure
        try:
            # Look for dictionary-like structure
            import re
            pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(pattern, response)
            if matches:
                return json.loads(matches[0])
        except:
            pass
        
        raise ValueError(f"Could not parse JSON response: {e}\nResponse: {response}")


def validate_llm_response(response: str, expected_fields: list) -> bool:
    """
    Validate that LLM response contains expected fields.
    
    Args:
        response: LLM response string
        expected_fields: List of expected field names
        
    Returns:
        bool: True if all expected fields are present
    """
    try:
        parsed = parse_json_response(response)
        return all(field in parsed for field in expected_fields)
    except:
        return False 