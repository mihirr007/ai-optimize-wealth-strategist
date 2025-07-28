import os
import json
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from enum import Enum
from pydantic import BaseModel
from typing import Tuple, List
from pathlib import Path


class ModelProvider(str, Enum):
    """Enum for supported LLM providers"""

    ALIBABA = "Alibaba"
    ANTHROPIC = "Anthropic"
    DEEPSEEK = "DeepSeek"
    GOOGLE = "Google"
    GROQ = "Groq"
    META = "Meta"
    MISTRAL = "Mistral"
    OPENAI = "OpenAI"
    OLLAMA = "Ollama"


class LLMModel(BaseModel):
    """Represents an LLM model configuration"""

    display_name: str
    model_name: str
    provider: ModelProvider

    def to_choice_tuple(self) -> Tuple[str, str, str]:
        """Convert to format needed for questionary choices"""
        return (self.display_name, self.model_name, self.provider.value)

    def is_custom(self) -> bool:
        """Check if the model is a custom model"""
        return self.model_name == "-"

    def has_json_mode(self) -> bool:
        """Check if the model supports JSON mode"""
        if self.is_deepseek() or self.is_gemini():
            return False
        # Only certain Ollama models support JSON mode
        if self.is_ollama():
            return "llama3" in self.model_name or "neural-chat" in self.model_name
        return True

    def is_deepseek(self) -> bool:
        """Check if the model is a DeepSeek model"""
        return self.model_name.startswith("deepseek")

    def is_gemini(self) -> bool:
        """Check if the model is a Gemini model"""
        return self.model_name.startswith("gemini")

    def is_ollama(self) -> bool:
        """Check if the model is an Ollama model"""
        return self.provider == ModelProvider.OLLAMA


# Load models from JSON file
def load_models_from_json(json_path: str) -> List[LLMModel]:
    """Load models from a JSON file"""
    with open(json_path, 'r') as f:
        models_data = json.load(f)
    
    models = []
    for model_data in models_data:
        # Convert string provider to ModelProvider enum
        provider_enum = ModelProvider(model_data["provider"])
        models.append(
            LLMModel(
                display_name=model_data["display_name"],
                model_name=model_data["model_name"],
                provider=provider_enum
            )
        )
    return models


# Get the path to the JSON files
current_dir = Path(__file__).parent
models_json_path = current_dir / "api_models.json"
ollama_models_json_path = current_dir / "ollama_models.json"

# Load available models from JSON
AVAILABLE_MODELS = load_models_from_json(str(models_json_path))

# Load Ollama models from JSON
OLLAMA_MODELS = load_models_from_json(str(ollama_models_json_path))

# Create LLM_ORDER in the format expected by the UI
LLM_ORDER = [model.to_choice_tuple() for model in AVAILABLE_MODELS]

# Create Ollama LLM_ORDER separately
OLLAMA_LLM_ORDER = [model.to_choice_tuple() for model in OLLAMA_MODELS]


def get_model_info(model_name: str, model_provider: str) -> LLMModel | None:
    """Get model info by name and provider"""
    for model in AVAILABLE_MODELS + OLLAMA_MODELS:
        if model.model_name == model_name and model.provider.value == model_provider:
            return model
    return None


def get_models_list():
    """Get list of all available models"""
    return [model.display_name for model in AVAILABLE_MODELS + OLLAMA_MODELS]


def get_model(model_name: str, model_provider: ModelProvider) -> ChatOpenAI | ChatGroq | ChatOllama | None:
    """Get LLM model instance based on provider"""
    if model_provider == ModelProvider.OPENAI:
        return ChatOpenAI(
            model=model_name,
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif model_provider == ModelProvider.GROQ:
        return ChatGroq(
            model=model_name,
            temperature=0.1,
            api_key=os.getenv("GROQ_API_KEY")
        )
    elif model_provider == ModelProvider.ANTHROPIC:
        return ChatAnthropic(
            model=model_name,
            temperature=0.1,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    elif model_provider == ModelProvider.DEEPSEEK:
        return ChatDeepSeek(
            model=model_name,
            temperature=0.1,
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )
    elif model_provider == ModelProvider.GOOGLE:
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.1,
            api_key=os.getenv("GOOGLE_API_KEY")
        )
    elif model_provider == ModelProvider.OLLAMA:
        return ChatOllama(
            model=model_name,
            temperature=0.1,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
    else:
        return None 