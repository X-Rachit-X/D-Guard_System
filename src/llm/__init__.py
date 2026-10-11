"""src/llm - optional local LLM (Ollama) used by the report agent to write prose."""
from src.llm.client import OllamaClient

__all__ = ["OllamaClient"]
