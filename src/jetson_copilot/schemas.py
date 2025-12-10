from typing import Protocol
from jetson_copilot.llm.llm_engine import LLMEngine


class AppState(Protocol):
    engine: LLMEngine
