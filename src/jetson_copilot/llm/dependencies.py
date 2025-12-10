from typing import cast

from fastapi import FastAPI, Request

from ..llm.llm_engine import LLMEngine
from ..schemas import AppState


def get_engine(request: Request) -> LLMEngine:
    """Retrieves the pre-loaded LLMEngine instance from the application state."""

    app = cast(FastAPI, request.app)
    state = cast(AppState, cast(object, app.state))
    engine = getattr(state, "engine", None)

    if engine is None:
        raise RuntimeError("LLMEngine is not initialized yet.")

    return cast(LLMEngine, engine)
