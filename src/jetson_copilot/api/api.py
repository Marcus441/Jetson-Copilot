from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, status
import logging

from llm.llm_engine import LLMEngine
from data_models.models import ChatRequest
import uvicorn

logging.basicConfig(level=logging.INFO)

engine = LLMEngine()


def get_engine() -> LLMEngine:
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):  # pyright: ignore[reportUnusedParameter]
    if engine.ensure_model_exists():
        try:
            engine.load_model()
        except Exception as e:
            logging.critical(f"Fatal Error: Failed to initialize LLama engine: {e}")
            raise e
    yield
    engine.unload_model()
    logging.info("FastAPI server shutdown complete.")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def check_health(engine: Annotated[LLMEngine, Depends(get_engine)]):
    if not engine.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM model is not loaded or initialized",
        )
    return {"status": "active", "model_status": "loaded"}


@app.post("/api/chat", response_model=ChatRequest)
def get_chat():
    pass
    return "Hello There"


def start():
    uvicorn.run("jetson_copilot.api:app", host="0.0.0.0", port=11434, reload=False)
