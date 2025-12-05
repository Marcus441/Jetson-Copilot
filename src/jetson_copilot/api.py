from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
import logging

from .llm_handler import ensure_model_exists, load_model, model_loaded, unload_model
from .models import ChatRequest
import uvicorn

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):  # pyright: ignore[reportUnusedParameter]
    if ensure_model_exists():
        try:
            load_model()
        except Exception as e:
            logging.critical(f"Fatal Error: Failed to initialize LLama engine: {e}")
            raise e
    yield
    unload_model()
    logging.info("FastAPI server shutdown complete.")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def check_health():
    if not model_loaded:
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
