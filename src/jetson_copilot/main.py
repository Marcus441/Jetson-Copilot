from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
import uvicorn
import logging


from .llm.dependencies import get_engine
from .llm.llm_engine import LLMEngine
from .auto_complete.router import chat_router

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.engine = LLMEngine()
    except Exception as e:
        logging.critical(f"Fatal Error: Failed to initialize LLama engine: {e}")
        raise e
    yield
    app.state.engine.unload_model()
    logging.info("FastAPI server shutdown complete.")


app = FastAPI(lifespan=lifespan)
router = APIRouter()


@router.get("/health")
def check_health(engine: Annotated[LLMEngine, Depends(get_engine)]):
    if not engine.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM model is not loaded or initialized",
        )
    return {"status": "active", "model_status": "loaded"}


app.include_router(router, prefix="/api")
app.include_router(chat_router, prefix="/api")


def start():
    uvicorn.run("jetson_copilot.main:app", host="0.0.0.0", port=11434, reload=False)
