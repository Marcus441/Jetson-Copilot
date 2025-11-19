# src/jetson_project/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .completion_service import CompletionService
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Jetson Copilot Autocompletion Engine",
    description="API for code completion using Qwen2.5-Coder model on Jetson Orin Nano.",
    version="0.1.0"
)

# Initialize the CompletionService
# For production on Jetson, ensure device is "cuda"
completion_service = CompletionService(device="cuda")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Jetson Copilot Autocompletion Engine...")
    try:
        completion_service.load_model()
        logger.info("Model loaded successfully during startup.")
    except Exception as e:
        logger.critical(f"Failed to load model during startup: {e}")
        # Depending on the desired behavior, you might want to exit here or keep running with degraded functionality
        # For a critical service, exiting might be preferred.
        raise RuntimeError("Model failed to load, cannot start service.") from e

class CompletionRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100 # Default value

class CompletionResponse(BaseModel):
    completion: str

@app.post("/complete", response_model=CompletionResponse)
async def complete_code(request: CompletionRequest):
    """
    Generates code completion for the given prompt.
    """
    logger.info(f"Received completion request for prompt (truncated): {request.prompt[:50]}...")
    try:
        completion = completion_service.get_completion(
            request.prompt,
            request.max_new_tokens
        )
        logger.info("Successfully generated completion.")
        return CompletionResponse(completion=completion)
    except Exception as e:
        logger.error(f"Error generating completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate completion: {e}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "ok", "model_loaded": completion_service.model_handler.model is not None}

