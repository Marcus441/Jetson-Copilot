import logging
from typing import Annotated
from fastapi import APIRouter, Depends
from fastapi.routing import request_response
from llama_cpp import CreateChatCompletionResponse

from jetson_copilot.llm.dependencies import get_engine
from jetson_copilot.llm.llm_engine import LLMEngine
from .schemas import ChatRequest, ChatResponse
from .service import CreateCompletion

logging.basicConfig(level=logging.INFO)

chat_router = APIRouter()


@chat_router.post("/chat", response_model=CreateChatCompletionResponse)
def get_chat(request: ChatRequest, engine: Annotated[LLMEngine, Depends(get_engine)]):
    response = CreateCompletion(
        request.keep_alive,
        request.messages,
        request.options,
        request.stream,
        request.model,
        engine,
    )
    return response
