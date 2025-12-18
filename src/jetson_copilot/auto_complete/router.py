import logging
from typing import Annotated
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from jetson_copilot.auto_complete.service import (
    create_completion,
    create_streamed_completion,
    stream_completion_safe,
)
from jetson_copilot.llm.dependencies import get_engine
from jetson_copilot.llm.llm_engine import LLMEngine
from .schemas import ChatRequest

logging.basicConfig(level=logging.INFO)

chat_router = APIRouter()


@chat_router.post("/chat")
async def get_chat(
    request: ChatRequest,
    engine: Annotated[LLMEngine, Depends(get_engine)],
):
    if request.stream:
        stream = await create_streamed_completion(
            request.messages,
            request.options,
            request.model,
            engine,
        )

        return StreamingResponse(
            stream_completion_safe(stream),
            media_type="text/event-stream",
        )
    else:
        return await create_completion(
            request.messages,
            request.options,
            request.model,
            engine,
        )
