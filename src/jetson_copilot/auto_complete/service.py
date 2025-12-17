from collections.abc import AsyncIterator
from llama_cpp import (
    ChatCompletionRequestMessage,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)
from jetson_copilot.llm.llm_engine import LLMEngine
from jetson_copilot.llm.schemas import ModelOptions


async def stream_completion_safe(
    iterator: AsyncIterator[CreateChatCompletionStreamResponse],
) -> AsyncIterator[bytes]:
    async for chunk in iterator:
        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
        if content:
            yield content.encode("utf-8")


async def create_completion(
    messages: list[ChatCompletionRequestMessage],
    options: ModelOptions,
    model: str,
    engine: LLMEngine,
) -> CreateChatCompletionResponse:
    """Non-streaming chat completion"""
    return await engine.create_completion(
        messages,
        options,
        model=model,
    )


async def create_streamed_completion(
    messages: list[ChatCompletionRequestMessage],
    options: ModelOptions,
    model: str,
    engine: LLMEngine,
) -> AsyncIterator[CreateChatCompletionStreamResponse]:
    """Streaming chat completion"""
    return engine.stream_chat_completion(
        messages,
        options,
        model=model,
    )
