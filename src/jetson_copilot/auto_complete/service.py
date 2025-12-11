from collections.abc import Iterator
from typing import cast
from llama_cpp import (
    ChatCompletionRequestMessage,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)
from jetson_copilot.llm.llm_engine import LLMEngine
from jetson_copilot.llm.schemas import ModelOptions


def stream_completion_safe(
    iterator: Iterator[CreateChatCompletionStreamResponse],
) -> Iterator[bytes]:
    for chunk in iterator:
        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
        if content:
            yield content.encode("utf-8")


def create_completion(
    messages: list[ChatCompletionRequestMessage],
    options: ModelOptions,
    model: str,
    engine: LLMEngine,
) -> CreateChatCompletionResponse:
    """Non-streaming chat completion"""
    return cast(
        CreateChatCompletionResponse,
        engine.model.create_chat_completion(
            messages,
            temperature=options.temperature,
            stream=False,  # always non-streaming
            model=model,
        ),
    )


def create_streamed_completion(
    messages: list[ChatCompletionRequestMessage],
    options: ModelOptions,
    model: str,
    engine: LLMEngine,
) -> Iterator[CreateChatCompletionStreamResponse]:
    """Streaming chat completion"""
    return cast(
        Iterator[CreateChatCompletionStreamResponse],
        engine.model.create_chat_completion(
            messages,
            temperature=options.temperature,
            stream=True,
            model=model,
        ),
    )
