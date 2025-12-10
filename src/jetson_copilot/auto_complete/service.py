from llama_cpp import ChatCompletionRequestMessage
from jetson_copilot.llm.llm_engine import LLMEngine
from jetson_copilot.llm.schemas import ModelOptions


def CreateCompletion(
    keep_alive: str,
    messages: list[ChatCompletionRequestMessage],
    options: ModelOptions,
    stream: bool,
    model: str,
    engine: LLMEngine,
):
    response = engine.model.create_chat_completion(
        messages,
        temperature=options.temperature,
        stream=stream,
        model=model,
    )
    return response
