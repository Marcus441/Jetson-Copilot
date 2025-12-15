from llama_cpp import ChatCompletionRequestMessage
from pydantic import BaseModel, Field
from ..llm.schemas import ModelOptions


class ChatRequest(BaseModel):
    model: str = Field(..., description="Model used for the chat")
    messages: list[ChatCompletionRequestMessage] = Field(
        ...,
        description="Chat history as an array of message objects (each with a role and content)",
    )
    options: ModelOptions = Field(
        ModelOptions(), description="Runtime options that control text generation"
    )
    stream: bool = Field(default=False, description="Whether or not to stream")
    keep_alive: str = Field(
        default="5m", description="Keep alive duration for the model"
    )


class StreamedChatResponse(BaseModel):
    id: str = Field(default="chatcmpl-00001")  # Fake OpenAI-style ID
    model: str
    content: str


class ChatResponse(BaseModel):
    id: str = Field(default="chatcmpl-00001")  # Fake OpenAI-style ID
    model: str
    content: str
