from pydantic import BaseModel, Field
from enum import Enum


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str


class ModelOptions(BaseModel):
    temperature: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="Controls randomness in generation (higher = more random)",
    )

    num_ctx: int = Field(
        default=256, gt=0, description="Context length size (number of tokens)"
    )

    stop: list[str] = Field(
        default_factory=list, description="Stop sequences that will halt generation"
    )


class ChatRequest(BaseModel):
    model: str = Field(..., description="Model used for the chat")
    messages: list[ChatMessage] = Field(
        ...,
        description="Chat history as an array of message objects (each with a role and content)",
    )
    options: ModelOptions = Field(
        ModelOptions(), description="Runtime options that control text generation"
    )
    stream: bool = Field(default=False, description="Whether or not to stream")


class ChatResponse(BaseModel):
    id: str = Field(default="chatcmpl-00001")  # Fake OpenAI-style ID
    model: str
    content: str
