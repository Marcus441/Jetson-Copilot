from enum import Enum
from pydantic import BaseModel, Field


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
        default=4096, gt=0, description="Context length size (number of tokens)"
    )
    num_predict: int = Field(
        default=64, gt=0, description="Maximum number of tokens to generate"
    )

    keep_alive: str = Field(
        default="5m", description="Keep alive duration for the stream"
    )

    stop: list[str] = Field(
        default_factory=list, description="Stop sequences that will halt generation"
    )
