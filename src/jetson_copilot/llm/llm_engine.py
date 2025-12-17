from asyncio import (
    Queue,
    Semaphore,
    Task,
    create_task,
    to_thread,
)
from collections.abc import AsyncIterator
from typing import cast
from llama_cpp import (
    ChatCompletionRequestMessage,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)
from huggingface_hub import hf_hub_download  # pyright: ignore[reportUnknownVariableType]
from llama_cpp import Llama
import logging

from jetson_copilot.llm.schemas import ModelOptions


from ..config import FILENAME, MODEL_DIR, MODEL_PATH, REPO_ID

logging.basicConfig(level=logging.INFO)


class LLMEngine:
    model: Llama
    loaded: bool
    active_task: Task[None] | None
    _semaphore: Semaphore

    def __init__(self) -> None:
        self.active_task = None
        self._semaphore = Semaphore(1)
        if self.ensure_model_exists():
            try:
                self.model = Llama(
                    model_path=str(MODEL_PATH),
                    verbose=True,
                    # n_ctx=4096,
                    # n_gpu_layers=-1,
                )
                self.loaded = True
            except Exception as e:
                logging.critical(f"Fatal Error: Failed to initialize LLama engine: {e}")
                self.loaded = False

    async def stream_chat_completion(
        self,
        messages: list[ChatCompletionRequestMessage],
        options: ModelOptions,
        model: str,
    ) -> AsyncIterator[CreateChatCompletionStreamResponse]:
        async with self._semaphore:
            queue: Queue[CreateChatCompletionStreamResponse | None] = Queue()

            def worker():
                try:
                    for chunk in self.model.create_chat_completion(
                        messages,
                        temperature=options.temperature,
                        max_tokens=options.num_ctx,
                        stream=True,
                        model=model,
                    ):
                        queue.put_nowait(
                            cast(CreateChatCompletionStreamResponse, chunk)
                        )
                finally:
                    queue.put_nowait(None)

            task = create_task(to_thread(worker))
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item

    async def create_completion(
        self,
        messages: list[ChatCompletionRequestMessage],
        options: ModelOptions,
        model: str,
    ) -> CreateChatCompletionResponse:
        async with self._semaphore:
            return cast(
                CreateChatCompletionResponse,
                await to_thread(
                    self.model.create_chat_completion,
                    messages,
                    temperature=options.temperature,
                    max_tokens=options.num_ctx,
                    stream=False,
                    model=model,
                ),
            )

    @staticmethod
    def ensure_model_exists() -> bool:
        if MODEL_PATH.exists():
            print(f"Model found:  {MODEL_PATH}")
            return True
        print(f"Model not found. Downloading {FILENAME} from Hugging Face...")

        try:
            download = hf_hub_download(
                repo_id=REPO_ID, filename=FILENAME, local_dir=str(MODEL_DIR)
            )
            logging.info(f"Successfully downloaded model to: {download}")
            return True
        except Exception as e:
            logging.info(f"Error downloading model from hugging face: {e}")
            return False

    def unload_model(self) -> None:
        del self.model
        logging.info("LLM Engine successfully unloaded and memory freed.")
