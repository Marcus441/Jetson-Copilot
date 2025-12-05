from huggingface_hub import (
    hf_hub_download,  # pyright: ignore[reportUnknownVariableType]
)
from llama_cpp import Llama
import logging

from ..config import FILENAME, MODEL_DIR, MODEL_PATH, REPO_ID

logging.basicConfig(level=logging.INFO)


class LLMEngine:
    model: Llama | None

    def __init__(self) -> None:
        self.model = None

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

    def load_model(self) -> None:
        try:
            self.model = Llama(model_path=str(MODEL_PATH), verbose=False)
        except Exception as e:
            raise e

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def unload_model(self) -> None:
        del self.model
        logging.info("LLM Engine successfully unloaded and memory freed.")
