from huggingface_hub import hf_hub_download  # pyright: ignore[reportUnknownVariableType]
from llama_cpp import Llama
import logging

from ..config import FILENAME, MODEL_DIR, MODEL_PATH, REPO_ID

logging.basicConfig(level=logging.INFO)


class LLMEngine:
    model: Llama
    loaded: bool

    def __init__(self) -> None:
        if self.ensure_model_exists():
            try:
                self.model = Llama(
                    model_path=str(MODEL_PATH),
                    verbose=True,
                    n_ctx=4096,
                    n_gpu_layers=-1,
                )
                self.loaded = True
            except Exception as e:
                logging.critical(f"Fatal Error: Failed to initialize LLama engine: {e}")
                self.loaded = False
                raise e

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
