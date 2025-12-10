from pathlib import Path

MODEL_DIR: Path = Path(__file__).parent.parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

REPO_ID: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"
FILENAME: str = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"

MODEL_PATH: Path = MODEL_DIR / FILENAME
