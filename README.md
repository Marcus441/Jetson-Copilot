# Jetson Copilot Autocompletion Engine

This project provides a powerful code autocompletion engine powered by the
**Qwen2.5-Coder-1.5B-Instruct** model, specifically optimized for deployment on
the **Jetson Orin Nano** platform. It is designed to integrate seamlessly with
**avant.nvim**, offering an AI-driven coding experience directly within your
Neovim environment.

## Features

- **Qwen2.5-Coder Model**: Leverages a state-of-the-art large language model
  fine-tuned for code generation and completion.
- **Jetson Orin Nano Optimization**: Configured for efficient inference on
  NVIDIA Jetson Orin Nano, utilizing CUDA and `float16` precision.
- **FastAPI Server**: A robust and asynchronous web server built with FastAPI,
  providing a `/complete` API endpoint for code suggestions and a `/health`
  endpoint for status checks.
- **`avant.nvim` Compatibility**: Designed to integrate as a custom completion
  source for `avant.nvim`, emulating a Copilot-like experience.
- **Robust Error Handling**: Comprehensive logging and error management to
  ensure stable operation.

## Setup

Follow these steps to set up and run the autocompletion engine on your Jetson
Orin Nano.

### 1. Prerequisites

- **Jetson Orin Nano**: Ensure your Jetson Orin Nano is running the latest
  JetPack SDK (Ubuntu-based).
- **Python 3.10+**: Install Python 3.10 or newer.
- **Poetry**: This project uses Poetry for dependency management. Install it if
  you haven't already:

  ```bash
  curl -sSL https://install.python-poetry.org | python -
  ```

### 2. Clone the Repository

```bash
git clone <repository_url>
cd Jetson-Copilot
```

### 3. Install Dependencies

Activate the Poetry environment and install the required Python packages. Ensure
you have the correct `torch` wheel for your Jetson's CUDA version.

```bash
poetry shell
poetry install
```

**Note on PyTorch**: For optimal performance on Jetson, ensure you install a
PyTorch version compiled for your specific JetPack/CUDA version. You might need
to download a pre-built wheel from NVIDIA's Jetson Zoo or build it from source
if `poetry install` fails to find a compatible `torch` package.

### 4. Model Download

The `Qwen/Qwen2.5-Coder-1.5B-Instruct` model will be automatically downloaded by
the `transformers` library when the server starts for the first time. Ensure
your Jetson has internet access during the initial startup.

## Usage

### 1. Start the Autocompletion Server

Navigate to the project root directory and start the FastAPI server:

```bash
poetry run start
```

The server will typically run on `http://0.0.0.0:8000`. You can verify its
status by accessing `http://<your_jetson_ip>:8000/health` in a web browser or
using `curl`.

For continuous operation, consider running the server in the background using
`nohup` or a process manager like `systemd`:

```bash
nohup poetry run start &
```

### 2. Configure `avant.nvim`

In your Neovim configuration (e.g., `init.lua`), configure `avant.nvim` to use
your Jetson server's endpoint as a Copilot provider. Replace `<your_jetson_ip>`
with the actual IP address of your Jetson Orin Nano.

```lua
require("avante").setup({
  providers = {
    copilot = {
      endpoint = "http://<your_jetson_ip>:8000/complete",
      -- `avant.nvim`'s internal Copilot provider should handle the request/response format.
      -- If you encounter issues, you might need to implement custom `parse_response`
      -- and `parse_curl_args` functions within your avant.nvim configuration
      -- to precisely match the server's API.
    },
  },
  -- Other avant.nvim settings
})
```

## Deployment Considerations

- **Device**: Ensure `CompletionService` is initialized with `device="cuda"` in
  `src/jetson_project/api.py` for GPU acceleration on Jetson.
- **Memory Management**: Monitor GPU and system memory. If you face
  out-of-memory errors, consider reducing `max_new_tokens` or exploring further
  model quantization techniques.
- **Performance**: For maximum inference speed, consider converting the model to
  a TensorRT engine. This is an advanced optimization step.
- **Power Mode**: Set your Jetson to maximum performance power mode for
  consistent inference speeds.

## Project Structure

```bash
. # Project Root
├── pyproject.toml
├── README.md
└── src/
    └── jetson_project/
        ├── __init__.py
        ├── api.py                # FastAPI application and API endpoints
        ├── completion_service.py # Logic for generating code completions
        ├── main.py               # Example usage (can be removed or adapted)
        └── model_handler.py      # Handles Qwen2.5-Coder model loading and inference
```
