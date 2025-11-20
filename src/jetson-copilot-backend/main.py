# main.py
from .model_handler import ModelHandler

# Initialize the ModelHandler
handler = ModelHandler()
handler.load_model()

prompt = "write a quick sort algorithm."

# Generate completion using the ModelHandler
response = handler.generate_completion(prompt, max_new_tokens=512)

print(response)
