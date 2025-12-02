from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.get("/")
def check_health():
    pass
    return {"status": "active"}


def start():
    """Entry point for the application script."""
    uvicorn.run("jetson_copilot.api:app", host="0.0.0.0", port=11434, reload=False)
