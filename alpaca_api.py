import os
from typing import Optional

from fastapi import FastAPI

from alpaca import Alpaca, InferenceRequest

global_model: Optional[Alpaca] = None
app = FastAPI()


def get_model() -> Alpaca:
    assert global_model is not None
    return global_model


@app.on_event("startup")
async def startup_event():
    global global_model
    global_model = Alpaca(os.environ["ALPACA_CLI_PATH"], os.environ["ALPACA_MODEL_PATH"])


@app.on_event("shutdown")
def shutdown_event():
    global global_model
    if global_model is not None:
        global_model.stop()
        global_model = None


@app.get("/")
def get_system_info() -> Optional[dict]:
    return get_model().system_info


@app.post("/run")
def run(
    input: InferenceRequest = InferenceRequest(
        input_text="Are alpacas afraid of snakes?"
    ).wrap_with_default_prompt(),
) -> dict:
    """Runs the text through the model as is."""
    return get_model().run(input)


@app.post("/run_simple")
def run_simple(
    input: InferenceRequest = InferenceRequest(input_text="Are alpacas afraid of snakes?"),
) -> dict:
    """Wraps the text with a standard prompt before passing it to the model."""
    return get_model().run_simple(input)
