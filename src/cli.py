import os
import typer
import subprocess

from typing_extensions import Annotated
from clip_api_service import list_models as _list_models
from clip_api_service.build import build_bento
from clip_api_service.models import MODEL_ENV_VAR_KEY, DEFAULT_MODEL_NAME
app = typer.Typer()

@app.command()
def serve(model_name: Annotated[str, typer.Option(help="CLIP Model name")] = DEFAULT_MODEL_NAME):
    env = os.environ.copy()
    env[MODEL_ENV_VAR_KEY] = model_name

    try:
        subprocess.run(["bentoml", "serve", "clip_api_service.service:svc"], env=env, check=True)
    except subprocess.CalledProcessError as e:
        typer.echo(f"Command 'bentoml serve {model_name}' failed with error code {e.returncode}")


@app.command()
def build(
    model_name: Annotated[str, typer.Option(help="CLIP Model name")] = DEFAULT_MODEL_NAME,
    use_gpu: Annotated[bool, typer.Option(help="Use GPU for build")] = False,
):
    build_bento(model_name=model_name, use_gpu=use_gpu)

@app.command()
def list_models():
    print(_list_models())

if __name__ == "__main__":
    app()