from __future__ import annotations

from clip_api_service.models import MODEL_ENV_VAR_KEY

import os


def build_bento(model_name: str | None = None, use_gpu: bool = False):
    import bentoml.bentos

    from clip_api_service.models import get_clip_module, init_model

    bento_model = init_model(model_name)
    clip_module = get_clip_module(bento_model)

    bento_file = clip_module.bentofile_path(use_gpu=use_gpu)

    # Setting Model name as environment variable to pass 
    # into subprocess for bento build
    if model_name:
        os.environ[MODEL_ENV_VAR_KEY] = model_name

    # Generate a service file
    # TODO: Refactor the code gen to a mode generalize function
    # TODO: Use temporary file instead of writing to the source file 
    src_service_file_path = os.path.join(os.path.dirname(__file__), "_service.py")
    target_service_file_path = os.path.join(os.path.dirname(__file__), "service.py")

    with open(src_service_file_path, 'r') as src_file, open(target_service_file_path, 'w') as target_file:
        data = src_file.read().replace('__model_name__', model_name)
        target_file.write(data)

    # Get parent directory path
    build_ctx = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bentoml.bentos.build_bentofile(bento_file, build_ctx=build_ctx)

    # Remove the generated service file
    os.remove(target_service_file_path)