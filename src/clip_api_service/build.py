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

    # Get parent directory path
    build_ctx = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bentoml.bentos.build_bentofile(bento_file, build_ctx=build_ctx)
