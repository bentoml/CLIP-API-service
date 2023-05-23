from __future__ import annotations

import os


def build_bento(model_name: str | None = None, use_gpu: bool = False):
    import bentoml.bentos

    from clip_api_service.models import get_clip_module, init_model

    bento_model = init_model(model_name)
    clip_module = get_clip_module(bento_model)

    bento_file = clip_module.bentofile_path(use_gpu=use_gpu)

    # Get parent directory path
    build_ctx = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bentoml.bentos.build_bentofile(bento_file, build_ctx=build_ctx)
