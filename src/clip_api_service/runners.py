from __future__ import annotations

import bentoml

from clip_api_service.models import get_clip_module, init_model


def get_clip_runner(model: str | None | bentoml.Model = None, init_local: bool = False):
    bento_model = model if isinstance(model, bentoml.Model) else init_model(model)
    clip_module = get_clip_module(bento_model)

    runner = bentoml.Runner(
        clip_module.clip_runnable(),
        name="clip_model_runner",
        runnable_init_params={"bento_model": bento_model},
        models=[bento_model],
    )

    if init_local:
        runner.init_local(quiet=True)

    return runner
