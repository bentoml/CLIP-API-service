from __future__ import annotations

from clip_api_service.models import (
    CLIP_MODULES as modules_registry,
)
from clip_api_service.models import (
    init_model,
    list_models,
    save_model,
)
from clip_api_service.runners import get_clip_runner

# def start_clip_server(in_process: bool=False) -> tuple[bentoml.HTTPServer, bentoml.client.HTTPClient]:
#     from clip_api_service.service import svc
#
#     if in_process:
#         pass
#     else:
#         server = bentoml.HTTPServer(svc)
#         server.start(blocking=False)
#         client = server.get_client()
#         return server, client


__all__ = [
    "get_clip_runner",
    "save_model",
    "init_model",
    "modules_registry",
    "list_models",
]
