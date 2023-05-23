from __future__ import annotations

import importlib
import logging
import os
import typing
from typing import Protocol, cast

import bentoml
from bentoml.exceptions import NotFound
from PIL import Image

if typing.TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt

MODEL_ENV_VAR_KEY = "CLIP_MODEL"
DEFAULT_MODEL_NAME = "openai/clip-vit-large-patch14"

logger = logging.getLogger(__name__)


class CLIPRunnable(bentoml.Runnable):
    """
    CLIPRunnable is a protocol for defining a runnable that implements the actual
    CLIP encoding logic. The interface is used by the CLIP API service to encode text
    and images into CLIP embeddings. See service.py for more details.
    """

    def __init__(self, bento_model: bentoml.Model):
        ...

    def encode_text(self, texts: list[str]) -> npt.NDArray:
        ...

    def encode_image(self, images: list[Image.Image]) -> npt.NDArray:
        ...


class CLIPModule(Protocol):
    """
    CLIPModule is a protocol for defining a CLIP flavor. A CLIP module is a
    Python module that contains a set of CLIP models. Each CLIP model is
    identified by a string name. The CLIP module is responsible for saving
    and loading the CLIP models. It also provides a CLIPRunnable class that
    implements the actual CLIP encoding logic for serving with BentoML.
    """

    MODELS: set[str]

    def download_model(
        self, model_name: str, model: Any, custom_objects: dict[str, Any]
    ) -> bentoml.Model:
        ...

    def clip_runnable(self) -> type[CLIPRunnable]:
        # TODO: add *args, **kwargs for customizing Runnable behavior
        ...

    def bentofile_path(self, use_gpu: bool = False) -> str:
        ...

    def get_bento_model_tag(self, model_name) -> bentoml.Tag:
        ...


class CLIPModuleRegistry:
    def __init__(self):
        self._modules: list[CLIPModule] = []
        self._register_default_clip_modules()

    def register(self, module: str | CLIPModule) -> None:
        if isinstance(module, str):
            self._modules.append(cast(CLIPModule, importlib.import_module(module)))
        else:
            self._modules.append(module)

    def _register_default_clip_modules(self):
        from clip_api_service.models import openai, openclip  # cclip, mclip

        self.register(cast(CLIPModule, openai))
        self.register(cast(CLIPModule, openclip))
        # self.register(cast(CLIPModule, cclip))
        # self.register(cast(CLIPModule, mclip))

    def __iter__(self):
        return iter(self._modules)


CLIP_MODULES = CLIPModuleRegistry()


def save_model(
    model_name: str,
    clip_module: str,
    model: Any,
    custom_objects: dict[str, Any],
) -> bentoml.Model:
    """
    Save a CLIP model to BentoML format. A clip_module must be specified to indicate
    its implementation. The clip_module must be a python module path that defines the
    CLIPModule protocol. A CLIPModule defines the following methods:
        - download_model(model_name: str, model: Any, custom_objects: Dict[str, Any]) -> bentoml.Model
        - clip_runnable() -> type[CLIPRunnable]
        - bentofile_path(use_gpu: bool = False) -> str
        - get_bento_model_tag(model_name) -> bentoml.Tag

    See clip_api_service.models.openai for an example.
    """
    if clip_module not in [
        "clip_api_service.models.openai",
        "clip_api_service.models.openclip",
        "clip_api_service.models.cclip",
        "clip_api_service.models.mclip",
    ]:
        logger.info("Using custom clip module %s", clip_module)

    _clip_module = importlib.import_module(clip_module)
    return _clip_module.save_model(
        model_name, model, custom_objects, labels={"clip_module": clip_module}
    )


def init_model(model_name: str | None = None) -> bentoml.Model:
    """
    Initialize a CLIP model from BentoML format. The model_name must be specified to
    support multiple models in a single BentoML bundle.
    """
    if model_name is None:
        model_name = os.environ.get(MODEL_ENV_VAR_KEY, DEFAULT_MODEL_NAME)

    for clip_module in CLIP_MODULES:
        if model_name in clip_module.MODELS:
            model_tag = clip_module.get_bento_model_tag(model_name)
            try:
                return bentoml.models.get(model_tag)
            except bentoml.exceptions.NotFound:
                return clip_module.download_model(model_name)

    raise NotFound(f"Model {model_name} not supported in any registered clip module")


def get_clip_module(bento_model: bentoml.Model) -> CLIPModule:
    clip_module_name = bento_model.info.labels.get("clip_module", None)
    if clip_module_name is None:
        raise ValueError("clip_module not found in model labels")

    clip_module = cast(CLIPModule, importlib.import_module(clip_module_name))
    assert hasattr(
        clip_module, "clip_runnable"
    ), f'clip_runnable not found in clip_module "{clip_module_name}"'
    return clip_module


def list_models() -> list[str]:
    models = []
    for clip_module in CLIP_MODULES:
        models.extend(clip_module.MODELS)
    return models
