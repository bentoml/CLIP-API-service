from __future__ import annotations

from typing import List

import bentoml
import torch
import numpy as np
import numpy.typing as npt
import lancedb
from PIL import Image
from bentoml.exceptions import NotFound

from .models import BENTO_MODEL_TAG


class CLIPRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self, bento_model: bentoml.Model):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = bentoml.transformers.load_model(bento_model).to(self.device)
        self.processor = bento_model.custom_objects['processor']


    @bentoml.Runnable.method(batchable=True)
    def encode_text(self, texts: List[str]) -> npt.NDArray:
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True
        ).to(self.device)
        text_embeddings = self.model.get_text_features(**inputs)
        return text_embeddings.cpu().detach().numpy()


    @bentoml.Runnable.method(batchable=True)
    def encode_image(self, images: List[Image.Image]) -> npt.NDArray:
        inputs = self.processor(
            images=images, return_tensors="pt", padding=True
        ).to(self.device)
        image_embeddings = self.model.get_image_features(**inputs)
        return image_embeddings.cpu().detach().numpy()


def get_clip_runner(model_name: str | None=None, init_local: bool=False):
    try:
        model_obj = bentoml.models.get(model_name or BENTO_MODEL_TAG)
    except NotFound:
        from init_model import init_model
        model_obj = init_model()

    runner = bentoml.Runner(
        CLIPRunnable,
        name="clip_runner",
        runnable_init_params=dict(bento_model=model_obj),
        models=[model_obj]
    )

    if init_local:
        runner.init_local(quiet=True)

    return runner
