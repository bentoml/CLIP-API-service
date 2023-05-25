from __future__ import annotations

import typing

import bentoml
from PIL import Image

from clip_api_service.models import CLIPRunnable

if typing.TYPE_CHECKING:
    import numpy.typing as npt

MODELS = {
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-large-patch14-336",
}


def get_bento_model_tag(model_name: str) -> bentoml.Tag:
    return bentoml.Tag.from_str(model_name.replace("openai/clip-", "openai-clip:"))


def download_model(model_name: str) -> bentoml.Model:
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    bento_model_tag = get_bento_model_tag(model_name)

    bentoml.transformers.save_model(
        str(bento_model_tag),
        model,
        custom_objects={"processor": processor},
        labels={
            "clip_module": "clip_api_service.models.openai",
        },
        metadata={
            "logit_scale": model.logit_scale.item(),
        },
    )
    return bentoml.models.get(bento_model_tag)

def bentofile_path(use_gpu: bool = False) -> str:
    import os
    build_ctx = os.path.dirname(os.path.dirname(__file__))

    declaration = {
        False: os.path.join(build_ctx, "bentofiles", "bentofile.openai.cpu.yaml"),
        True: os.path.join(build_ctx, "bentofiles", "bentofile.openai.gpu.yaml"),
    }

    return declaration[use_gpu]

class OpenAICLIPRunnable(CLIPRunnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self, bento_model: bentoml.Model):
        import torch

        if torch.cuda.is_available():
            self.device = "cuda"
            # by default, torch.FloatTensor will be used on CPU.
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            self.device = "cpu"

        self.model = bento_model.load_model().to(self.device)
        self.processor = bento_model.custom_objects["processor"]

    @bentoml.Runnable.method(batchable=True)
    def encode_text(self, texts: list[str]) -> npt.NDArray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(
            self.device
        )
        text_embeddings = self.model.get_text_features(**inputs)
        return text_embeddings.cpu().detach().numpy()

    @bentoml.Runnable.method(batchable=True)
    def encode_image(self, images: list[Image.Image]) -> npt.NDArray:
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(
            self.device
        )
        image_embeddings = self.model.get_image_features(**inputs)
        return image_embeddings.cpu().detach().numpy()


def clip_runnable():
    return OpenAICLIPRunnable
