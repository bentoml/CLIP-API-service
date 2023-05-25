from __future__ import annotations

import typing

import bentoml
import torch
from PIL import Image

from clip_api_service.models import CLIPRunnable

if typing.TYPE_CHECKING:
    import numpy.typing as npt

MODELS = {
    "RN50:openai",
    "RN50:yfcc15m",
    "RN50:cc12m",
    # "RN50-quickgelu:openai",
    # "RN50-quickgelu:yfcc15m",
    # "RN50-quickgelu:cc12m",
    "RN101:openai",
    "RN101:yfcc15m",
    # "RN101-quickgelu:openai",
    # "RN101-quickgelu:yfcc15m",
    "RN50x4:openai",
    "RN50x16:openai",
    "RN50x64:openai",
    "ViT-B-32:openai",
    "ViT-B-32:laion400m_e31",
    "ViT-B-32:laion400m_e32",
    "ViT-B-32:laion2b_e16",
    "ViT-B-32:laion2b_s34b_b79k",
    # "ViT-B-32:datacomp_m_s128m_b4k",
    # "ViT-B-32:commonpool_m_clip_s128m_b4k",
    # "ViT-B-32:commonpool_m_laion_s128m_b4k",
    # "ViT-B-32:commonpool_m_image_s128m_b4k",
    # "ViT-B-32:commonpool_m_text_s128m_b4k",
    # "ViT-B-32:commonpool_m_basic_s128m_b4k",
    # "ViT-B-32:commonpool_m_s128m_b4k",
    # "ViT-B-32:datacomp_s_s13m_b4k",
    # "ViT-B-32:commonpool_s_clip_s13m_b4k",
    # "ViT-B-32:commonpool_s_laion_s13m_b4k",
    # "ViT-B-32:commonpool_s_image_s13m_b4k",
    # "ViT-B-32:commonpool_s_text_s13m_b4k",
    # "ViT-B-32:commonpool_s_basic_s13m_b4k",
    # "ViT-B-32:commonpool_s_s13m_b4k",
    # "ViT-B-32-quickgelu:openai",
    # "ViT-B-32-quickgelu:laion400m_e31",
    # "ViT-B-32-quickgelu:laion400m_e32",
    "ViT-B-16:openai",
    "ViT-B-16:laion400m_e31",
    "ViT-B-16:laion400m_e32",
    # "ViT-B-16:laion2b_s34b_b88k",
    # "ViT-B-16:datacomp_l_s1b_b8k",
    # "ViT-B-16:commonpool_l_clip_s1b_b8k",
    # "ViT-B-16:commonpool_l_laion_s1b_b8k",
    # "ViT-B-16:commonpool_l_image_s1b_b8k",
    # "ViT-B-16:commonpool_l_text_s1b_b8k",
    # "ViT-B-16:commonpool_l_basic_s1b_b8k",
    # "ViT-B-16:commonpool_l_s1b_b8k",
    "ViT-B-16-plus-240:laion400m_e31",
    "ViT-B-16-plus-240:laion400m_e32",
    "ViT-L-14:openai",
    "ViT-L-14:laion400m_e31",
    "ViT-L-14:laion400m_e32",
    "ViT-L-14:laion2b_s32b_b82k",
    # "ViT-L-14:datacomp_xl_s13b_b90k",
    # "ViT-L-14:commonpool_xl_clip_s13b_b90k",
    # "ViT-L-14:commonpool_xl_laion_s13b_b90k",
    # "ViT-L-14:commonpool_xl_s13b_b90k",
    "ViT-L-14-336:openai",
    "ViT-H-14:laion2b_s32b_b79k",
    "ViT-g-14:laion2b_s12b_b42k",
    "ViT-g-14:laion2b_s34b_b88k",
    "ViT-bigG-14:laion2b_s39b_b160k",
    "roberta-ViT-B-32:laion2b_s12b_b32k",
    "xlm-roberta-base-ViT-B-32:laion5b_s13b_b90k",
    "xlm-roberta-large-ViT-H-14:frozen_laion5b_s13b_b90k",
    # "convnext_base:laion400m_s13b_b51k",
    # "convnext_base_w:laion2b_s13b_b82k",
    # "convnext_base_w:laion2b_s13b_b82k_augreg",
    # "convnext_base_w:laion_aesthetic_s13b_b82k",
    # "convnext_base_w_320:laion_aesthetic_s13b_b82k",
    # "convnext_base_w_320:laion_aesthetic_s13b_b82k_augreg",
    # "convnext_large_d:laion2b_s26b_b102k_augreg",
    # "convnext_large_d_320:laion2b_s29b_b131k_ft",
    # "convnext_large_d_320:laion2b_s29b_b131k_ft_soup",
    # "convnext_xxlarge:laion2b_s34b_b82k_augreg",
    # "convnext_xxlarge:laion2b_s34b_b82k_augreg_rewind",
    # "convnext_xxlarge:laion2b_s34b_b82k_augreg_soup",
    # "coca_ViT-B-32:laion2b_s13b_b90k",
    # "coca_ViT-B-32:mscoco_finetuned_laion2b_s13b_b90k",
    # "coca_ViT-L-14:laion2b_s13b_b90k",
    # "coca_ViT-L-14:mscoco_finetuned_laion2b_s13b_b90k",
    # "EVA01-g-14:laion400m_s11b_b41k",
    # "EVA01-g-14-plus:merged2b_s11b_b114k",
    # "EVA02-B-16:merged2b_s8b_b131k",
    # "EVA02-L-14:merged2b_s4b_b131k",
    # "EVA02-L-14-336:merged2b_s6b_b61k",
    # "EVA02-E-14:laion2b_s4b_b115k",
    # "EVA02-E-14-plus:laion2b_s9b_b144k",
}



def get_bento_model_tag(model_name: str) -> bentoml.Tag:
    model_version = model_name.replace(":", ".")
    return bentoml.Tag("openclip", model_version.lower())


def download_model(model_name: str) -> bentoml.Model:
    import open_clip

    bento_model_tag = get_bento_model_tag(model_name)
    _model_name, _pretrained = model_name.split(":")

    model, _, processor = open_clip.create_model_and_transforms(
        _model_name, pretrained=_pretrained
    )
    tokenizer = open_clip.get_tokenizer(_model_name)

    bentoml.pytorch.save_model(
        str(bento_model_tag),
        model,
        custom_objects={"processor": processor, "tokenizer": tokenizer},
        labels={
            "clip_module": "clip_api_service.models.openclip",
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
        False: os.path.join(build_ctx, "bentofiles", "bentofile.openclip.cpu.yaml"),
        True: os.path.join(build_ctx, "bentofiles", "bentofile.openclip.gpu.yaml"),
    }

    return declaration[use_gpu]

class OpenClipRunnable(CLIPRunnable):
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

        self.model = bento_model.load_model().to(self.device).eval()
        self.processor = bento_model.custom_objects["processor"]
        self.tokenizer = bento_model.custom_objects["tokenizer"]

    @bentoml.Runnable.method(batchable=True)
    def encode_text(self, texts: list[str]) -> npt.NDArray:
        texts_encodings = self.tokenizer(texts)
        with torch.inference_mode():
            text_embeddings = self.model.encode_text(texts_encodings)
            return text_embeddings.cpu().detach().numpy()

    @bentoml.Runnable.method(batchable=True)
    def encode_image(self, images: list[Image.Image]) -> npt.NDArray:
        image_encodings = torch.stack([self.processor(image) for image in images])
        with torch.inference_mode():
            image_embeddings = self.model.encode_image(image_encodings)
            return image_embeddings.cpu().detach().numpy()


def clip_runnable():
    return OpenClipRunnable
