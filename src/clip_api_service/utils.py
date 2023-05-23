from __future__ import annotations

import base64
import io
from typing import Any

import aiohttp
import numpy as np
from bentoml.exceptions import BadInput
from PIL import Image
from pydantic import BaseModel


def base64_to_image(base64_string: str) -> Image:
    image_data = base64.b64decode(base64_string)
    image_io = io.BytesIO(image_data)
    return Image.open(image_io)


async def download_image_from_url(url: str) -> Image.Image:
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, raise_for_status=True) as response:
                image_data = await response.read()
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                return image
        except aiohttp.client_exceptions.ClientResponseError as ex:
            raise BadInput(f"Failed to download image from {url}") from ex


def cosine_similarity(query_embeds, candidates_embeds):
    # Normalize each embedding to a unit vector
    query_embeds /= np.linalg.norm(query_embeds, axis=1, keepdims=True)
    candidates_embeds /= np.linalg.norm(candidates_embeds, axis=1, keepdims=True)

    # Compute cosine similarity
    cosine_similarities = np.matmul(query_embeds, candidates_embeds.T)

    return cosine_similarities


def softmax(scores):
    # Compute softmax scores (probabilities)
    exp_scores = np.exp(
        scores - np.max(scores, axis=-1, keepdims=True)
    )  # Subtract max for numerical stability
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)


class BaseItem(BaseModel):
    def dict(self, *args, **kwargs) -> dict[str, Any]:
        """
        Override the default dict method to exclude None values in the response
        """
        kwargs.pop("exclude_none", None)
        return super().dict(*args, exclude_none=True, **kwargs)


class ListModel(BaseItem):
    __root__: list[Any]

    def __iter__(self):
        return iter(self.__root__)

    def __getitem__(self, item):
        return self.__root__[item]

    def append(self, item):
        return self.__root__.append(item)