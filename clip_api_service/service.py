from __future__ import annotations
from typing import Optional, List, Dict

import io
import asyncio

import bentoml
import lancedb
from lancedb import embeddings

import numpy as np
import pandas as pd
import numpy.typing as npt
from PIL import Image
from pydantic import BaseModel
from bentoml.io import JSON, PandasDataFrame, Image

from .runners import get_clip_runner
from .utils import download_image_from_url, base64_to_image

DEFAULT_RETURN_LIMIST=9

clip_runner = get_clip_runner()

svc = bentoml.Service(
    "image-search-service",
    runners=[clip_runner],
)

SAMPLES = [
    {"image_uri": "https://hips.hearstapps.com/hmg-prod/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg"},
    {"text": "picture of a dog"},
    {"text": "picture of a cat"},
]

class Item(BaseModel):
    text: Optional[str]
    image_uri: Optional[str]
    image_b64: Optional[str]

class ItemList(BaseModel):
    __root__: List[Item]

    def __iter__(self):
        return iter(self.__root__)

    def __getitem__(self, item):
        return self.__root__[item]


async def _encode(item: Item) -> npt.NDArray:
    if item.image_uri:
        image = await download_image_from_url(item.image_uri)
        embedding = await clip_runner.encode_image.async_run([image])
    elif item.image_b64:
        image = base64_to_image(item.image_b64)
        embedding = await clip_runner.encode_image.async_run([image])
    else:
        embedding = await clip_runner.encode_text.async_run([item.text])

    return embedding[0]


@svc.api(input=JSON.from_sample(SAMPLES, pydantic_model=ItemList), output=JSON())
async def encode(items: ItemList) -> List[npt.NDArray]:
    results = [_encode(item) for item in items]
    return await asyncio.gather(*results)

@svc.api(input=Image(), output=JSON())
async def encode_image(image: Image.Image) -> npt.NDArray:
    return (await clip_runner.encode_image.async_run([image]))[0]
