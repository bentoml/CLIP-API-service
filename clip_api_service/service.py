from __future__ import annotations

import typing
from typing import Optional, List, Dict

import numpy as np
import asyncio
import bentoml
from bentoml.io import JSON

from .runners import get_clip_runner
from .models import init_model
from .samples import ENCODING_INPUT_SAMPLE, RANKING_INPUT_SAMPLE
from .utils import (
    download_image_from_url,
    base64_to_image,
    cosine_similarity,
    BaseItem,
    ListModel,
    softmax,
)

if typing.TYPE_CHECKING:
    import numpy.typing as npt


class Item(BaseItem):
    text: Optional[str]  # text to encode
    img_uri: Optional[str]  # url for downloading image
    img_blob: Optional[str]  # base64 encoded image


class ItemList(ListModel):
    __root__: List[Item]


class RankInput(BaseItem):
    queries: List[Item]
    candidates: List[Item]


class RankOutput(BaseItem):
    probabilities: List[List[float]]
    cosine_similarities: List[List[float]]


bento_model = init_model()
logit_scale = np.exp(bento_model.info.metadata.get("logit_scale", 4.60517))

clip_runner = get_clip_runner(bento_model)
svc = bentoml.Service(
    "clip-api-service",
    runners=[clip_runner],
)


async def _encode(item: Item) -> npt.NDArray:
    if item.img_uri:
        image = await download_image_from_url(item.img_uri)
        embedding = await clip_runner.encode_image.async_run([image])
    elif item.img_blob:
        image = base64_to_image(item.img_blob)
        embedding = await clip_runner.encode_image.async_run([image])
    else:
        embedding = await clip_runner.encode_text.async_run([item.text])

    return embedding[0]


@svc.api(
    input=JSON.from_sample(ENCODING_INPUT_SAMPLE, pydantic_model=ItemList),
    output=JSON(),
)
async def encode(items: ItemList) -> List[npt.NDArray[float]]:
    results = [_encode(item) for item in items]
    return await asyncio.gather(*results)


@svc.api(
    input=JSON.from_sample(RANKING_INPUT_SAMPLE, pydantic_model=RankInput),
    output=JSON(pydantic_model=RankOutput),
)
async def rank(rank_input: RankInput) -> Dict[str, npt.NDArray[float]]:
    queries = [_encode(query) for query in rank_input.queries]
    candidates = [_encode(item) for item in rank_input.candidates]

    # Encode embeddings
    query_embeds = np.array(await asyncio.gather(*queries))
    candidate_embeds = np.array(await asyncio.gather(*candidates))

    # Compute cosine similarities
    cosine_similarities = cosine_similarity(query_embeds, candidate_embeds)

    # Compute softmax scores
    prob_scores = softmax(logit_scale * cosine_similarities)
    return RankOutput(
        probabilities=prob_scores.tolist(),
        cosine_similarities=cosine_similarities.tolist(),
    )
