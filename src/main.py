import pprint

from clip_api_service.service import svc

input_sample = {
    "queries": [
        {
            "img_uri": "https://hips.hearstapps.com/hmg-prod/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg"
        },
        {
            "img_uri": "https://www.carscoops.com/wp-content/uploads/2023/02/2022-Mercedes-CLS-1024x576.jpg"
        },
        {
            "img_uri": "https://static.scientificamerican.com/sciam/cache/file/7A715AD8-449D-4B5A-ABA2C5D92D9B5A21_source.png"
        },
    ],
    "candidates": [
        {"text": "picture of a dog"},
        {"text": "picture of a cat"},
        {"text": "picture of a bird"},
        {"text": "picture of a car"},
        {"text": "picture of a plane"},
        {"text": "picture of a boat"},
    ],
}


def test():
    import asyncio

    from clip_api_service.service import RankInput

    rank_input = RankInput.parse_obj(input_sample)

    svc.init_local()
    result = asyncio.run(svc.apis.rank(rank_input))
    pprint.pprint(result)


if __name__ == "__main__":
    # server = bentoml.HTTPServer(svc)
    # server.start(blocking=False)
    test()

#%%
from clip_api_service.build import build_bento
# %%
build_bento(model_name="ViT-B-32:openai")
# %%
