from __future__ import annotations

SAMPLE_IMG_URL = "https://hips.hearstapps.com/hmg-prod/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg"

ENCODING_INPUT_SAMPLE = [
    {"img_uri": SAMPLE_IMG_URL},
    {"text": "picture of a dog"},
    {"text": "picture of a cat"},
    {"text": "picture of a bird"},
    {"text": "picture of a car"},
    {"text": "picture of a plane"},
    {"text": "picture of a boat"},
]

RANKING_INPUT_SAMPLE = {
    "queries": [
        {"img_uri": SAMPLE_IMG_URL},
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
