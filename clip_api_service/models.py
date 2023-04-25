import os
import re

DEFAULT_MODEL_NAME = "openai/clip-vit-large-patch14"
MODEL_NAME = os.environ.get("CLIP_MODEL_NAME", DEFAULT_MODEL_NAME)
BENTO_MODEL_TAG = "clip:" + re.sub('[^a-zA-Z0-9]+', '-', MODEL_NAME)

def init_model():
    import bentoml
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    bentoml.transformers.save_model(
        BENTO_MODEL_TAG,
        model,
        custom_objects={'processor': processor}
    )
    return bentoml.models.get(BENTO_MODEL_TAG)

if __name__ == "__main__":
    init_model()
