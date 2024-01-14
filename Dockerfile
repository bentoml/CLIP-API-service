FROM python:3.11-slim
LABEL authors="sushka0 <barabum@duck.com>, BentoML Authors <contact@bentoml.com>"

COPY . .

RUN pip install .

ENV MODEL_NAME="ViT-B-32:openai"

ENTRYPOINT ["clip-as-service", "serve", "--model-name=${MODEL_NAME}"]