FROM python:3.11-slim
LABEL authors="sushka0 <barabum@duck.com>, BentoML Authors <contact@bentoml.com>"

RUN apt update -y
RUN apt-get install gcc python3-dev -y

COPY . .

RUN pip install .

ENV MODEL_NAME=ViT-B-16:openai

ENTRYPOINT clip-api-service serve --model-name=${MODEL_NAME}