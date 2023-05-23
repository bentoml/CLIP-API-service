# CLIP-API-service

# Intro



# Installation

pip install clip-api-service
clip-api-service serve --model=ViT-B-32:openai


----------


# Use cases:

## Encode
1. Text and image embedding
   1. neural search 
   2. custom ranking

## Rank
2. zero-shot image classification.
   1. a picture of a dog
   2. a picture of a cat
   3. ...
   
3. visual reasoning
   1. this is a picture of {1..10} dogs 
   2. the car is {red, blue, green, ...}
   3. the car is {parked, moving, ...}
   4. the car is {in the street, on the sidewalk, ...}
   5. the big car is on the left, the small car is on the right

# Deployment

1. Build Bento: clip-api-service build --model=ViT-B-32:openai
2. Get BentoCloud account
3. Deploy `bentoml deploy ...`


API reference
* /encode
  * input/output schema
  * img url vs. base64 how to use it
* /rank

clip-api-service list_models