import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import clip
from PIL import Image

from description import desc_dict

def random_encoder(output_dim, input_shape):
    return np.random.rand(input_shape[0], output_dim)

def get_embeddings_from_env(observation, game_name, is_random=False):

    # CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    text = desc_dict[game_name]

    if observation.max() > 1.0:
        observation = observation.astype(np.uint8)
    image = Image.fromarray(observation)
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_embedding = model.encode_image(image)
        text_embedding = model.encode_text(clip.tokenize([text], truncate=True).to(device))

    image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

    # print("Image Embedding:", image_embedding, image_embedding.shape)
    # print("Text Embedding:", text_embedding, text_embedding.shape)

    if is_random:
        image_embedding_random = random_encoder(output_dim=image_embedding.shape[1], input_shape=image_embedding.shape)
        text_embedding_random = random_encoder(output_dim=image_embedding.shape[1], input_shape=image_embedding.shape)
        return image_embedding_random, text_embedding_random

    return image_embedding.cpu().numpy(), text_embedding.cpu().numpy()
