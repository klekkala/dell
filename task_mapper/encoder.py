import numpy as np
import torch
import clip

def clip_encoder():
    """
    This function replaces the previous VAE model with CLIP for encoding images and text.
    It loads the CLIP model, freezes the encoder (image part), and returns it.
    """
    # CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Get the image encoder part from CLIP
    encoder = model.encode_image

    print("Freezing CLIP encoder layers")
    # Freeze the encoder layers
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    return encoder, preprocess  # Return both encoder and preprocessing function