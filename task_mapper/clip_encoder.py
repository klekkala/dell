import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import clip
from PIL import Image

def clip_encoder(observation):
    # CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    if isinstance(observation, torch.Tensor):
        print("Observation shape:", observation.shape)
        batch_size = observation.shape[0]
        
        image_embeddings = []
        for i in range(batch_size):
            single_image = observation[i].cpu().numpy()
            
            # (C, H, W) -> (H, W, C)
            if single_image.ndim == 3:
                if single_image.shape[0] in [1, 3, 4]:
                    single_image = np.transpose(single_image, (1, 2, 0))
                elif single_image.shape[2] in [1, 3, 4]:
                    pass
                else:
                    raise ValueError("Unexpected observation shape. Expected 3D tensor with shape (C, H, W) or (H, W, C), got {}".format(single_image.shape))
            else:
                raise ValueError("Unexpected observation shape. Expected 3D tensor, got {}".format(single_image.shape))
            # if single_image.ndim == 3 and single_image.shape[0] in [1, 3]:
            #     single_image = np.transpose(single_image, (1, 2, 0))
            # else:
            #     raise ValueError("Unexpected observation shape. Expected 3D tensor with shape (C, H, W), got {}".format(single_image.shape))
            
            # if observation.max() > 1.0:
            #     observation = observation.astype(np.uint8)
            image = Image.fromarray(single_image.astype(np.uint8))
            image = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_embedding = model.encode_image(image)
                image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
                image_embeddings.append(image_embedding)

        image_embeddings = torch.cat(image_embeddings, dim=0).to(device)
        return image_embeddings
    else:
        raise ValueError("Unexpected CLIP observation type")