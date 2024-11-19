import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class RandomEncoder(nn.Module):
    def __init__(self, input_channels=3, embedding_dim=512):
        super(RandomEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.pool(x) 
        x = x.squeeze(-1).squeeze(-1)
        return x


def random_encoder(observation):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RandomEncoder(input_channels=observation.shape[1], embedding_dim=512).to(device)

    if isinstance(observation, torch.Tensor):
        print("Observation shape:", observation.shape)
        batch_size = observation.shape[0]
        
        image_embeddings = []
        for i in range(batch_size):
            single_image = observation[i].unsqueeze(0).to(device)

            with torch.no_grad():
                image_embedding = model(single_image)
                image_embeddings.append(image_embedding)

        image_embeddings = torch.cat(image_embeddings, dim=0).to(device)
        return image_embeddings
    else:
        raise ValueError("Unexpected observation type")