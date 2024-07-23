import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from torch.autograd import Variable




#class TEncoder(nn.Module):
#    def __init__(self, channel_in=3, ch=16, z=64, h_dim=512, activation="relu"):
#        super(TEncoder, self).__init__()
#        self.encoder = nn.Sequential(
#                nn.Conv2d(4, 32, 8, stride=4, padding=0),
#                nn.ReLU(),
#                nn.Conv2d(32, 64, 4, stride=2, padding=0),
#                nn.ReLU(),
#                nn.Conv2d(64, 64, 3, stride=1, padding=0),
#                nn.ReLU(),
#                nn.Flatten()
#                )
#
#        self.conv_mu = nn.Sequential(
#                nn.Linear(3136, z),
#                nn.Tanh()
#                )
#
#    def forward(self, x):
#        print(x.shape)
#        x = x/255.0
#        #x = torch.moveaxis(x, -1, 1)
#        #print(torch.max(x))
#        x = self.encoder(x)
#        x = self.conv_mu(x)
#        x = torch.reshape(x, (x.shape[0], x.shape[1], 1, 1))
#        return x


class TEncoder(nn.Module):
    def __init__(self, channel_in=3, ch=16, z=64, h_dim=512, div=1.0, is_task_mapper=False):
        super(TEncoder, self).__init__()
        self.div = div
        self.is_task_mapper = is_task_mapper
        assert(self.div == 255.0 or self.div == 1.0)
        
        self.encoder = nn.Sequential(
            nn.ZeroPad2d((2, 2, 2, 2)),
            nn.Conv2d(channel_in, ch, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(ch, ch*2, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(ch*2, ch*32, kernel_size=(11, 11), stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_mu = nn.Conv2d(ch*32, z, 1, 1)

        self.task_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(z * 1 * 1, h_dim),  # assuming the final feature map size is 10x10, it can be adjusted according to the actual size
            nn.ReLU(),
            nn.Linear(h_dim, 64)  # TODO: num_polices is the number of polices, adjust it
        )
        
    def forward(self, x):
        
        x = x/self.div
        #print(torch.max(x), torch.min(x))
        x = self.encoder(x)
        x = self.conv_mu(x)
        #if self.activation == "elu":
        #    embed()
        #    x = torch.flatten(x, start_dim=1)
        if self.is_task_mapper: 
            task_logits = self.task_classifier(x)
            return x, task_logits
        return x


class Encoder(TEncoder):
    def __init__(self, channel_in=3, ch=16, z=64, h_dim=512):
        super(Encoder, self).__init__(channel_in=channel_in, ch=ch, z=z, h_dim=h_dim)
        self.adapter = nn.Conv2d(z, z, 1, 1)


    def forward(self, x):
        x = self.encoder(x)
        x = self.conv_mu(x)
        x = self.adapter(x)
        #x = torch.flatten(x, start_dim=1)
        return x
