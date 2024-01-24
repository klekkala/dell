import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from torch.autograd import Variable



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size, 1, 1)


class VAEBEV(nn.Module):
    def __init__(self, channel_in=3, ch=32, h_dim=512, z=32):
        super(VAEBEV, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channel_in, ch, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch, ch*2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch*2, ch*4, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch*4, ch*8, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z)
        self.fc2 = nn.Linear(h_dim, z)
        self.fc3 = nn.Linear(z, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, ch*8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch*8, ch*4, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch*4, ch*2, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch*2, channel_in, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().cuda()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def recon(self, z):
        z = self.fc3(z)
        return self.decoder(z)

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return self.recon(z), mu, logvar
        
class VAE(nn.Module):
    def __init__(self, channel_in=3, ch=16, z=64, h_dim=512):
        super(VAE, self).__init__()
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
        self.conv_log_var = nn.Conv2d(ch*32, z, 1, 1)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, ch*8, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch*8, ch*4, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch*4, ch*2, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch*2, channel_in, kernel_size=6, stride=2),
        )
        
    def sample(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

        
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.conv_mu(h), self.conv_log_var(h)
        mu = torch.flatten(mu, start_dim=1)
        log_var = torch.flatten(log_var, start_dim=1)
        encoding = self.sample(mu, log_var)
        encoding = mu
        return self.decoder(encoding), mu, log_var
        #return mu


class TBeoEncoder(nn.Module):
    def __init__(self, channel_in=3, ch=16, z=64, h_dim=512, div=1.0):
        super(TBeoEncoder, self).__init__()
        self.div = div
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
        self.joint_layer = nn.Linear(in_features=514, out_features=512, bias=True)

    def forward(self, x, aux):
        x = torch.moveaxis(x, -1, 1)
        x = x/self.div
        #print(torch.max(x))
        x = self.encoder(x)
        x = self.conv_mu(x)
        x = torch.flatten(x, start_dim=1)
        
        #concat aux
        x = torch.concat((x, aux), axis=1)
        x = self.joint_layer(x)
        x = torch.flatten(x, start_dim=1)
        return x


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
    def __init__(self, channel_in=3, ch=16, z=64, h_dim=512, div=1.0):
        super(TEncoder, self).__init__()
        self.div = div
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
        
    def forward(self, x):
        
        x = x/self.div
        #print(torch.max(x), torch.min(x))
        x = self.encoder(x)
        x = self.conv_mu(x)
        #if self.activation == "elu":
        #    embed()
        #    x = torch.flatten(x, start_dim=1)
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
