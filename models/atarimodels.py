
#This file contains 2 models singletask, multitask


import functools
from typing import Optional

import numpy as np
import tree
from gym.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
from ray.rllib.models.preprocessors import get_preprocessor, Preprocessor
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.models.torch.model import TorchModel
from ray.rllib.models.base_model import RecurrentModel, Model, ModelIO
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from typing import Dict, List
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.visionnet import VisionNetwork
# from atari_vae import VAE
from atari_vae import Encoder, TEncoder
from IPython import embed

from ray.rllib.models.torch.misc import (
    normc_initializer,
    same_padding,
    SlimConv2d,
    SlimFC,
)

from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
torch, nn = try_import_torch()


# The global, shared layer to be used by both models.
# this model outputs a 512 latent dimension
ATARI_GLOBAL_SHARED_BACKBONE = TEncoder(channel_in=4, ch=32, z=512)


#if using lstm this could be used:
# ATARI_GLOBAL_SHARED_1CHANNEL_VAE = VAE(channel_in=1, z=512)

#if using 4stack this could be used:
# ATARI_GLOBAL_SHARED_4STACK_VAE = VAE(channel_in=1, z=512)

ATARI_GLOBAL_SHARED_POLICY = nn.Sequential(
    nn.ZeroPad2d((0, 0, 0, 0)),
    nn.Conv2d(512, 6, kernel_size=(1, 1), stride=(1, 1))
    )

ATARI_GLOBAL_SHARED_VF = nn.Linear(512, 1, bias=True)

class SingleAtariModel(VisionNetwork):

    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        chan_in = 1 if model_config['custom_model_config']['temporal'] == 'notemp' or model_config['custom_model_config']['temporal'] == 'lstm' else 4

        activation = "elu" if model_config['custom_model_config']['temporal'] == 'lstm' else 'relu'
        
        if "RESNET" in model_config['custom_model_config']['backbone'] and "DUAL" in model_config['custom_model_config']['backbone']:            
            self._convs = TEncoder(channel_in=chan_in, ch=64, z=512)
        elif "RESNET" in model_config['custom_model_config']['backbone']:
            self._convs = TEncoder(channel_in=chan_in, ch=64, z=512)
        elif 'DUAL' in model_config['custom_model_config']['backbone']:
            self._convs = Encoder(channel_in=chan_in, ch=32, z=512)
        elif '4STACK_CONT' in model_config['custom_model_config']['backbone']:
            self._convs = TEncoder(channel_in=chan_in, ch=32, z=512)
        # elif '4STACK_VAE' in model_config['custom_model_config']['backbone']:
            # self._convs = VAE(channel_in=chan_in, ch=32, z=512)
        else:
            self._convs = TEncoder(channel_in=chan_in, ch=32, z=512, div=model_config['custom_model_config']['div'])


        #if 'e2e' not in model_config['custom_model_config']['backbone'] and model_config['custom_model_config']['backbone'] != 'random':
        #if 'e2e' not in model_config['custom_model_config']['backbone']:
            #if model_config['custom_model_config']['backbone'] == 'random':
            #    self._convs = VAE(channel_in=observation_space.shape[-1], z=512)

            
        #else:
            #in the case of e2e or random
        #    self._convs = backbone
        #    if model_config['custom_model_config']['temporal'] == 'lstm' or model_config['custom_model_config']['temporal'] == 'attention':
        #        self._convs = LSTMVAE(channel_in=observation_space.shape[-1], z=512)
        #    else:
        #        self._convs = VAE(channel_in=observation_space.shape[-1], z=512)
        #else:
        #    raise NotImplementedError("lol")
        
        
        if "e2e" not in model_config['custom_model_config']['backbone_path'] and "random" not in model_config['custom_model_config']['backbone_path']:
            print(model_config['custom_model_config']['backbone_path'])
            print("loading model weights")
            #checkpoint = torch.load(model_config['custom_model_config']['backbone_path'], map_location="cpu")
            
            checkpoint = torch.load(model_config['custom_model_config']['backbone_path'])
            self._convs.load_state_dict(checkpoint['model_state_dict'])
            print("loss_log is", np.mean(checkpoint['loss_log']))
            #embed() 
        if not model_config['custom_model_config']['train_backbone']:
            print("freezing encoder layers")
            #freeze the entire backbone
            self._convs.eval()
            for param in self._convs.parameters():
                param.requires_grad = False
            #unfreeze the adapter
            #self._convs.conv_mu.train()
            #for param in self._convs.conv_mu.parameters():
            #    param.requires_grad = True
        
        #dd = self.trainable_variables(True)
        #embed()

#this is class is reused for every game/city/town
#this is equivalent to a spec in rl_module api
class SharedBackboneAtariModel(SingleAtariModel):

    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        super().__init__(observation_space, action_space, num_outputs, model_config, name)
        self._convs = ATARI_GLOBAL_SHARED_BACKBONE

#this is class is reused for every game/city/town
#this is equivalent to a spec in rl_module api

class SharedBackbonePolicyAtariModel(SingleAtariModel):

    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        super().__init__(observation_space, action_space, num_outputs, model_config, name)
        
        self._logits = ATARI_GLOBAL_SHARED_POLICY
        self._value_branch = ATARI_GLOBAL_SHARED_VF


class AtariCNNV2PlusRNNModel(TorchRNN, nn.Module):
    """A conv. + recurrent torch net example using a pre-trained MobileNet."""

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name
    ):

        TorchRNN.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        #HARDCODED!!
        self.num_outputs = 6

        self.lstm_state_size = 512
        self.visual_size_out = 512

        self.cnn_shape = [1, 84, 84]
        self.visual_size_in = self.cnn_shape[0] * self.cnn_shape[1] * self.cnn_shape[2]
        # MobileNetV2 has a flat output of (1000,).


        # Load the MobileNetV2 from torch.hub.
        if "RESNET" in model_config['custom_model_config']['backbone'] and "DUAL" in model_config['custom_model_config']['backbone']:            
            self._convs = Encoder(channel_in=1, ch=64, z=512, activation="elu")
        elif "RESNET" in model_config['custom_model_config']['backbone']:
            self._convs = TEncoder(channel_in=1, ch=64, z=512, activation="elu")
        elif 'DUAL' in model_config['custom_model_config']['backbone']:
            self._convs = Encoder(channel_in=1, ch=32, z=512, activation="elu")
        else:
            #self._convs = TEncoder(channel_in=1, ch=32, z=512)
            self._convs = TEncoder(channel_in=1, ch=32, z=512, activation="elu")

        
        print(self._convs)
        self.lstm = nn.LSTM(
            self.visual_size_out, self.lstm_state_size, batch_first=True
        )

        # Postprocess LSTM output with another hidden layer and compute values.
        self.logits = SlimFC(self.lstm_state_size, self.num_outputs)
        self.value_branch = SlimFC(self.lstm_state_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

        if "e2e" not in model_config['custom_model_config']['backbone_path'] and "random" not in model_config['custom_model_config']['backbone_path']:
            print(model_config['custom_model_config']['backbone_path'])
            print("loading model weights")
            checkpoint = torch.load(model_config['custom_model_config']['backbone_path'], map_location="cpu")
            
            lstm_ckpt = {}
            convs_ckpt = {}
            #for eachkey in checkpoint['model_state_dict']:
                #if 'lstm' in eachkey:
                #    newkey = eachkey.replace('lstm.', '')
                #    lstm_ckpt[newkey] = checkpoint['model_state_dict'][eachkey]
                #else:
                #    if 'conv_mu' in eachkey:
                #        newkey = eachkey.replace('encoder.', '')
                #    else:
                #        newkey = eachkey.replace('encoder.encoder', 'encoder')
                #convs_ckpt[newkey] = checkpoint['model_state_dict'][eachkey]


            #for each in self._convs.named_parameters():
            #    print(each[0])

            #create cnn_modstdict
            #self._convs.load_state_dict(convs_ckpt)

            #create lstm_modstdict
            #self.lstm.load_state_dict(lstm_ckpt)
            self._convs.load_state_dict(checkpoint['model_state_dict'])

        if not model_config['custom_model_config']['train_backbone']:
            print("freezing encoder layers")
            #freeze the entire backbone
            self._convs.eval()
            for param in self._convs.parameters():
                param.requires_grad = False

            #self.lstm.eval()
            #for param in self.lstm.parameters():
            #    param.requires_grad = False

        
    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        # Create image dims.
        vision_in = torch.reshape(inputs, [-1] + self.cnn_shape)
        
        # Flatten.
        vision_out = self._convs(vision_in)
        #vision_out = torch.flatten(self._convs(vision_in), start_dim=1)
        
        vision_out_time_ranked = torch.reshape(
            vision_out, [inputs.shape[0], inputs.shape[1], vision_out.shape[-1]]
        )

        if len(state[0].shape) == 2:
            state[0] = state[0].unsqueeze(0)
            state[1] = state[1].unsqueeze(0)
        # Forward through LSTM.
        self._features, [h, c] = self.lstm(vision_out_time_ranked, state)
        # Forward LSTM out through logits layer and value layer.
        logits = self.logits(self._features)
        return logits, [h.squeeze(0), c.squeeze(0)]

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        h = [
            list(self._convs.modules())[-1]
            .weight.new(1, self.lstm_state_size)
            .zero_()
            .squeeze(0),
            list(self._convs.modules())[-1]
            .weight.new(1, self.lstm_state_size)
            .zero_()
            .squeeze(0),
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])
    
