import sys
from PIL import Image
from datetime import datetime
import tempfile
import yaml
import random
import numpy as np
import math, argparse, csv, copy, time, os
from pathlib import Path
#import graph_tool.all as gt
import argparse
from IPython import embed
import ray
from ray.rllib.utils.annotations import override
from ray import air, tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

import train
import configs
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print, UnifiedLogger, Logger, LegacyLoggerCallback
from ray.tune.registry import get_trainable_cls
from arguments import get_args
import datetime


if __name__ == "__main__":
    # Load the hdf5 files into a global variable

    torch, nn = try_import_torch()
    args = get_args()


    ray.init(local_mode=args.local_mode)

    
    #log directory
    suffix = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    str_logger = args.prefix + "_" + args.set + "_" + args.shared + "_" + args.backbone + "_" + args.policy + "_" + str(args.kl_coeff) + "_" + str(args.buffer_size) + "_" + str(args.batch_size) + "_" + str(args.div) + "_" + args.temporal + "/" + suffix



    #Before you start training. run evaluate to check how much reward can a random agent do
    if args.eval:
        print("Implement eval")


    if args.train:
        train.seq_train(str_logger)
