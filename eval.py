import sys
from PIL import Image
from datetime import datetime
import tempfile
import yaml
import random
import numpy as np
import math, argparse, csv, copy, time, os
from pathlib import Path
import envs
from arguments import get_args
import ray
import configs
#import graph_tool.all as gt
from ray.rllib.utils.annotations import override
from ray import air, tune
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print, UnifiedLogger, Logger, LegacyLoggerCallback
from ray.tune.registry import get_trainable_cls
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from models.atarimodels import SingleAtariModel, SharedBackboneAtariModel, SharedBackbonePolicyAtariModel, AtariCNNV2PlusRNNModel
from ray.rllib.algorithms.ppo import PPOConfig
from typing import Dict, Tuple
import gym
import distutils.dir_util
from gym import spaces
from ray.rllib.policy.sample_batch import SampleBatch
from IPython import embed
import shutil
import distutils.dir_util
from pathlib import Path
from ray.rllib.algorithms.algorithm import Algorithm
from typing import List, Optional, Type, Union
from ray.rllib.utils.typing import AlgorithmConfigDict, ResultDict
from ray.tune.schedulers import PopulationBasedTraining, pb2
args = get_args()


import json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import clip
from PIL import Image

from sklearn.metrics.pairwise import cosine_similarity

from description import desc_dict


def pick_config_env(str_env):
    # modify atari_config to incorporate the environments
    
    if args.env_name == 'atari':
        use_config = configs.atari_config
        use_env = envs.atari[str_env]
    elif args.env_name == 'beogym':
        use_config = configs.beogym_config
        use_env = envs.beogym[str_env]
    elif env_name == 'carla':
        use_config = configs.carla
        use_env = envs.carla[str_env]
    return use_config, use_env

def random_encoder(output_dim, input_shape):
    return np.random.rand(input_shape[0], output_dim)


mar_dict = {}
min_rewards_dict = {}
max_rewards_dict = {}
embedding_dict = {}
with open("embedding_dict.json", "r") as f:
    embedding_dict_loaded = json.load(f)

embedding_dict = {eval(k): v for k, v in embedding_dict_loaded.items()}

def get_embeddings_from_env(config):
    env = gym.make(config['env_config']['env'])
    
    observation = env.reset()

    print(type(observation), observation.shape)

    # CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    text = desc_dict[config['env_config']['env']]
    

    if observation.max() > 1.0:
        observation = observation.astype(np.uint8)
    image = Image.fromarray(observation)
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_embedding = model.encode_image(image)
        text_embedding = model.encode_text(clip.tokenize([text], truncate=True).to(device))

    image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

    print("Image Embedding:", image_embedding, image_embedding.shape)
    print("Text Embedding:", text_embedding, text_embedding.shape)

    # image_embedding_random = random_encoder(output_dim=image_embedding.shape[1], input_shape=image_embedding.shape)
    # text_embedding_random = random_encoder(output_dim=image_embedding.shape[1], input_shape=image_embedding.shape)

    return image_embedding.cpu().numpy(), text_embedding.cpu().numpy()
    # return image_embedding_random, text_embedding_random

def find_similar_embedding(curr_image_embedding, curr_text_embedding, embedding_dict, threshold=0.9):
    best_weights_game = None
    highest_similarity = 0.0

    curr_image_embedding_flat = curr_image_embedding.flatten()
    curr_text_embedding_flat = curr_text_embedding.flatten()

    for key, game_name in embedding_dict.items():
        stored_image_embedding, stored_text_embedding = key
        
        image_similarity = cosine_similarity([curr_image_embedding_flat], [stored_image_embedding])[0][0]
        text_similarity = cosine_similarity([curr_text_embedding_flat], [stored_text_embedding])[0][0]

        combined_similarity = (image_similarity + text_similarity) / 2
        
        if combined_similarity >= threshold and combined_similarity > highest_similarity:
            highest_similarity = combined_similarity
            best_weights_game = game_name
    return best_weights_game


#Generic train fucntion that is used across all the below setups

#list of envs, what is the backbone, 
#No Sequential transfer. Single task on all envs.
def rllib_loop(config, str_logger):

    #final modifications in the config
    #if args.temporal == "lstm" or args.temporal == "attention":
    if args.env_name == "beogym":
        args.stop_timesteps = 20000000

    print("program running for, ", args.stop_timesteps)
    
    if args.env_name=='beogym':
        config['env_config']['data_path']=args.data_path
    
    print(config)

    if "multiagent" in config:
        algo = MultiPPO(config=config)
        print("Using MultiPPO")
    else:
        algo = PPO(config=config)

    plc = algo.get_policy()

    #Only train the backbone if backbone is e2e
    #get backbone and policy from setting.
    #you need to load the weights into the backbone or policy here!

    
    # if config['env_config']['env'] != configs.all_envs[0]:
    #     policy_ckpt = Policy.from_checkpoint(args.log + "/" + args.env_name + "/" + args.temporal + "/" + args.set + "/" + str_logger.replace(config['env_config']['env'] + '/', '') + "/checkpoint")
    #     plc.set_weights(policy_ckpt.get_weights())

    curr_image_embedding, curr_text_embedding = get_embeddings_from_env(config)

    similar_weights_game = find_similar_embedding(curr_image_embedding, curr_text_embedding, embedding_dict)

    curr_game_name = config['env_config']['env']
    if similar_weights_game is not None:
        print("Found similar embeddings, using previous policy weights.")
        # policy_ckpt = Policy.from_checkpoint(args.log + "/" + args.env_name + "/" + args.temporal + "/" + args.set + "/" + str_logger.replace(config['env_config']['env'] + '/', '') + "/checkpoint")
        curr_game_name = similar_weights_game
        policy_ckpt = Policy.from_checkpoint(args.ckpt + "/" + args.env_name + "/" + args.temporal + "/" + args.set + "/" + str_logger + '/' + curr_game_name + '/'+ "/checkpoint")
        plc.set_weights(policy_ckpt.get_weights())
        # plc.set_weights(similar_weights)
    else:
        print("No similar embeddings found, training new policy.")
        embedding_key = (tuple(curr_image_embedding.flatten()), tuple(curr_text_embedding.flatten()))
        embedding_dict[embedding_key] = config['env_config']['env']

    #if args.policy != None:
    #    backbone_ckpt = Policy.from_checkpoint('/lab/kiran/ckpts/trained/' + args.backbone).get_weights()
    #    for params in plc.keys():
            #load the policy
    #        if 'mlp' in params:
    #            plc[i] = res_wts[i]

    #load the backbone
    if args.backbone != 'e2e' and 'e2e' in args.backbone:
        embed()
        load_ckpt = Policy.from_checkpoint(args.log + "/" + args.temporal + "/" + args.env_name + "/" + args.set + "/" + args.backbone + "/checkpoint").get_weights()
        embed()
        orig_wts = plc.get_weights()
        chng_wts = {}
        for params in load_ckpt.keys():
            if 'logits' not in params and 'value' not in params:
                print(params)
                chng_wts[params] = load_ckpt[params]
            else:
                chng_wts[params] = orig_wts[params]
        plc.set_weights(chng_wts)


    # run manual training loop and print results after each iteration
    for _ in range(args.stop_timesteps):
        result = algo.train()
        #embed()
        # print(pretty_print(result))

        if curr_game_name not in mar_dict:
            mar_dict[curr_game_name] = []
            min_rewards_dict[curr_game_name] = None
            max_rewards_dict[curr_game_name] = None

        current_reward = result.get("episode_reward_mean", 0)
        mar_dict[curr_game_name].append(current_reward)

        if min_rewards_dict[curr_game_name] is None or max_rewards_dict[curr_game_name] is None:
            min_rewards_dict[curr_game_name] = current_reward
            max_rewards_dict[curr_game_name] = current_reward
        else:
            min_rewards_dict[curr_game_name] = min(min_rewards_dict[curr_game_name], current_reward)
            max_rewards_dict[curr_game_name] = max(max_rewards_dict[curr_game_name], current_reward)

        print(f'mar_dict: {mar_dict}')
        print(f'min_rewards_dict: {min_rewards_dict}')
        print(f'max_rewards_dict: {max_rewards_dict}')
        
        # stop training of the target train steps or reward are reached
        #MAKE SURE YOU KEEP SAVING CHECKPOINTS
        if result["timesteps_total"] >= args.stop_timesteps:
            print(args.temporal, args.set, str_logger.replace(config['env_config']['env'] + '/', ''))
            # algo.save(checkpoint_dir=args.log + "/" + args.env_name + "/" + args.temporal + "/" + args.set + "/" + str_logger.replace(config['env_config']['env'] + '/', '') + "/checkpoint/wholealgo")
            algo.save(checkpoint_dir=args.ckpt + "/" + args.env_name + "/" + args.temporal + "/" + args.set + "/" + str_logger + '/' + curr_game_name + '/'+ "/checkpoint/wholealgo")
            policy = algo.get_policy()
            # policy.export_checkpoint(args.log + "/" + args.env_name + "/" +  args.temporal + "/" + args.set + "/" + str_logger.replace(config['env_config']['env'] + '/', '') + "/checkpoint")
            policy.export_checkpoint(args.ckpt + "/" + args.env_name + "/" +  args.temporal + "/" + args.set + "/" + str_logger + '/' + curr_game_name + '/'+ "/checkpoint")
            break
    
    algo.stop()


#sequential learning
def seq_train(str_logger):

    print("SEQUENTIAL MODE!")
    #get the base atari_config to incorporate the environments
    #construct the base env class from envs.py based on the env_name
    use_config, use_env = pick_config_env('single')

    if args.backbone == "e2e":
        args.train_backbone = True

    #register the model
    if args.env_name == "atari":
        ModelCatalog.register_custom_model("model", SingleAtariModel)
    
    else:
        ModelCatalog.register_custom_model("model", SingleBeogymModel)
    
    print(configs.all_envs)

    #In the forloop base config and spec stays the same
    for eachenv in configs.all_envs:
        #in the for loop set the previous models weights
        #adapter, policy, backbone
        #env_config consists of which games we use
        use_config.update(
            {"env" : use_env, 
             "env_config" : {'env': eachenv, "full_action_space": True, 'framestack': args.temporal == '4stack'},
             "logger_config" : {
                "type": UnifiedLogger,
                "logdir": os.path.expanduser(args.log + '/' + args.env_name + '/' + args.set + '/'  + str_logger + "/" + eachenv + "/")
                }
            }
        )

        rllib_loop(use_config, str_logger)
    
    data_to_save = {
        "mar_list": mar_dict,
        "min_rewards": min_rewards_dict,
        "max_rewards": max_rewards_dict
    }

    with open("rewards_data.json", "w") as f:
        json.dump(data_to_save, f, indent=4)

    embedding_dict_str_keys = {str(k): v for k, v in embedding_dict.items()}
    with open("embedding_dict.json", "w") as f:
        json.dump(embedding_dict_str_keys, f, indent=4)

