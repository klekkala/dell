import argparse
import os
import gym
import yaml
import shutil
import random
import json
import torch

from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print, UnifiedLogger
from models.atarimodels import SingleAtariModel

import envs
import configs
from arguments import get_args

args = get_args()

def train_policy(game_list):
    use_config = configs.atari_config
    use_env = envs.atari['single']
    ModelCatalog.register_custom_model("model", SingleAtariModel)

    for eachenv in game_list: 
        use_config.update(
            {"env" : use_env, 
            "env_config" : {'env': eachenv, "full_action_space": True, 'framestack': args.temporal == '4stack'},
            "logger_config" : {
                "type": UnifiedLogger,
                "logdir": os.path.expanduser(args.log + '/' + args.env_name + '/' + args.set + '/'  + str_logger + "/" + eachenv + "/")
                }
            }
        )

        algo = PPO(config=use_config)

        env = use_env(env_config={'env': eachenv, 'full_action_space': True, 'framestack': args.temporal == '4stack'})
        obs = env.reset()

        plc = algo.get_policy()

        for _ in range(args.stop_timesteps):
            result = algo.train()
            print(pretty_print(result))

            if result["timesteps_total"] >= args.stop_timesteps:
                algo.save(checkpoint_dir="./ckpts/" + curr_game_name + "/wholealgo")
                policy = algo.get_policy()
                policy.export_checkpoint("./ckpts/" + curr_game_name + "/checkpoint")
                break

    algo.stop()
    torch.cuda.empty_cache()
