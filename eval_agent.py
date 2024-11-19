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

import envs
import configs
import meta
from arguments import get_args
from models.atarimodels import SingleAtariModel

args = get_args()

# all_envs = ["AirRaidNoFrameskip-v4","AssaultNoFrameskip-v4","BeamRiderNoFrameskip-v4", "CarnivalNoFrameskip-v4","DemonAttackNoFrameskip-v4","NameThisGameNoFrameskip-v4","PhoenixNoFrameskip-v4","RiverraidNoFrameskip-v4","SpaceInvadersNoFrameskip-v4"]

def get_model_size():
    return round(random.uniform(50.0, 100.0), 2)

def get_buffer_size():
    return round(random.uniform(10.0, 50.0), 2) 

def generate_game_list(alpha, beta):
    selected_games = random.sample(configs.all_envs, alpha)

    result = selected_games.copy()

    while len(result) < beta:
        result.append(random.choice(selected_games))
    random.shuffle(result)

    return result

def create_resource_yaml(path, model_size, buffer_size):
    resource_data = {
        'model_size': f"{model_size} MB",
        'buffer_size': f"{buffer_size} KB"
    }
    yaml_file = os.path.join(path, 'resource.yaml')
    with open(yaml_file, 'w') as f:
        yaml.dump(resource_data, f)
    print(f"Created {yaml_file} with model and buffer sizes.")

def create_buffer_data_json(path):
    file_path = os.path.join(path, "buffer_data.json")

    empty_data = {}
    
    with open(file_path, 'w') as json_file:
        json.dump(empty_data, json_file, indent=4)
    print(f"JSON file created at: {file_path}")

def eval_agent(str_logger):
    # parser = argparse.ArgumentParser(description="Train games and manage resources")
    # parser.add_argument('--alpha', type=int, required=True, help='Number of unique games')
    # parser.add_argument('--beta', type=int, required=True, help='Total number of games')
    # parser.add_argument('--path', type=str, required=True, help='Path to save agent and resources folder')
    # parser.add_argument("--run", type=int, default=3, help="Number of Run of all games")

    # args = parser.parse_args()

    os.makedirs(args.path, exist_ok=True)
    
    model_size = get_model_size()
    buffer_size = get_buffer_size()

    create_resource_yaml(args.path, model_size, buffer_size)

    create_buffer_data_json(args.path)

    # create_checkpoints(args.path, args.beta)

    shutil.copy("meta.py", os.path.join(args.path, "meta.py"))

    game_list = generate_game_list(args.alpha, args.beta)
    
    print(game_list)

    use_config = configs.atari_config
    use_env = envs.atari['single']
    ModelCatalog.register_custom_model("model", SingleAtariModel)

    for i in range(args.run):
        epoch_rewards = {}
        cnt = 0
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
            
            # text, query_image = load_from_env(eachenv)

            done = False
            total_reward = 0
            is_need_compare = True

            while not done:
                action = meta.load_model(obs, eachenv, algo, env, is_need_compare)
                
                obs, reward, done, info = env.step(action)
                is_need_compare = False

                print(f"Action: {action}, Reward: {reward}")
                total_reward += reward

            epoch_rewards[f"{eachenv}:{cnt}"] = total_reward
            print(f"Total reward for {eachenv}: {total_reward}")
            cnt += 1

            algo.stop()
            torch.cuda.empty_cache()
        
        results_dir = "./epoch_results/"
        os.makedirs(results_dir, exist_ok=True)

        with open(os.path.join(results_dir, f"epoch_{i+1}_rewards.json"), 'w') as f:
            json.dump(epoch_rewards, f, indent=4)
        
        print(f"Epoch {i+1} rewards saved to {results_dir}")