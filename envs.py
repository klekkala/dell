import gym
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.env import BaseEnv
from typing import Dict, Tuple
from ray.rllib.policy.policy import Policy
from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
from ray import air, tune
import numpy as np
import cv2
import random
import string
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env.wrappers.atari_wrappers import FrameStack, ScaledFloatFrame, WarpFrame, NoopResetEnv, MonitorEnv, MaxAndSkipEnv, FireResetEnv
import ray
from IPython import embed
#import graph_tool.all as gt
from ray import air, tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from gym import spaces
from ray.rllib.utils.images import rgb2gray, resize

#from beogym.beogym import BeoGym
##SingleTask, MultiTask, MultiEnv classes and their related classes/functions


def wrap_custom(env, dim=84, framestack=True):
    """Configure environment for DeepMind-style Atari.

    Note that we assume reward clipping is done outside the wrapper.

    Args:
        env: The env object to wrap.
        dim: Dimension to resize observations to (dim x dim).
        framestack: Whether to framestack observations.
    """
    env = MonitorEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    if env.spec is not None and "NoFrameskip" in env.spec.id:
        env = MaxAndSkipEnv(env, skip=4)

    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    
    env = WarpFrame(env, dim)

    #env = ScaledFloatFrame(env)  # TODO: use for dqn?
    # env = ClipRewardEnv(env)  # reward clipping is handled by policy eval
    # 4x image framestacking.
    if framestack is True:
        env = FrameStack(env, 4)
    else:
        env = FrameStack(env, 1)
    return env



from ray.rllib.utils.annotations import override

atari_rewards={"AirRaidNoFrameskip-v4": 8000, "AssaultNoFrameskip-v4": 883,"BeamRiderNoFrameskip-v4": 1400, "CarnivalNoFrameskip-v4": 4384,"DemonAttackNoFrameskip-v4": 415, "NameThisGameNoFrameskip-v4": 6000,"PhoenixNoFrameskip-v4":4900,"RiverraidNoFrameskip-v4": 8400,"SpaceInvadersNoFrameskip-v4":500}
atari_envs = ["AirRaidNoFrameskip-v4", "AssaultNoFrameskip-v4", "BeamRiderNoFrameskip-v4", "CarnivalNoFrameskip-v4", "DemonAttackNoFrameskip-v4", "NameThisGameNoFrameskip-v4", "PhoenixNoFrameskip-v4", "RiverraidNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4"]



class MultiCallbacks(DefaultCallbacks):
    def on_episode_end(
    self,
    *,
    worker: RolloutWorker,
    base_env: BaseEnv,
    policies: Dict[str, Policy],
    episode: Episode,
    env_index: int,
    **kwargs
    ):
        env_keys = list(episode.agent_rewards.keys())
        for each_id in range(len(env_keys)):
            episode.custom_metrics[base_env.envs[0].envs[env_keys[each_id][0]]] = episode.agent_rewards[(env_keys[each_id][0], env_keys[each_id][1])]

from PIL import Image

class SingleAtariEnv(gym.Env):
    def __init__(self, env_config):

        self.env = wrap_custom(gym.make(env_config['env'], full_action_space=env_config['full_action_space']), framestack=env_config['framestack'])

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    """
    def step(self, action):
        res = ''.join(random.choices(string.ascii_lowercase +
                             string.digits, k=7))
        ab = self.env.step(action)
        obs = ab[0][:,:,0]
        im = Image.fromarray(obs)
        im.save("/lab/kiran/beamrider_rllib_imgs/" + res + ".png")
        return ab
    """




atari = {'single': SingleAtariEnv}






