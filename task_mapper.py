import ray
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.algorithms.ppo import PPO

# load mutiple policies
policy_paths = ["path_to_policy_1", "path_to_policy_2", ...]
policies = []

for path in policy_paths:
    algo = PPO(config=config)
    algo.restore(path)
    policies.append(algo.get_policy())

env = gym.make('Breakout-v0')

observation = env.reset()

def process_observation(obs):
    return obs


def select_policy(processed_obs):
    # random select a policy
    policy_index = np.random.choice(len(policies))

    input_dim = 3 
    feature_dim = 128
    h_dim = 64
    num_policies = len(policy_paths)
    model = PolicySelector(feature_dim=feature_dim, h_dim=h_dim, num_policies=num_policies)
    policy_logits = model(processed_obs)

    return policies[policy_index]


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 8 * 8, feature_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return x
    
class PolicySelector(nn.Module):
    def __init__(self, feature_dim, h_dim, num_policies):
        super(PolicySelector, self).__init__()
        self.feature_extractor = FeatureExtractor(input_dim=3, feature_dim=feature_dim)
        self.task_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, num_policies)  # num_policies
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        policy_logits = self.task_classifier(features)
        return policy_logits
    

total_reward = 0

while True:
    processed_obs = process_observation(observation)
    selected_policy = select_policy(processed_obs)
    action = selected_policy.compute_single_action(processed_obs)
    observation, reward, done, info = env.step(action)
    total_reward += reward

    if done:
        print(f"Total Reward: {total_reward}")
        observation = env.reset()
        total_reward = 0