"""
Some utility functions.
"""

import gymnasium as gym
import numpy as np

from RLEnvs.MyFetchRobot import push, reach
from RLEnvs.MyMujoco import ant_v4, humanoid_v4, humanoidstandup_v4, walker2d_v4


def classic_control_env_maker(env_id, seed=1, render=False):
    """
    Make the environment.
    :param env_id: the name of the environment
    :param seed: the random seed
    :param render: whether to render the environment
    :return: the environment
    """
    env = gym.make(env_id) if not render else gym.make(env_id, render_mode="human")

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    env = gym.wrappers.RecordEpisodeStatistics(env)

    return env


def robotics_env_maker(env_id, seed=1, render=False):
    env = gym.make(env_id) if not render else gym.make(env_id, render_mode="human")

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # + transform the reward from {-1, 0} to {0, 1}
    env = gym.wrappers.TransformReward(env, lambda reward: reward + 1.0)
    # + flatten the dict observation space to a vector
    env = gym.wrappers.TransformObservation(env, lambda obs: np.concatenate(
        [obs["observation"], obs["achieved_goal"], obs["desired_goal"]]))

    new_obs_length = env.observation_space['observation'].shape[0] + env.observation_space['achieved_goal'].shape[0] + \
                     env.observation_space['desired_goal'].shape[0]

    # redefine the observation of the environment, make it the same size of the flattened dict observation space
    env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(new_obs_length,),
                                           dtype=np.float32)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    return env
