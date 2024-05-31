"""
The Reinforcement Learning with An Assistant Reward Agent (ReLara) algorithm.
"""

import numpy as np

import gymnasium as gym

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.buffers import ReplayBuffer

import os
import random
import datetime
import time


class ReLaraAlgo:
    """
    The ReLara algorithm for SINGLE environment with Reward Agent to learn the reward function R(s,a).
    """

    def __init__(self, env, pa_actor_class, pa_critic_class, ra_actor_class, ra_critic_class, exp_name="ReLara",
                 seed=1, cuda=0, gamma=0.99, proposed_reward_scale=1, beta=0.2, pa_buffer_size=int(1e6),
                 pa_rb_optimize_memory=False, ra_buffer_size=int(1e6), ra_rb_optimize_memory=False, pa_batch_size=256,
                 ra_batch_size=256, pa_actor_lr=3e-4, pa_critic_lr=1e-3, pa_alpha_lr=1e-4, ra_actor_lr=3e-4,
                 ra_critic_lr=3e-4, ra_alpha_lr=1e-4, pa_policy_frequency=2, pa_target_frequency=1,
                 ra_policy_frequency=2, ra_target_frequency=1, pa_tau=0.005, ra_tau=0.005, pa_alpha=0.2,
                 pa_alpha_autotune=True, ra_alpha=0.2, ra_alpha_autotune=True, write_frequency=100,
                 save_frequency=100000, save_folder="./ReLara/"):
        """
        Initialize the ReLara algorithm.
        :param env: the environment
        :param pa_actor_class: the actor class of PA
        :param pa_critic_class: the critic class of PA
        :param ra_actor_class: the actor class of RA
        :param ra_critic_class: the critic class of RA
        :param exp_name: the name of the experiment
        :param seed: the random seed
        :param cuda: the cuda device
        :param gamma: the discount factor
        :param proposed_reward_scale: the scale of the proposed reward
        :param beta: the weight of the proposed reward
        :param pa_buffer_size: the buffer size of PA
        :param pa_rb_optimize_memory: whether to optimize the memory of PA
        :param ra_buffer_size: the buffer size of RA
        :param ra_rb_optimize_memory: whether to optimize the memory of RA
        :param pa_batch_size: the batch size of PA
        :param ra_batch_size: the batch size of RA
        :param pa_actor_lr: the learning rate of the actor of PA
        :param pa_critic_lr: the learning rate of the critic of PA
        :param pa_alpha_lr: the learning rate of the alpha of PA
        :param ra_actor_lr: the learning rate of the actor of RA
        :param ra_critic_lr: the learning rate of the critic of RA
        :param ra_alpha_lr: the learning rate of the alpha of RA
        :param pa_policy_frequency: the policy frequency of PA
        :param pa_target_frequency: the target frequency of PA
        :param ra_policy_frequency: the policy frequency of RA
        :param ra_target_frequency: the target frequency of RA
        :param pa_tau: the tau of PA
        :param ra_tau: the tau of RA
        :param pa_alpha: the alpha of PA
        :param pa_alpha_autotune: whether to autotune the alpha of PA
        :param ra_alpha: the alpha of RA
        :param ra_alpha_autotune: whether to autotune the alpha of RA
        :param write_frequency: the frequency to write the tensorboard
        :param save_frequency: the frequency to save the checkpoint
        :param save_folder: the folder to save the model
        """

        # init the basic parts of ReLara algorithm
        self.exp_name = exp_name

        # set the random seeds
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

        self.env = env

        # + create the proposed reward space
        self.proposed_reward_space = gym.spaces.Box(low=-proposed_reward_scale, high=proposed_reward_scale,
                                                    shape=(1,), dtype=np.float32, seed=seed)

        # + create the observation space for the reward agent
        self.ra_obs_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(env.observation_space.shape[0] + env.action_space.shape[0],),
                                           dtype=np.float32, seed=seed)

        self.beta = beta
        self.gamma = gamma

        # for the tensorboard writer
        run_name = "{}-{}-{}-{}".format(exp_name, env.unwrapped.spec.id, seed,
                                        datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S'))
        os.makedirs("./runs/", exist_ok=True)
        self.writer = SummaryWriter(os.path.join("./runs/", run_name))
        self.write_frequency = write_frequency

        self.save_folder = save_folder
        self.save_frequency = save_frequency
        os.makedirs(self.save_folder, exist_ok=True)

        # initialize the networks for PA:
        self.pa_actor = pa_actor_class(env.observation_space, env.action_space).to(self.device)
        self.pa_qf_1 = pa_critic_class(env.observation_space, env.action_space).to(self.device)
        self.pa_qf_2 = pa_critic_class(env.observation_space, env.action_space).to(self.device)
        self.pa_qf_1_target = pa_critic_class(env.observation_space, env.action_space).to(self.device)
        self.pa_qf_2_target = pa_critic_class(env.observation_space, env.action_space).to(self.device)

        self.pa_qf_1_target.load_state_dict(self.pa_qf_1.state_dict())
        self.pa_qf_2_target.load_state_dict(self.pa_qf_2.state_dict())

        # initialize the optimizers for PA:
        self.pa_actor_optimizer = optim.Adam(self.pa_actor.parameters(), lr=pa_actor_lr)
        self.pa_critic_optimizer = optim.Adam(list(self.pa_qf_1.parameters()) + list(self.pa_qf_2.parameters()),
                                              lr=pa_critic_lr)

        # initialize the alpha for PA:
        self.pa_alpha_autotune = pa_alpha_autotune
        if pa_alpha_autotune:
            # set the target entropy as the negative of the action space dimension
            self.pa_target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
            self.pa_log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.pa_alpha = self.pa_log_alpha.exp().item()
            self.pa_alpha_optimizer = optim.Adam([self.pa_log_alpha], lr=pa_alpha_lr)
        else:
            self.pa_alpha = pa_alpha

        # initialize the networks for RA:
        # + the input of actor will be (s, a) pair, output will be the proposed reward
        self.ra_actor = ra_actor_class(self.ra_obs_space, self.proposed_reward_space).to(self.device)
        # + the input of Q-critic will be [(s, a), r_p] pair, output will be the Q-value
        self.ra_qf_1 = ra_critic_class(self.ra_obs_space, self.proposed_reward_space).to(self.device)
        self.ra_qf_2 = ra_critic_class(self.ra_obs_space, self.proposed_reward_space).to(self.device)
        self.ra_qf_1_target = ra_critic_class(self.ra_obs_space, self.proposed_reward_space).to(self.device)
        self.ra_qf_2_target = ra_critic_class(self.ra_obs_space, self.proposed_reward_space).to(self.device)

        self.ra_qf_1_target.load_state_dict(self.ra_qf_1.state_dict())
        self.ra_qf_2_target.load_state_dict(self.ra_qf_2.state_dict())

        # initialize the optimizers for RA:
        self.ra_actor_optimizer = optim.Adam(self.ra_actor.parameters(), lr=ra_actor_lr)
        self.ra_critic_optimizer = optim.Adam(list(self.ra_qf_1.parameters()) + list(self.ra_qf_2.parameters()),
                                              lr=ra_critic_lr)

        # initialize the alpha for RA:
        self.ra_alpha_autotune = ra_alpha_autotune
        if ra_alpha_autotune:
            # set the target entropy as the negative of the action space dimension
            self.ra_target_entropy = -torch.prod(torch.Tensor(self.proposed_reward_space.shape).to(self.device)).item()
            self.ra_log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.ra_alpha = self.ra_log_alpha.exp().item()
            self.ra_alpha_optimizer = optim.Adam([self.ra_log_alpha], lr=ra_alpha_lr)
        else:
            self.ra_alpha = ra_alpha

        # initialize the replay buffers:
        # + modify the observation space to be float32
        self.env.observation_space.dtype = np.float32

        # + the replay buffer for PA stores <s_t, a_t, s_{t+1}, r_P_t, d_t>
        self.pa_replay_buffer = ReplayBuffer(
            pa_buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            optimize_memory_usage=pa_rb_optimize_memory,
            handle_timeout_termination=False
        )

        # + the replay buffer for RA stores <s_t, r_P_t, s_{t+1}, r_E_t, d_t>
        self.ra_replay_buffer = ReplayBuffer(
            ra_buffer_size,
            self.ra_obs_space,
            self.proposed_reward_space,
            self.device,
            optimize_memory_usage=ra_rb_optimize_memory,
            handle_timeout_termination=False
        )

        # initialize other parameters for PA:
        self.pa_batch_size = pa_batch_size
        self.pa_policy_frequency = pa_policy_frequency
        self.pa_target_frequency = pa_target_frequency
        self.pa_tau = pa_tau

        # initialize other parameters for RA:
        self.ra_batch_size = ra_batch_size
        self.ra_policy_frequency = ra_policy_frequency
        self.ra_target_frequency = ra_target_frequency
        self.ra_tau = ra_tau

    def learn(self, total_timesteps=int(1e6), pa_learning_starts=int(5e3), ra_learning_starts=int(5e3)):
        obs, _ = self.env.reset()

        for global_step in range(total_timesteps):

            # only at the first step, sample an action from the environment
            if global_step == 0:
                action = self.env.action_space.sample()

            # + to store transition into RA replay buffer, stack the observation and action as obs
            obs_ra = np.hstack((obs, action))

            if global_step < ra_learning_starts:
                reward_pro = self.proposed_reward_space.sample()
            else:
                # + get a proposed reward from the RA, stack the obs and action as obs
                reward_pro, _, _ = self.ra_actor.get_action(
                    torch.Tensor(np.expand_dims(obs_ra, axis=0)).to(self.device))
                reward_pro = reward_pro.detach().cpu().numpy()[0]

            next_obs, reward_env, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # + for the PA, only store the environmental reward
            self.pa_replay_buffer.add(obs, next_obs, action, reward_env, done, info)

            if not done:
                obs = next_obs

            else:
                # reset the environment
                obs, _ = self.env.reset()
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # + predict and store the next action
            if global_step < pa_learning_starts:
                action = self.env.action_space.sample()
            else:
                action, _, _ = self.pa_actor.get_action(torch.Tensor(np.expand_dims(obs, axis=0)).to(self.device))
                action = action.detach().cpu().numpy()[0]

            # + stack the next_observation and next_action as next_obs of RA
            next_obs_ra = np.hstack((obs, action))

            # + store the transition into RA replay buffer
            self.ra_replay_buffer.add(obs_ra, next_obs_ra, reward_pro, reward_env, done, info)

            # ALGO LOGIC: training.
            if global_step > pa_learning_starts:
                self.optimize_pa(global_step)

            if global_step > ra_learning_starts:
                self.optimize_ra(global_step)

            if (global_step + 1) % self.save_frequency == 0:
                self.save(indicator=f"{global_step / 1000}k")

        self.env.close()
        self.writer.close()

    def optimize_pa(self, global_step):
        """
        optimize the Policy Agent (RA) agent using SAC approach.
        """

        data = self.pa_replay_buffer.sample(self.pa_batch_size)

        # update the critic networks
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.pa_actor.get_action(data.next_observations)
            qf_1_next_target = self.pa_qf_1_target(data.next_observations, next_state_actions)
            qf_2_next_target = self.pa_qf_2_target(data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf_1_next_target, qf_2_next_target) - self.pa_alpha * next_state_log_pi

            # + compute the proposed reward by the RA
            # stack the obs and action with the dim=1
            obs_ra = torch.cat((data.observations, data.actions), dim=1)

            reward_pro, _, _ = self.ra_actor.get_action(obs_ra)

            next_q_value = data.rewards.flatten() + self.beta * reward_pro + \
                           (1 - data.dones.flatten()) * self.gamma * min_qf_next_target.view(-1)

        qf_1_a_values = self.pa_qf_1(data.observations, data.actions).view(-1)
        qf_2_a_values = self.pa_qf_2(data.observations, data.actions).view(-1)
        qf_1_loss = F.mse_loss(qf_1_a_values, next_q_value)
        qf_2_loss = F.mse_loss(qf_2_a_values, next_q_value)
        qf_loss = qf_1_loss + qf_2_loss

        self.pa_critic_optimizer.zero_grad()
        qf_loss.backward()
        self.pa_critic_optimizer.step()

        # update the actor networks
        if global_step % self.pa_policy_frequency == 0:
            for _ in range(self.pa_policy_frequency):
                pi, log_pi, _ = self.pa_actor.get_action(data.observations)
                qf_1_pi = self.pa_qf_1(data.observations, pi)
                qf_2_pi = self.pa_qf_2(data.observations, pi)
                min_qf_pi = torch.min(qf_1_pi, qf_2_pi)
                actor_loss = ((self.pa_alpha * log_pi) - min_qf_pi).mean()

                self.pa_actor_optimizer.zero_grad()
                actor_loss.backward()
                self.pa_actor_optimizer.step()

                if self.pa_alpha_autotune:
                    with torch.no_grad():
                        _, log_pi, _ = self.pa_actor.get_action(data.observations)
                    alpha_loss = (-self.pa_log_alpha.exp() * (log_pi + self.pa_target_entropy)).mean()

                    self.pa_alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.pa_alpha_optimizer.step()
                    self.pa_alpha = self.pa_log_alpha.exp().item()

        # update the target networks by softly copying the weights
        if global_step % self.pa_target_frequency == 0:
            for param, target_param in zip(self.pa_qf_1.parameters(), self.pa_qf_1_target.parameters()):
                target_param.data.copy_(self.pa_tau * param.data + (1 - self.pa_tau) * target_param.data)
            for param, target_param in zip(self.pa_qf_2.parameters(), self.pa_qf_2_target.parameters()):
                target_param.data.copy_(self.pa_tau * param.data + (1 - self.pa_tau) * target_param.data)

        if global_step % self.write_frequency == 0:
            self.writer.add_scalar("losses/pa_qf_1_values", qf_1_a_values.mean().item(), global_step)
            self.writer.add_scalar("losses/pa_qf_2_values", qf_2_a_values.mean().item(), global_step)
            self.writer.add_scalar("losses/pa_qf_1_loss", qf_1_loss.item(), global_step)
            self.writer.add_scalar("losses/pa_qf_2_loss", qf_2_loss.item(), global_step)
            self.writer.add_scalar("losses/pa_qf_loss", qf_loss.item() / 2.0, global_step)
            self.writer.add_scalar("losses/pa_actor_loss", actor_loss.item(), global_step)
            self.writer.add_scalar("losses/pa_alpha", self.pa_alpha, global_step)
            if self.pa_alpha_autotune:
                self.writer.add_scalar("losses/pa_alpha_loss", alpha_loss.item(), global_step)

    def optimize_ra(self, global_step):
        """
        optimize the Reward Agent (RA) using SAC approach.
        """

        data = self.ra_replay_buffer.sample(self.ra_batch_size)

        # update the critic networks
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.ra_actor.get_action(data.next_observations)
            qf_1_next_target = self.ra_qf_1_target(data.next_observations, next_state_actions)
            qf_2_next_target = self.ra_qf_2_target(data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf_1_next_target, qf_2_next_target) - self.ra_alpha * next_state_log_pi
            next_q_value = data.rewards.flatten() + \
                           (1 - data.dones.flatten()) * self.gamma * min_qf_next_target.view(-1)

        qf_1_a_values = self.ra_qf_1(data.observations, data.actions).view(-1)
        qf_2_a_values = self.ra_qf_2(data.observations, data.actions).view(-1)
        qf_1_loss = F.mse_loss(qf_1_a_values, next_q_value)
        qf_2_loss = F.mse_loss(qf_2_a_values, next_q_value)
        qf_loss = qf_1_loss + qf_2_loss

        self.ra_critic_optimizer.zero_grad()
        qf_loss.backward()
        self.ra_critic_optimizer.step()

        # update the actor networks
        if global_step % self.ra_policy_frequency == 0:
            for _ in range(self.ra_policy_frequency):
                pi, log_pi, _ = self.ra_actor.get_action(data.observations)
                qf_1_pi = self.ra_qf_1(data.observations, pi)
                qf_2_pi = self.ra_qf_2(data.observations, pi)
                min_qf_pi = torch.min(qf_1_pi, qf_2_pi)
                actor_loss = ((self.ra_alpha * log_pi) - min_qf_pi).mean()

                self.ra_actor_optimizer.zero_grad()
                actor_loss.backward()
                self.ra_actor_optimizer.step()

                if self.ra_alpha_autotune:
                    with torch.no_grad():
                        _, log_pi, _ = self.ra_actor.get_action(data.observations)
                    alpha_loss = (-self.ra_log_alpha.exp() * (log_pi + self.ra_target_entropy)).mean()

                    self.ra_alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.ra_alpha_optimizer.step()
                    self.ra_alpha = self.ra_log_alpha.exp().item()

        # update the target networks by softly copying the weights
        if global_step % self.ra_target_frequency == 0:
            for param, target_param in zip(self.ra_qf_1.parameters(), self.ra_qf_1_target.parameters()):
                target_param.data.copy_(self.ra_tau * param.data + (1 - self.ra_tau) * target_param.data)
            for param, target_param in zip(self.ra_qf_2.parameters(), self.ra_qf_2_target.parameters()):
                target_param.data.copy_(self.ra_tau * param.data + (1 - self.ra_tau) * target_param.data)

        if global_step % self.write_frequency == 0:
            self.writer.add_scalar("losses/ra_qf_1_values", qf_1_a_values.mean().item(), global_step)
            self.writer.add_scalar("losses/ra_qf_2_values", qf_2_a_values.mean().item(), global_step)
            self.writer.add_scalar("losses/ra_qf_1_loss", qf_1_loss.item(), global_step)
            self.writer.add_scalar("losses/ra_qf_2_loss", qf_2_loss.item(), global_step)
            self.writer.add_scalar("losses/ra_qf_loss", qf_loss.item() / 2.0, global_step)
            self.writer.add_scalar("losses/ra_actor_loss", actor_loss.item(), global_step)
            self.writer.add_scalar("losses/ra_alpha", self.ra_alpha, global_step)
            if self.ra_alpha_autotune:
                self.writer.add_scalar("losses/ra_alpha_loss", alpha_loss.item(), global_step)

    def save(self, indicator="final"):

        torch.save(self.pa_actor.state_dict(),
                   os.path.join(self.save_folder,
                                "pa-actor-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed)))

        torch.save(self.ra_actor.state_dict(),
                   os.path.join(self.save_folder,
                                "ra-actor-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed)))
