"""
The script to run ReLara algorithm.
"""

import argparse

from ReLara.Algorithms import ReLaraAlgo
from ReLara.Networks import BasicActor, ActorResidual, BasicQNetwork, QNetworkResidual
from ReLara.utils import robotics_env_maker, classic_control_env_maker


def parse_args():
    parser = argparse.ArgumentParser(description="Run ReLara on continuous control environments.")

    parser.add_argument("--exp-name", type=str, default="ReLara-rsa")

    parser.add_argument("--env-id", type=str, default="MyMujoco/Ant-Height-Sparse")
    parser.add_argument("--render", type=bool, default=False)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--proposed-reward-scale", type=float, default=1)
    parser.add_argument("--beta", type=float, default=0.2)

    parser.add_argument("--pa-buffer-size", type=int, default=1000000)
    parser.add_argument("--pa-rb-optimize-memory", type=bool, default=False)
    parser.add_argument("--pa-batch-size", type=int, default=256)

    parser.add_argument("--ra-buffer-size", type=int, default=1000000)
    parser.add_argument("--ra-rb-optimize-memory", type=bool, default=False)
    parser.add_argument("--ra-batch-size", type=int, default=256)

    parser.add_argument("--pa-actor-lr", type=float, default=3e-4)
    parser.add_argument("--pa-critic-lr", type=float, default=1e-3)
    parser.add_argument("--pa-alpha-lr", type=float, default=1e-4)

    parser.add_argument("--ra-actor-lr", type=float, default=3e-4)
    parser.add_argument("--ra-critic-lr", type=float, default=1e-3)
    parser.add_argument("--ra-alpha-lr", type=float, default=1e-4)

    parser.add_argument("--pa-policy-frequency", type=int, default=2)
    parser.add_argument("--pa-target-frequency", type=int, default=1)

    parser.add_argument("--ra-policy-frequency", type=int, default=2)
    parser.add_argument("--ra-target-frequency", type=int, default=1)

    parser.add_argument("--pa-tau", type=float, default=0.005)
    parser.add_argument("--ra-tau", type=float, default=0.005)

    parser.add_argument("--pa-alpha", type=float, default=0.2)
    parser.add_argument("--pa-alpha-autotune", type=bool, default=True)

    parser.add_argument("--ra-alpha", type=float, default=0.2)
    parser.add_argument("--ra-alpha-autotune", type=bool, default=True)

    parser.add_argument("--write-frequency", type=int, default=100)
    parser.add_argument("--save-frequency", type=int, default=10000)
    parser.add_argument("--save-folder", type=str, default="./ReLara-rsa/")

    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--pa-learning-starts", type=int, default=1e4)
    parser.add_argument("--ra-learning-starts", type=int, default=5e3)

    args = parser.parse_args()
    return args


def run():
    args = parse_args()

    env = robotics_env_maker(env_id=args.env_id, seed=args.seed, render=args.render) if args.env_id.startswith(
        "My") else classic_control_env_maker(env_id=args.env_id, seed=args.seed, render=args.render)

    agent = ReLaraAlgo(env=env,
                       pa_actor_class=BasicActor, pa_critic_class=BasicQNetwork,
                       ra_actor_class=ActorResidual, ra_critic_class=QNetworkResidual,
                       exp_name=args.exp_name, seed=args.seed, cuda=args.cuda, gamma=args.gamma,
                       proposed_reward_scale=args.proposed_reward_scale, beta=args.beta,
                       pa_buffer_size=args.pa_buffer_size, pa_rb_optimize_memory=args.pa_rb_optimize_memory,
                       pa_batch_size=args.pa_batch_size,
                       ra_buffer_size=args.ra_buffer_size, ra_rb_optimize_memory=args.ra_rb_optimize_memory,
                       ra_batch_size=args.ra_batch_size,
                       pa_actor_lr=args.pa_actor_lr, pa_critic_lr=args.pa_critic_lr, pa_alpha_lr=args.pa_alpha_lr,
                       ra_actor_lr=args.ra_actor_lr, ra_critic_lr=args.ra_critic_lr, ra_alpha_lr=args.ra_alpha_lr,
                       pa_policy_frequency=args.pa_policy_frequency, pa_target_frequency=args.pa_target_frequency,
                       ra_policy_frequency=args.ra_policy_frequency, ra_target_frequency=args.ra_target_frequency,
                       pa_tau=args.pa_tau, ra_tau=args.ra_tau,
                       pa_alpha=args.pa_alpha, pa_alpha_autotune=args.pa_alpha_autotune,
                       ra_alpha=args.ra_alpha, ra_alpha_autotune=args.ra_alpha_autotune,
                       write_frequency=args.write_frequency, save_frequency=args.save_frequency,
                       save_folder=args.save_folder)

    agent.learn(total_timesteps=args.total_timesteps,
                pa_learning_starts=args.pa_learning_starts, ra_learning_starts=args.ra_learning_starts)

    agent.save(indicator="final")


if __name__ == "__main__":
    run()
