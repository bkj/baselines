#!/usr/bin/env python

"""
    run_simple_atari.py
"""

import gym
import logging
import argparse
from mpi4py import MPI
from baselines import bench, logger
from baselines.common import set_global_seeds

import baselines.common.tf_util as U
from baselines.ppo1 import pposgd_simple, cnn_policy
from baselines.common.atari_wrappers import (wrap_deepmind, FrameStack)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-frames', type=int, default=int(40e6))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    
    workerseed = args.seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make(args.env)
    
    def policy_fn(name, ob_space, ac_space):
        return cnn_policy.CnnPolicy(
            name=name,
            ob_space=ob_space,
            ac_space=ac_space
        )
    
    env = bench.Monitor(env, "atari-%i.monitor.json" % rank)
    
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)
    
    env = FrameStack(wrap_deepmind(env, clip_rewards=True), 4)
    num_timesteps = int(args.num_frames / 4 * 1.1)
    env.seed(workerseed)
    
    pposgd_simple.learn(env,
        policy_fn,
        max_timesteps=num_timesteps,
        timesteps_per_batch=256,
        clip_param=0.2,
        entcoeff=0.01,
        optim_epochs=4,
        optim_stepsize=1e-3,
        optim_batchsize=64,
        gamma=0.99,
        lam=0.95,
        schedule='linear',
    )
    
    env.close()

