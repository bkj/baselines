#!/usr/bin/env python

import argparse
from mpi4py import MPI

import gym, logging
from baselines import logger, bench
from baselines.common import set_global_seeds

def wrap_train(env):
    from baselines.common.atari_wrappers import (wrap_deepmind, FrameStack)
    env = wrap_deepmind(env, clip_rewards=True)
    env = FrameStack(env, 4)
    return env

def train(env_id, num_frames, seed):
    from baselines.ppo1 import pposgd_simple, cnn_policy
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space)
        
    env = bench.Monitor(env, "atari-monitor.json")
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)
    
    env = wrap_train(env)
    num_timesteps = int(num_frames / 4 * 1.1)
    env.seed(workerseed)
    
    pposgd_simple.learn(
        env,
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
        schedule='linear'
    )
    env.close()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='PongNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train(args.env, num_frames=40e6, seed=args.seed)
