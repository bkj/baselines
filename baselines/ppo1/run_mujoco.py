#!/usr/bin/env python

"""
    run_mujoco.py
"""

import sys
import gym
import logging
import argparse

from baselines import bench, logger
from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1 import mlp_policy, pposgd_simple

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Hopper-v1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env)
    
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(
            name=name,
            ob_space=ob_space,
            ac_space=ac_space,
            hid_size=64, 
            num_hid_layers=2
        )
    
    env = bench.Monitor(env, "monitor.json")
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    
    pposgd_simple.learn(
        env,
        policy_fn, 
        max_timesteps=args.num_timesteps,
        timesteps_per_batch=2048,
        clip_param=0.2,
        entcoeff=0.0,
        optim_epochs=10,
        optim_stepsize=3e-4,
        optim_batchsize=64,
        gamma=0.99,
        lam=0.95,
        schedule='linear',
    )
    
    env.close()
