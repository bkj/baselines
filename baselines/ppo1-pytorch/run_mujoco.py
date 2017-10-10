#!/usr/bin/env python

"""
    run_mujoco.py
"""

import sys
import gym
import time
import logging
import argparse
import numpy as np
import tensorflow as tf
from collections import deque

from baselines import bench, logger
import baselines.common.tf_util as U
from baselines.common.mpi_adam import MpiAdam
from baselines.common import set_global_seeds, Dataset, zipsame


import mlp_policy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Hopper-v1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser.parse_args()


# --
# Helpers

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


def make_segment_generator(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()
    
    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...
    
    # Initialize history arrays
    obs     = np.array([ob for _ in range(horizon)])
    rews    = np.zeros(horizon, 'float32')
    vpreds  = np.zeros(horizon, 'float32')
    news    = np.zeros(horizon, 'int32')
    acs     = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()
    
    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {
                "ob" : obs,
                "rew" : rews,
                "vpred" : vpreds,
                "new" : news,
                "ac" : acs,
                "prevac" : prevacs,
                "nextvpred" : vpred * (1 - new),
                "ep_rets" : ep_rets,
                "ep_lens" : ep_lens,
            }
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac
        
        ob, rew, new, _ = env.step(ac)
        rews[i] = rew
        
        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        
        t += 1

# --
# Run

if __name__ == '__main__':
    args = parse_args()
    
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env)
    
    def policy_func(name, ob_space, ac_space):
        return 
    
    env = bench.Monitor(env, "monitor.json")
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    
    # Set parameters
    max_timesteps=args.num_timesteps
    timesteps_per_batch=2048
    clip_param=0.2
    entcoeff=0.0
    optim_epochs=10
    optim_stepsize=3e-4
    optim_batchsize=64
    gamma=0.99
    lam=0.95
    schedule='linear'
    adam_epsilon=1e-5
    
    # Policy networks
    pi = mlp_policy.MlpPolicy(
        name='pi',
        ob_space=env.observation_space,
        ac_space=env.action_space,
        hid_size=64, 
        num_hid_layers=2
    )
    
    oldpi = mlp_policy.MlpPolicy(
        name='oldpi',
        ob_space=env.observation_space,
        ac_space=env.action_space,
        hid_size=64, 
        num_hid_layers=2
    )
    
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Observed return
    
    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epsilon
    
    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])
    
    # --
    # Define loss function
    
    # PPO's pessimistic surrogate (L^CLIP)
    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    pol_surr = - U.mean(
        tf.minimum(
            ratio * atarg, # surrogate from conservative policy iteration
            U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg, # clipped surrogate
        )
    )
    
    # Entropy penalty
    pol_entpen = (-entcoeff) * U.mean(pi.pd.entropy())
    
    # value function loss
    vf_loss = U.mean(tf.square(pi.vpred - ret))
    
    total_loss = pol_surr + pol_entpen + vf_loss
    
    # Wrap everything above in functions
    var_list = pi.get_trainable_variables()
    compute_grad = U.function([ob, ac, atarg, ret, lrmult], U.flatgrad(total_loss, var_list))
    
    # Optimizer
    opt = MpiAdam(var_list, epsilon=adam_epsilon)
    
    # Helper
    update_oldpi = U.function([],[], updates=[tf.assign(oldv, newv) 
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    
    U.initialize()
    opt.sync()
    
    # --
    # Initialize rollouts
    
    segment_generator = make_segment_generator(pi, env, timesteps_per_batch, stochastic=True)
    
    episodes_so_far  = 0
    timesteps_so_far = 0
    iters_so_far     = 0
    
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    
    # --
    # Run
    
    while timesteps_so_far < max_timesteps:
        
        # Learning rate decay
        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError
        
        logger.log("\n\n-- Iteration %i --"%iters_so_far)
        
        # Sample rollouts
        seg = segment_generator.__next__()
        
        # ?? Compute vtarg and atarg -- not sure what they are
        new, vpred, rew = seg['new'], seg['vpred'], seg['rew']
        new = np.append(new, 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
        vpred = np.append(vpred, seg["nextvpred"])
        atarg = np.zeros(len(rew) + 1, 'float32')
        
        for t in reversed(range(len(rew))):
            nonterminal = 1 - new[t+1]
            delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
            atarg[t] = delta + gamma * lam * nonterminal * atarg[t+1]
        
        vpred = vpred[:-1]
        atarg = atarg[:-1]
        
        # make dataset
        dataset = Dataset({
            "ob" : seg['ob'],
            "ac" : seg['ac'],
            "vtarg" : atarg + vpred,
            "atarg" : (atarg - atarg.mean()) / atarg.std(),
        }, shuffle=not pi.recurrent)
        
        # update running mean/std for policy, if applicable
        if hasattr(pi, "ob_rms"):
            pi.ob_rms.update(seg['ob'])
        
         # set oldpi = pi
        update_oldpi()
        
        # Optimize
        for _ in range(optim_epochs):
            for batch in dataset.iterate_once(optim_batchsize or seg['ob'].shape[0]):
                g = compute_grad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                opt.update(g, optim_stepsize * cur_lrmult) 
        
        # Tracking steps + logging
        episodes_so_far += len(seg["ep_lens"])
        timesteps_so_far += sum(seg["ep_lens"])
        iters_so_far += 1
        
        lenbuffer.extend(seg["ep_lens"])
        rewbuffer.extend(seg["ep_rets"])
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(seg["ep_lens"]))
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        logger.dump_tabular()
    
    env.close()
