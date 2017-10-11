#!/usr/bin/env

"""
    pposgd_simple.py
"""

import time
import numpy as np
import tensorflow as tf
from collections import deque

import json
from hashlib import md5

from baselines import logger
import baselines.common.tf_util as U
from baselines.common import Dataset, zipsame
from baselines.common.mpi_adam import MpiAdam

import torch

# --
# Helpers

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def do_hash(x):
     return md5(str(x).encode()).hexdigest()

# --
# Segment generator

def make_segment_generator(pi, env, horizon, stochastic, gamma, lam):
    t = 0
    new = True # marks if we're on first timestep of an episode
    ac = torch.from_numpy(env.action_space.sample()) # not used, just so we have the datatype
    ob = torch.from_numpy(env.reset())
    
    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...
    
    # Initialize history arrays
    obs     = torch.zeros((horizon, ob.size(0)))
    rews    = torch.zeros(horizon)
    vpreds  = torch.zeros(horizon)
    news    = torch.zeros(horizon).long()
    acs     = torch.zeros((horizon, ac.size(0)))
    
    while True:
        ac, vpred = pi.act(stochastic, ob.numpy())
        ac = torch.from_numpy(ac)
        
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            
            # Compute targets
            nextvpred = vpred * (1 - new)
            news_     = torch.cat([news, torch.LongTensor([0])]) # last element is only used for last vtarg, but we already zeroed it if last new = 1
            vpreds_   = torch.cat([vpreds, torch.FloatTensor([nextvpred])])
            atargs    = torch.zeros(len(rews) + 1)
            
            for t in reversed(range(len(rews))):
                nonterminal = 1 - news_[t+1]
                delta = rews[t] + gamma * vpreds_[t+1] * nonterminal - vpreds_[t]
                atargs[t] = delta + gamma * lam * nonterminal * atargs[t+1]
            
            atargs = atargs[:-1]
            
            yield ({
                "obs"        : obs.numpy(),
                "acs"        : acs.numpy(),
                "atargs"     : ((atargs - atargs.mean()) / atargs.std()).numpy(),
                "vtargs"     : (atargs + vpreds).numpy(),
            }, {
                "ep_lens" : ep_lens,
                "ep_rets" : ep_rets,
            })
            
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        
        i = t % horizon
        obs[i]    = ob
        vpreds[i] = float(vpred)
        news[i]   = new
        acs[i]    = ac
        
        ob, rew, new, _ = env.step(ac.numpy())
        ob = torch.from_numpy(ob)
        rews[i] = rew
        
        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = torch.from_numpy(env.reset())
        
        t += 1

# --
# Run

def learn(env, policy_func, *,
        timesteps_per_batch, # timesteps per actor per update
        clip_param,
        entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs,
        optim_stepsize,
        optim_batchsize, # optimization hypers
        gamma,
        lam, # advantage estimation
        max_timesteps=0,
        max_episodes=0,
        max_iters=0,
        max_seconds=0, # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant' # annealing for stepsize parameters (epsilon and adam)
    ):
    
    # --
    # Placeholders
    
    pi = policy_func("pi", env.observation_space, env.action_space) # Construct network for new policy
    oldpi = policy_func("oldpi", env.observation_space, env.action_space) # Network for old policy
    
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    vtarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Observed return (value target)
    
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
    vf_loss = U.mean(tf.square(pi.vpred - vtarg))
    
    total_loss = pol_surr + pol_entpen + vf_loss
    
    # Wrap everything above in functions
    var_list = pi.get_trainable_variables()
    compute_grad = U.function([ob, ac, atarg, vtarg, lrmult], U.flatgrad(total_loss, var_list))
    
    # Optimizer
    opt = MpiAdam(var_list, epsilon=adam_epsilon)
    
    # Helper
    update_oldpi = U.function([],[], updates=[tf.assign(oldv, newv) 
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    
    U.initialize()
    opt.sync()
    
    # --
    # Initialize rollouts
    
    segment_generator = make_segment_generator(pi, env, timesteps_per_batch, stochastic=True, gamma=gamma, lam=lam)
    
    episodes_so_far  = 0
    timesteps_so_far = 0
    iters_so_far     = 0
    
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    
    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0, max_seconds > 0]) == 1, "Only one time constraint permitted"
    
    # --
    # Run
    
    while True:
        logger.log("\n\n-- Iteration %i --"%iters_so_far)
        
        if callback:
            callback(locals(), globals())
        
        # Stopping conditions
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break
        
        # Learning rate decay
        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError
        
        # Sample rollouts
        seg, meta = segment_generator.__next__()
        
        # make dataset
        dataset = Dataset(seg, shuffle=not pi.recurrent)
        
        print(do_hash((seg['acs'], seg['obs'])))
        
        # update running mean/std for policy, if applicable
        if hasattr(pi, "ob_rms"):
            pi.ob_rms.update(seg['obs'])
        
         # set oldpi = pi
        update_oldpi()
        
        # Optimize
        for _ in range(optim_epochs):
            for batch in dataset.iterate_once(optim_batchsize or seg['obs'].shape[0]):
                g = compute_grad(batch["obs"], batch["acs"], batch["atargs"], batch["vtargs"], cur_lrmult)
                opt.update(g, optim_stepsize * cur_lrmult) 
        
        # Tracking steps + logging
        episodes_so_far += len(meta['ep_lens'])
        timesteps_so_far += sum(meta['ep_lens'])
        iters_so_far += 1
        
        lenbuffer.extend(meta['ep_lens'])
        rewbuffer.extend(meta['ep_rets'])
        print("EpLenMean = %f" % np.mean(lenbuffer))
        print("EpRewMean = %f" % np.mean(rewbuffer))
        print("EpThisIter = %f" % len(meta['ep_lens']))
        print("EpisodesSoFar = %f" % episodes_so_far)
        print("TimestepsSoFar = %f" % timesteps_so_far)
        print("TimeElapsed = %f" % (time.time() - tstart))

