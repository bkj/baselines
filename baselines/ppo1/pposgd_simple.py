#!/usr/bin/env

"""
    pposgd_simple.py
"""

import time
import numpy as np
from mpi4py import MPI
import tensorflow as tf
from collections import deque

from baselines import logger
import baselines.common.tf_util as U
from baselines.common import Dataset, zipsame
from baselines.common.mpi_adam import MpiAdam

# --
# Helpers

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

# --
# Segment generator

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
                "nextvpred": vpred * (1 - new),
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

def learn(env, policy_func, *,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize, # optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0, # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant' # annealing for stepsize parameters (epsilon and adam)
    ):
    
    # --
    # Placeholders
    
    pi = policy_func("pi", env.observation_space, env.action_space) # Construct network for new policy
    oldpi = policy_func("oldpi", env.observation_space, env.action_space) # Network for old policy
    
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
    
    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0, max_seconds > 0]) == 1, "Only one time constraint permitted"
    
    # --
    # Run
    
    while True:
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
        
        # Learning rate
        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError
        
        logger.log("\n\n-- Iteration %i --"%iters_so_far)
        
        # Sample rollouts
        seg = segment_generator.__next__()
        
        # ?? Compute vtarg and atarg
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
        
        dataset = Dataset({
                "ob" : seg['ob'],
                "ac" : seg['ac'],
                "vtarg" : atarg + vpred,
                "atarg" : (atarg - atarg.mean()) / atarg.std(),
            }, 
            shuffle=not pi.recurrent
        )
        
        # update running mean/std for policy, if applicable
        if hasattr(pi, "ob_rms"):
            pi.ob_rms.update(seg['ob'])
        
         # oldpi = pi
        update_oldpi()
        
        # Optimize
        for _ in range(optim_epochs):
            for batch in dataset.iterate_once(optim_batchsize or seg['ob'].shape[0]):
                g = compute_grad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                opt.update(g, optim_stepsize * cur_lrmult) 
        
        # Keep track of number of episodes, steps, etc
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.dump_tabular()
