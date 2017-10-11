#!/usr/bin/env python

"""
    mlp_policy.py
"""

import gym
import time
import numpy as np
from collections import deque

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

# --
# Dataset class

class Dataset(object):
    def __init__(self, data_map, shuffle=True):
        self.data_map = data_map
        self.shuffle = shuffle
        self.n = next(iter(data_map.values())).size(0)
        self._next_id = 0
        self.do_shuffle()
        
    def do_shuffle(self):
        perm = torch.randperm(self.n)
        for key in self.data_map:
            self.data_map[key] = self.data_map[key][perm]
        
        self._next_id = 0
    
    def next_batch(self, batch_size):
        if self._next_id >= self.n and self.shuffle:
            self.do_shuffle()
            
        cur_id = self._next_id
        cur_batch_size = min(batch_size, self.n - self._next_id)
        self._next_id += cur_batch_size
        
        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][cur_id:cur_id+cur_batch_size]
        
        return data_map
        
    def iterate_once(self, batch_size):
        if self.shuffle:
            self.do_shuffle()
        
        while self._next_id <= self.n - batch_size:
            yield self.next_batch(batch_size)
        
        self._next_id = 0


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
        ac, vpred = pi.act(observation=ob, stochastic=stochastic)
        
        if t > 0 and t % horizon == 0:
            
            # Compute targets
            nextvpred = vpred * (1 - new)
            news_     = torch.cat([news, torch.zeros(1).long()])
            vpreds_   = torch.cat([vpreds, nextvpred])
            atargs    = torch.zeros(len(rews) + 1)
            
            for t in reversed(range(len(rews))):
                nonterminal = 1 - news_[t+1]
                delta = rews[t] + gamma * vpreds_[t+1] * nonterminal - vpreds_[t]
                atargs[t] = delta + gamma * lam * nonterminal * atargs[t+1]
            
            atargs = atargs[:-1]
            
            yield ({
                "obs"        : obs,
                "acs"        : acs,
                "atargs"     : (atargs - atargs.mean()) / atargs.std(),
                "vtargs"     : (atargs + vpreds),
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
        vpreds[i] = float(vpred[0])
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
# Probability distributions

class DiagGaussianPdType(object):
    def __init__(self, size):
        self.size = size
    
    def pdclass(self):
        return DiagGaussianPd
    
    def param_shape(self):
        return [2 * self.size]
    
    def sample_shape(self):
        return [self.size]
        
    def pdfromflat(self, flat):
        return self.pdclass()(flat)


class DiagGaussianPd(object):
    def __init__(self, flat):
        assert (flat.size()[0] % 2) == 0
        
        self.flat = flat
        tmp = self.flat.view(2, -1)
        self.mean = tmp[0]
        self.logstd = tmp[1]
        self.std = torch.exp(tmp[1])
    
    def mode(self):
        return self.mean
    
    # def entropy(self):
    #     return torch.sum(self.logstd + 0.5 * np.log(2.0 * np.pi * np.e), dim=-1)
    
    def sample(self):
        return self.mean + self.std * Variable(torch.randn(self.mean.size()))
    
    def logp(self, x):
        return -1 * (
            0.5 * (((x - self.mean) / self.std) ** 2).sum(dim=-1) + \
            0.5 * np.log(2.0 * np.pi) + \
            self.logstd.sum(dim=-1)
        )


class MlpPolicyTorch(nn.Module):
    """
        !! Fix initializations
        !! Running mean + stddev
        !! gaussian_fixed_var
    """
    
    recurrent = False
    
    def __init__(self, observation_space, action_space, hidden_dim, n_hidden,
        gaussian_fixed_var=False):
        
        super(MlpPolicyTorch, self).__init__()
        
        self.pdtype = DiagGaussianPdType(action_space.shape[0])
        
        # Define MLP value function
        value_fn = []
        for i in range(n_hidden):
            value_fn += [
                nn.Linear(observation_space.shape[0] if not i else hidden_dim, hidden_dim), # ... initialization ...
                nn.Tanh(),
            ]
        
        value_fn.append(nn.Linear(hidden_dim, 1))
        self.value_fn = nn.Sequential(*value_fn)
        
        # Define MLP policy function
        policy_fn = []
        for i in range(n_hidden):
            policy_fn += [
                nn.Linear(observation_space.shape[0] if not i else hidden_dim, hidden_dim), # ... initialization ...
                nn.Tanh(),
            ]
        
        if gaussian_fixed_var and isinstance(action_space, gym.spaces.Box):
            raise NotImplementedError
        else:
            policy_fn.append(nn.Linear(hidden_dim, self.pdtype.param_shape()[0]))
        
        self.policy_fn = nn.Sequential(*policy_fn)
        
    def act(self, observation, stochastic=True):
        # ... running mean + clip ...
        observation = Variable(observation).float()
        value_prediction = self.value_fn(observation)
        self.action_distribution = self.pdtype.pdfromflat(self.policy_fn(observation))
        
        if stochastic:
            action = self.action_distribution.sample()
        else:
            action = self.action_distribution.mode()
        
        return action.data, value_prediction.data



# --
# Params

seed = 0
env_name = 'Hopper-v1'
max_timesteps = int(1e6)
horizon = 2048
clip_param = 0.2
entcoeff = 0.0
optim_epochs = 10
optim_stepsize = 3e-4
optim_batchsize = 64
gamma = 0.99
lam = 0.95
schedule = 'linear'
adam_epsilon = 1e-5

# --
# Run

env = gym.make(env_name)
env.seed(seed)

pi, oldpi = [MlpPolicyTorch(
    env.observation_space,
    env.action_space,
    hidden_dim=64, 
    n_hidden=2
) for _ in range(2)]

opt = torch.optim.Adam(pi.parameters(), eps=adam_epsilon)

segment_generator = make_segment_generator(pi, env, horizon, stochastic=True, gamma=gamma, lam=lam)

episodes_so_far  = 0
timesteps_so_far = 0
iters_so_far     = 0

tstart = time.time()
lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards


def copy_net(source, target):
    target.load_state_dict(source.state_dict())
    target.action_distribution = source.action_distribution

while timesteps_so_far < max_timesteps:
    
    # Learning rate decay
    if schedule == 'constant':
        cur_lrmult = 1.0
    elif schedule == 'linear':
        cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
    else:
        raise NotImplementedError
    
    print('Iteration=%d' % iters_so_far)
    
    # Sample rollouts
    seg, meta = segment_generator.__next__()
    
    # Make dataset
    dataset = Dataset(seg, shuffle=True)
    
    # Set oldpi = pi
    copy_net(pi, oldpi)
    
    clip_param *= cur_lrmult
    for _ in range(optim_epochs):
        for batch in dataset.iterate_once(optim_batchsize or seg['obs'].shape[0]):
            opt.zero_grad()
            
            acs = Variable(batch['acs'])
            # atargs = Variable(batch['atargs'])
            
            ratio = torch.exp(
                pi.action_distribution.logp(acs) - oldpi.action_distribution.logp(acs)
            )
            # pol_surr = - torch.mean(
            #     torch.min(
            #         ratio * atargs,
            #         torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * atargs
            #     )
            # )
            
            total_loss = ratio.sum()
            
            grad = torch.autograd.grad(total_loss, pi.parameters())
            print(grad)
            # total_loss.backward()
            # opt.step()
            
            