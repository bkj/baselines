#!/usr/bin/env python

"""
    mlp_policy.py
"""

import gym
import numpy as np
import tensorflow as tf
import baselines.common.tf_util as U

from baselines.common.mpi_running_mean_std import RunningMeanStd

# --
# Probability distributions

class DiagGaussianPdType(object):
    def __init__(self, size):
        self.size = size
    
    def pdclass(self):
        return DiagGaussianPd
    
    def param_shape(self):
        return [2*self.size]
    
    def sample_shape(self):
        return [self.size]
    
    def sample_dtype(self):
        return tf.float32
        
    def pdfromflat(self, flat):
        return self.pdclass()(flat)
        
    def param_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape+self.param_shape(), name=name)
    
    def sample_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape+self.sample_shape(), name=name)


class DiagGaussianPd(object):
    def __init__(self, flat):
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape)-1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
    
    # @classmethod
    # def fromflat(cls, flat):
    #     return cls(flat)
    
    # def flatparam(self):
    #     return self.flat
    
    def mode(self):
        return self.mean
    
    # def kl(self, other):
    #     assert isinstance(other, DiagGaussianPd)
    #     return U.sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (2.0 * tf.square(other.std)) - 0.5, axis=-1)
    
    def entropy(self):
        return U.sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
    
    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))
    
    def logp(self, x):
        return -1 * (
            0.5 * U.sum(tf.square((x - self.mean) / self.std), axis=-1) + \
            0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) + \
            U.sum(self.logstd, axis=-1)
        )


# --
# Policy

class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name
            
    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)
        assert isinstance(ac_space, gym.spaces.Box)
        
        # Action space probability distribution
        self.pdtype = pdtype = DiagGaussianPdType(ac_space.shape[0])
        sequence_length = None
        
        # Observation
        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        
        # runmean of ob
        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)
        
        # % difference from runmean
        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        
        # Define MLP value function -- x_(i+1) = tanh(dense(x_i))
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        
        self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]
        
        # Define MLP policy function
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        
        # ??
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = U.dense(last_out, pdtype.param_shape()[0] // 2, "polfinal", U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0] // 2], initializer=tf.zeros_initializer())
            pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))
        
        # Convert weights back to "policy"
        self.pd = pdtype.pdfromflat(pdparam)
        
        # Appear not to be used
        self.state_in = []
        self.state_out = []
        
        # Whether or not to sample from policy or take maximum
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        
        # Run
        self._act = U.function([stochastic, ob], [ac, self.vpred])
        
    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    
    def get_initial_state(self):
        return []

