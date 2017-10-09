#!/usr/bin/env python

"""
    cnn_policy.py
"""

import gym
import tensorflow as tf
import baselines.common.tf_util as U

# --
# Probability distributions

class CategoricalPdType(object):
    def __init__(self, ncat):
        self.ncat = ncat
    
    def pdclass(self):
        return CategoricalPd
    
    def param_shape(self):
        return [self.ncat]
    
    def sample_shape(self):
        return []
    
    def sample_dtype(self):
        return tf.int32
    
    def pdfromflat(self, flat):
        return self.pdclass()(flat)
    
    def param_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape+self.param_shape(), name=name)
    
    def sample_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape+self.sample_shape(), name=name)


class CategoricalPd(object):
    def __init__(self, logits):
        self.logits = logits
    
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)
    
    def flatparam(self):
        return self.logits
    
    def mode(self):
        return U.argmax(self.logits, axis=-1)
    
    def kl(self, other):
        a0 = self.logits - U.max(self.logits, axis=-1, keepdims=True)
        a1 = other.logits - U.max(other.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = U.sum(ea0, axis=-1, keepdims=True)
        z1 = U.sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return U.sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)
    
    def entropy(self):
        a0 = self.logits - U.max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = U.sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return U.sum(p0 * (tf.log(z0) - a0), axis=-1)
    
    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits))
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)
    
    def logp(self, x):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return - tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits,
            labels=one_hot_actions
        )



class CnnPolicy(object):
    recurrent = False
    def __init__(self, name, ob_space, ac_space, kind='large'):
        with tf.variable_scope(name):
            self._init(ob_space, ac_space, kind)
            self.scope = tf.get_variable_scope().name
    
    def _init(self, ob_space, ac_space, kind):
        assert isinstance(ob_space, gym.spaces.Box)
        assert isinstance(ac_space, gym.spaces.Discrete)
        
        # Action space probability distribution
        self.pdtype = pdtype = CategoricalPdType(ac_space.n)
        sequence_length = None
        
        # Observation
        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        
        # Define policy function -- CNN w/ categorical outputs
        x = ob / 255.0
        if kind == 'small': # from A3C paper
            x = tf.nn.relu(U.conv2d(x, 16, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(U.dense(x, 256, 'lin', U.normc_initializer(1.0)))
        elif kind == 'large': # Nature DQN
            x = tf.nn.relu(U.conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(U.dense(x, 512, 'lin', U.normc_initializer(1.0)))
        else:
            raise NotImplementedError
            
        logits = U.dense(x, pdtype.param_shape()[0], "logits", U.normc_initializer(0.01))
        
        # Define CNN value function
        self.vpred = U.dense(x, 1, "value", U.normc_initializer(1.0))[:,0]
        
        # Convert weights back to "policy"
        self.pd = pdtype.pdfromflat(logits)
        
        # Appear not to be used
        self.state_in = []
        self.state_out = []
        
        # !! Always sample -- no deterministic mode
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample() # XXX
        
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
