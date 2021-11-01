# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:54:56 2020

@author: nastavirs
"""
import tensorflow as tf
import numpy as np
np.random.seed(1234)
tf.set_random_seed(1234)
def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)