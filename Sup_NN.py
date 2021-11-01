# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:55:15 2020

@author: nastavirs
"""
import tensorflow as tf
import numpy as np
np.random.seed(1234)
tf.set_random_seed(1234)
def net_NS(self, x, y, z, t):

        u_v_w_p = self.neural_net(tf.concat([x, y, z, t], 1), self.weights, self.biases)
        u = u_v_w_p[:, 0:1]
        v = u_v_w_p[:, 1:2]
        w = u_v_w_p[:, 2:3]
        p = u_v_w_p[:, 3:4]
        return u, v, w, p