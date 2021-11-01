# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:55:30 2020

@author: nastavirs
"""
import tensorflow as tf
import numpy as np
def net_f_NS(self, x, y, z, t):

        Re = 1

        u_v_w_p = self.neural_net(tf.concat([x, y, z, t], 1), self.weights, self.biases)
        u = u_v_w_p[:, 0:1]
        v = u_v_w_p[:, 1:2]
        w = u_v_w_p[:, 2:3]
        p = u_v_w_p[:, 3:4]

        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_z = tf.gradients(u, z)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        u_zz = tf.gradients(u_z, z)[0]

        v_t = tf.gradients(v, t)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_z = tf.gradients(v, z)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]
        v_zz = tf.gradients(v_z, z)[0]

        w_t = tf.gradients(w, t)[0]
        w_x = tf.gradients(w, x)[0]
        w_y = tf.gradients(w, y)[0]
        w_z = tf.gradients(w, z)[0]
        w_xx = tf.gradients(w_x, x)[0]
        w_yy = tf.gradients(w_y, y)[0]
        w_zz = tf.gradients(w_z, z)[0]

        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]
        p_z = tf.gradients(p, z)[0]

        f_u = u_t + (u * u_x + v * u_y + w * u_z) + p_x - 1/Re * (u_xx + u_yy + u_zz)
        f_v = v_t + (u * v_x + v * v_y + w * v_z) + p_y - 1/Re * (v_xx + v_yy + v_zz)
        f_w = w_t + (u * w_x + v * w_y + w * w_z) + p_z - 1/Re * (w_xx + w_yy + w_zz)
        f_e = u_x + v_y + w_z

        return u, v, w, p, f_u, f_v, f_w, f_e