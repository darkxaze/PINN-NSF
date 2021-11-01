# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:56:19 2020

@author: nastavirs
"""
import tensorflow as tf
import numpy as np
import time
np.random.seed(1234)
tf.set_random_seed(1234)
def Adam_train(self, nIter=5000, learning_rate=1e-3):

        tf_dict = {self.x_ini_tf: self.xi, self.y_ini_tf: self.yi, self.z_ini_tf: self.zi, self.t_ini_tf: self.ti,
                   self.u_ini_tf: self.ui, self.v_ini_tf: self.vi, self.w_ini_tf: self.wi,
                   self.x_boundary_tf: self.xb, self.y_boundary_tf: self.yb, self.z_boundary_tf: self.zb,
                   self.t_boundary_tf: self.tb, self.u_boundary_tf: self.ub, self.v_boundary_tf: self.vb,
                   self.w_boundary_tf: self.wb, self.x_tf: self.x, self.y_tf: self.y, self.z_tf: self.z,
                   self.t_tf: self.t, self.learning_rate: learning_rate}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)