# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:56:32 2020

@author: nastavirs
"""
import tensorflow as tf
import numpy as np
import time
np.random.seed(1234)
tf.set_random_seed(1234)
def BFGS_train(self):

        tf_dict = {self.x_ini_tf: self.xi, self.y_ini_tf: self.yi, self.z_ini_tf: self.zi, self.t_ini_tf: self.ti,
                   self.u_ini_tf: self.ui, self.v_ini_tf: self.vi, self.w_ini_tf: self.wi,
                   self.x_boundary_tf: self.xb, self.y_boundary_tf: self.yb, self.z_boundary_tf: self.zb,
                   self.t_boundary_tf: self.tb, self.u_boundary_tf: self.ub, self.v_boundary_tf: self.vb,
                   self.w_boundary_tf: self.wb, self.x_tf: self.x, self.y_tf: self.y, self.z_tf: self.z,
                   self.t_tf: self.t}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)
    
    # mini-batch to be implemented
    # def train(self, epoch=10, nIter=150, learning_rate=1e-3):
    #
    #     for ep in range(epoch):
    #
    #         batch_size1 = len(self.x0) // nIter
    #         batch_size2 = len(self.xb) // nIter
    #         batch_size3 = len(self.x) // nIter
    #
    #         arr1 = np.arange(batch_size1 * nIter)
    #         arr2 = np.arange(batch_size2 * nIter)
    #         arr3 = np.arange(batch_size3 * nIter)
    #
    #         permu1 = np.random.permutation(arr1).reshape((nIter, batch_size1))
    #         permu2 = np.random.permutation(arr2).reshape((nIter, batch_size2))
    #         permu3 = np.random.permutation(arr3).reshape((nIter, batch_size3))
    #
    #         start_time = time.time()
    #         for it in range(nIter):
    #             tf_dict = {self.x_ini_tf: self.x0[permu1[it, :], :],
    #                        self.y_ini_tf: self.y0[permu1[it, :], :],
    #                        self.z_ini_tf: self.z0[permu1[it, :], :],
    #                        self.t_ini_tf: self.t0[permu1[it, :], :],
    #                        self.u_ini_tf: self.u0[permu1[it, :], :],
    #                        self.v_ini_tf: self.v0[permu1[it, :], :],
    #                        self.w_ini_tf: self.w0[permu1[it, :], :],
    #                        self.x_boundary_tf: self.xb[permu2[it, :], :],
    #                        self.y_boundary_tf: self.yb[permu2[it, :], :],
    #                        self.z_boundary_tf: self.zb[permu2[it, :], :],
    #                        self.t_boundary_tf: self.tb[permu2[it, :], :],
    #                        self.u_boundary_tf: self.ub[permu2[it, :], :],
    #                        self.v_boundary_tf: self.vb[permu2[it, :], :],
    #                        self.w_boundary_tf: self.wb[permu2[it, :], :],
    #                        self.x_tf: self.x[permu3[it, :], :],
    #                        self.y_tf: self.y[permu3[it, :], :],
    #                        self.z_tf: self.z[permu3[it, :], :],
    #                        self.t_tf: self.t[permu3[it, :], :],
    #                        self.learning_rate: learning_rate}
    #
    #             self.sess.run(self.train_op_Adam, tf_dict)
    #
    #             # Print
    #             if it % 10 == 0:
    #                 elapsed = time.time() - start_time
    #                 loss_value = self.sess.run(self.loss, tf_dict)
    #                 print('epoch: %d, It: %d, Loss: %.3e, Time: %.2f' %
    #                       (ep, it, loss_value, elapsed))
    #                 start_time = time.time()
    #
    #     self.optimizer.minimize(self.sess,
    #                             feed_dict=tf_dict,
    #                             fetches=[self.loss],
    #                             loss_callback=self.callback)

    