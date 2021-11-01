# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:56:40 2020

@author: nastavirs
"""
import numpy as np
import tensorflow as tf
def predictNS(self, x_star, y_star, z_star, t_star):
        
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.z_tf: z_star, self.t_tf: t_star}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        w_star = self.sess.run(self.w_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)
        
        return u_star, v_star, w_star, p_star