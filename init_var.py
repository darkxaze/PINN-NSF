# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:54:01 2020

@author: nastavirs
"""

import tensorflow as tf
import numpy as np
import time
import scipy.io
np.random.seed(1234)
tf.set_random_seed(1234)
class NSPINN:
    # notational conventions
    # _tf: placeholders for input/output data and points used to regress the equations
    # _pred: output of neural network
    # _data: input-output data
    # _star: preditions
    from init_NN import initialize_NN
    from Xavi_init import xavier_init
    from NN import neural_net
    from Sup_NN import net_NS
    from Unsup_NN import net_f_NS
    from Callback import callback
    from Adam_train import Adam_train
    from BGFS_train import BFGS_train
    from predict import predict
    def __init__(self, xi, yi, zi, ti, ui, vi, wi, xb, yb, zb, tb, ub, vb, wb, x, y, z, t, layers):
        xyzt_i = np.concatenate([xi, yi, zi, ti], 1)  
        xyzt_b = np.concatenate([xb, yb, zb, tb], 1)
        xyzt = np.concatenate([x, y, z, t], 1)

        self.lowb = xyzt_b.min(0) 
        self.upb = xyzt_b.max(0)

        self.xyzt_i = xyzt_i
        self.xyzt_b = xyzt_b
        self.xyzt = xyzt

        self.xi = xyzt_i[:, 0:1]
        self.yi = xyzt_i[:, 1:2]
        self.zi = xyzt_i[:, 2:3]
        self.ti = xyzt_i[:, 3:4]

        self.xb = xyzt_b[:, 0:1]
        self.yb = xyzt_b[:, 1:2]
        self.zb = xyzt_b[:, 2:3]
        self.tb = xyzt_b[:, 3:4]

        self.x = xyzt[:, 0:1]
        self.y = xyzt[:, 1:2]
        self.z = xyzt[:, 2:3]
        self.t = xyzt[:, 3:4]

        self.ui = ui
        self.vi = vi
        self.wi = wi

        self.ub = ub
        self.vb = vb
        self.wb = wb

        self.layers = layers

        self.weights, self.biases = self.initialize_NN(layers)

        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_ini_tf = tf.placeholder(tf.float32, shape=[None, self.xi.shape[1]])
        self.y_ini_tf = tf.placeholder(tf.float32, shape=[None, self.yi.shape[1]])
        self.z_ini_tf = tf.placeholder(tf.float32, shape=[None, self.zi.shape[1]])
        self.t_ini_tf = tf.placeholder(tf.float32, shape=[None, self.ti.shape[1]])
        self.u_ini_tf = tf.placeholder(tf.float32, shape=[None, self.ui.shape[1]])
        self.v_ini_tf = tf.placeholder(tf.float32, shape=[None, self.vi.shape[1]])
        self.w_ini_tf = tf.placeholder(tf.float32, shape=[None, self.wi.shape[1]])

        self.x_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.xb.shape[1]])
        self.y_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.yb.shape[1]])
        self.z_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.zb.shape[1]])
        self.t_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.tb.shape[1]])
        self.u_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.ub.shape[1]])
        self.v_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.vb.shape[1]])
        self.w_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.wb.shape[1]])

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.z_tf = tf.placeholder(tf.float32, shape=[None, self.z.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])

        self.u_ini_pred, self.v_ini_pred, self.w_ini_pred, self.p_ini_pred = \
            self.net_NS(self.x_ini_tf, self.y_ini_tf, self.z_ini_tf, self.t_ini_tf)
        self.u_boundary_pred, self.v_boundary_pred, self.w_boundary_pred, self.p_boundary_pred = \
            self.net_NS(self.x_boundary_tf, self.y_boundary_tf, self.z_boundary_tf, self.t_boundary_tf)
        self.u_pred, self.v_pred, self.w_pred, self.p_pred, self.f_u_pred, self.f_v_pred, self.f_w_pred, self.f_e_pred = \
            self.net_f_NS(self.x_tf, self.y_tf, self.z_tf, self.t_tf)

        alpha = 100
        beta = 100

        self.loss = alpha * tf.reduce_mean(tf.square(self.u_ini_tf - self.u_ini_pred)) + \
                    alpha * tf.reduce_mean(tf.square(self.v_ini_tf - self.v_ini_pred)) + \
                    alpha * tf.reduce_mean(tf.square(self.w_ini_tf - self.w_ini_pred)) + \
                    beta * tf.reduce_mean(tf.square(self.u_boundary_tf - self.u_boundary_pred)) + \
                    beta * tf.reduce_mean(tf.square(self.v_boundary_tf - self.v_boundary_pred)) + \
                    beta * tf.reduce_mean(tf.square(self.w_boundary_tf - self.w_boundary_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_v_pred)) + \
                    tf.reduce_mean(tf.square(self.f_w_pred)) + \
                    tf.reduce_mean(tf.square(self.f_e_pred))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        
             
        
if __name__ == "__main__":
 
    N_train = 10000

    layers = [4, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 4]

    def data_generate(x, y, z, t):

        #a, d = 1, 1
        # m =   np.dot(np.exp(a*x),np.sin(a*y+d*z))+np.dot(np.exp(a*z),np.cos(a*x+d*y))
        # u = - a *np.dot(m,np.exp(-d*d*t))
        # n =   np.dot(np.exp(a*y),np.sin(a*z+d*x))+np.dot(np.exp(a*x),np.cos(a*y+d*z))
        # v = - a * np.dot(n,np.exp(-d*d*t))
        # o =   np.dot(np.exp(a*z),np.sin(a*x+d*y))+np.dot(np.exp(a*y),np.cos(a*z+d*x))
        # w = - a * np.dot(o,np.exp(-d*d*t))
        # k =   np.exp(2*a*x)+np.exp(2*a*y)+np.exp(2*a*z)+2*np.dot(np.dot(np.sin(a*x+d*y),np.cos(a*z+d*x)),np.exp(a*(y+z))) + 2 * np.dot(np.dot(np.sin(a*y+d*z),np.cos(a*x+d*y)),np.exp(a*(z+x))) + 2 * np.dot(np.dot(np.sin(a*z+d*x),np.cos(a*y+d*z)),np.exp(a*(x+y)))
        # p = - 0.5*a*a*np.dot(k,np.exp(-2*d*d*t))
        # a, d = 1, 1
        # u = - a* (np.exp(a * x) * np.sin(a * y + d * z) + np.exp(a * z) * np.cos(a * x + d * y))@np.transpose(np.exp(- d * d * t))
        # v = - a * (np.exp(a * y) * np.sin(a * z + d * x) + np.exp(a * x) * np.cos(a * y + d * z))@np.transpose(np.exp(- d * d * t))
        # w = - a * (np.exp(a * z) * np.sin(a * x + d * y) + np.exp(a * y) * np.cos(a * z + d * x))@np.transpose(np.exp(- d * d * t))
        # p = - 0.5 * a * a * (np.exp(2 * a * x) + np.exp(2 * a * y) + np.exp(2 * a * z) +
        #                      2 * np.sin(a * x + d * y) * np.cos(a * z + d * x) * np.exp(a * (y + z)) +
        #                      2 * np.sin(a * y + d * z) * np.cos(a * x + d * y) * np.exp(a * (z + x)) +
        #                      2 * np.sin(a * z + d * x) * np.cos(a * y + d * z) * np.exp(a * (x + y)))@np.transpose(np.exp(
        #     -2 * d * d * t))
        a, d = 1, 1
        u = - a * (np.exp(a * x) * np.sin(a * y + d * z) + np.exp(a * z) * np.cos(a * x + d * y)) * np.exp(- d * d * t)
        v = - a * (np.exp(a * y) * np.sin(a * z + d * x) + np.exp(a * x) * np.cos(a * y + d * z)) * np.exp(- d * d * t)
        w = - a * (np.exp(a * z) * np.sin(a * x + d * y) + np.exp(a * y) * np.cos(a * z + d * x)) * np.exp(- d * d * t)
        p = - 0.5 * a * a * (np.exp(2 * a * x) + np.exp(2 * a * y) + np.exp(2 * a * z) +
                             2 * np.sin(a * x + d * y) * np.cos(a * z + d * x) * np.exp(a * (y + z)) +
                             2 * np.sin(a * y + d * z) * np.cos(a * x + d * y) * np.exp(a * (z + x)) +
                             2 * np.sin(a * z + d * x) * np.cos(a * y + d * z) * np.exp(a * (x + y))) * np.exp(
            -2 * d * d * t)

        return u, v, w, p

    xdata = np.linspace(-1, 1, 31)
    ydata = np.linspace(-1, 1, 31)
    zdata = np.linspace(-1, 1, 31)
    tdata = np.linspace(0, 1, 11)
    b0 = np.array([-1] * 900)
    b1 = np.array([1] * 900)

   #boundary values
    xr1 = np.tile(xdata[0:30], 30)
    yr1 = np.tile(ydata[0:30], 30)
    zr1 = np.tile(zdata[0:30], 30)
    xr2 = np.tile(xdata[1:31], 30)
    yr2 = np.tile(ydata[1:31], 30)
    zc2 = np.tile(zdata[1:31], 30)

    xc1 = xdata[0:30].repeat(30)
    yc1 = ydata[0:30].repeat(30)
    zc1 = zdata[0:30].repeat(30)
    xc2 = xdata[1:31].repeat(30)
    yc2 = ydata[1:31].repeat(30)
    zr1 = zdata[1:31].repeat(30)

    trainx = np.concatenate([b1, b0, xr2, xr1, xr2, xr1], 0).repeat(tdata.shape[0])
    trainy = np.concatenate([yr1, yr2, b1, b0, yc2, yc1], 0).repeat(tdata.shape[0])
    trainz = np.concatenate([zc1, zr1, zc1, zr1, b1, b0], 0).repeat(tdata.shape[0])
    traint = np.tile(tdata, 5400)

    trainub, trainvb, trainwb, trainpb = data_generate(trainx, trainy, trainz, traint)

    xb_train = trainx.reshape(trainx.shape[0], 1)
    yb_train = trainx.reshape(trainy.shape[0], 1)
    zb_train = trainx.reshape(trainz.shape[0], 1)
    tb_train = trainx.reshape(traint.shape[0], 1)
    ub_train = trainx.reshape(trainub.shape[0], 1)
    vb_train = trainx.reshape(trainvb.shape[0], 1)
    wb_train = trainx.reshape(trainwb.shape[0], 1)
    pb_train = trainx.reshape(trainpb.shape[0], 1)

    # inital values
    x_0 = np.tile(xdata, 31 * 31)
    y_0 = np.tile(ydata.repeat(31), 31)
    z_0 = zdata.repeat(31 * 31)
    t_0 = np.array([0] * x_0.shape[0])

    u_0, v_0, w_0, p_0 = data_generate(x_0, y_0, z_0, t_0)

    ui_train = u_0.reshape(u_0.shape[0], 1)
    vi_train = v_0.reshape(v_0.shape[0], 1)
    wi_train = w_0.reshape(w_0.shape[0], 1)
    p0_train = p_0.reshape(p_0.shape[0], 1)
    xi_train = x_0.reshape(x_0.shape[0], 1)
    yi_train = y_0.reshape(y_0.shape[0], 1)
    zi_train = z_0.reshape(z_0.shape[0], 1)
    ti_train = t_0.reshape(t_0.shape[0], 1)
    
    # xyzt data
    xx = np.random.randint(31, size=10000) / 15 - 1
    yy = np.random.randint(31, size=10000) / 15 - 1
    zz = np.random.randint(31, size=10000) / 15 - 1
    tt = np.random.randint(11, size=10000) / 10

    uu, vv, ww, pp = data_generate(xx, yy, zz, tt)

    x_train = xx.reshape(xx.shape[0], 1)
    y_train = yy.reshape(yy.shape[0], 1)
    z_train = zz.reshape(zz.shape[0], 1)
    t_train = tt.reshape(tt.shape[0], 1)

    model = NSPINN(xi_train, yi_train, zi_train, ti_train,
                     ui_train, vi_train, wi_train,
                     xb_train, yb_train, zb_train, tb_train,
                     ub_train, vb_train, wb_train,
                     x_train, y_train, z_train, t_train, layers)

    model.Adam_train(5000, 1e-3)
    model.Adam_train(5000, 1e-4)
    model.Adam_train(50000, 1e-5)
    model.Adam_train(50000, 1e-6)
    model.BFGS_train()

    x_star = (np.random.rand(1000, 1) - 1 / 2) * 2
    y_star = (np.random.rand(1000, 1) - 1 / 2) * 2
    z_star = (np.random.rand(1000, 1) - 1 / 2) * 2
    t_star = np.random.randint(11, size=(1000,1)) / 10

    u_star, v_star, w_star, p_star = data_generate(x_star, y_star, z_star, t_star)
       
    u_pred, v_pred, w_pred, p_pred = model.predict(x_star, y_star, z_star, t_star)

    # Error
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_w = np.linalg.norm(w_star - w_pred, 2) / np.linalg.norm(w_star, 2)
    error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)

    print('Error u: %e' % error_u)
    print('Error v: %e' % error_v)
    print('Error w: %e' % error_w)
    print('Error p: %e' % error_p)
    
    scipy.io.savemat('../NS3D_beltrami_data_%s.mat',
                     {'x_star':x_star, 'y_star':y_star,'z_star':z_star,'U_star':u_star, 'V_star':v_star, 'W_pred':w_star, 'P_pred':p_star})

    scipy.io.savemat('../NS3D_beltrami_%s.mat',
                     {'U_pred':u_pred, 'V_pred':v_pred, 'W_pred':w_pred, 'P_pred':p_pred})