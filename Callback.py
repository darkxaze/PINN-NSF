# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:55:53 2020

@author: nastavirs
"""
import tensorflow as tf
import numpy as np
def callback(self, loss):
        print('Loss: %.3e' % loss)