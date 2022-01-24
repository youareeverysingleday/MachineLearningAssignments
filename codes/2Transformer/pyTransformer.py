#!/usr/bin/python3

# author: Youareeverysingleday
# contact: implement yourself transformer model.
# datetime:2020/3/16 11:08
# Description: 
#   1. reference https://tensorflow.google.cn/tutorials/text/transformer?hl=zh_cn
#   2. 
# software: VSCode


import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

class TransformerModel:
    
    def __init__(self, N, d_model, d_ff, h, d_k, d_v, P_drop, l_rate, train_steps):
        """[summary]
        Implement Transformer model by yourself. complete transformer structure.
        Args:
            N ([type]): [description]
            d_model ([type]): [description]
            d_ff ([type]): [description]
            h ([type]): [description]
            d_k ([type]): [description]
            d_v ([type]): [description]
            P_drop ([type]): [description]
            l_rate ([type]): [description]
            train_steps ([type]): [description]
        """
        self.N = N
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.P_drop = P_drop
        self.l_rate = l_rate
        self.train_steps = train_steps
    
    def input(self):
        pass
    
    def Embedding(self):
        pass
    
    def PositionEncoding(self):
        pass
    
    def MutltHeadAttention(self, Query, Key, Value):
        pass

    def FeedForward(self):
        pass

    def LinearLayer(self):
        pass
    
    def softmax(self):
        pass
    
    def output(self):
        pass