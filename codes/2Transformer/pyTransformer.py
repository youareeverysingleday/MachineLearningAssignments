#!/usr/bin/python3

# author: Youareeverysingleday
# contact: implement yourself transformer model.
# datetime:2020/3/16 11:08
# Description: 
#   1. reference https://tensorflow.google.cn/tutorials/text/transformer?hl=zh_cn
#   2. embedding reference: https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/text/word_embeddings.ipynb
#                           https://tensorflow.google.cn/text/guide/word_embeddings
#                           the content of the above two examples are the same, the difference is that the first is Chinese and the second is English.
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
    """[summary]
    refer to offical code of tensorflow, use the top-down method to 
    implement the Transformer Model that yourself understanding.
    参照tensorflow的官方代码，采用自顶向下的方法来实现自己理解的transformer模型。
    """
    
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
    
    def Embedding(self, input_vocabulary_size, d_model):
        """[summary]

        Args:
            input_vocabulary_size ([type]): [description]
            d_model ([type]): [description]
        """
        self.embedding = tf.keras.layers.Embedding(input_vocabulary_size, d_model)
        pass
    
    def get_angles(self, pos, i, d_model):
        """[summary]
        compute position of vocabulary.
        Args:
            pos ([type]): [description]
            i ([type]): [description]
            d_model ([type]): [description]

        Returns:
            [type]: [description]
        """
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
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