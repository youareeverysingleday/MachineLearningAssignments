
# author:Administrator
# contact: SystemEngineer
# datetime:2021/11/17 15:26
# LastEditTime: 2021-11-17 22:08:38
# software: VSCode
# Description: Implements NeuMF by tensorflow 2.x.

import tensorflow as tf
import numpy as np
import pandas as pd

class GMF:
    def __init__(self) -> None:
        pass


class MLP:
    def __init__(self) -> None:
        pass
    
    def MLPModel(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Embedding(output_dim=32, input_dim=4000, input_length=400))
        # 增加平坦层。
        model.add(tf.keras.layers.Flatten())
        # 建立全连接的隐藏层。
        model.add(tf.keras.layers.Dense(units=256, activation='relu'))
        # 设置了0.3的丢弃层。
        model.add(tf.keras.layers.Dropout(0.3))
        # 输出层。
        model.add(tf.keras.layers.Dense(units=4, activation='softmax'))
        # 查看效果
        model.summary()
        return model


class NeuMF:
    def __init__(self) -> None:
        pass
