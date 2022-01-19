#!/usr/bin/env python3
# encoding: utf-8
'''
@author: NioConner
@contact: 798225589@qq.com
@file: CGC.py
@time: 2022-01-14 16:22
@desc:
'''
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
import tensorflow as tf


class CGC(Layer):
    '''
    cgc model
    '''

    def __init__(self, num_tasks=2, num_experts_task=1, num_expert_share=2, units=[256], activation='relu',
                 output_activation='linear', output_share=False, use_bias=True, l2_reg=0, seed=1024, **kwargs):
        self.num_tasks = num_tasks
        self.num_experts_task = num_experts_task
        self.num_experts_share = num_expert_share
        self.units = units
        self.activition = activation
        self.output_activation = output_activation
        self.output_share = output_share
        self.use_bias = use_bias
        self.l2_reg = l2_reg
        self.seed = seed
        self.num_experts = self.num_tasks * self.num_experts_task + self.num_experts_share

    def build(self, input_shape):
        '''
        Method for creating the layer weight.
        :param input_shape: tensor_shape
        '''
        assert input_shape is not None and len(input_shape) >= 2
        units = input_shape[-1] + self.units
        self.experts = []
        self.experts_bais = []

        for n in range(self.num_experts):
            self.experts.append([self.ad_weights(name=f'experts_weight_{n}_{i}', shape=(units[i], units[i + 1]),
                                                 initializer=tf.initializers.glorot_normal(seed=self.seed),
                                                 regularizer=l2(self.l2_reg), trainable=True)
                                 for i in range(len(units) - 1)])

            self.experts_bais.append(
                self.add_weight(name=f'expert_bais_weight_{n}_{i}', shape=(units[i]),
                                initializer=tf.initializers.Zeros(), trainable=True) for i in range(len(units) - 1))

        self.gating = [tf.keras.layers.Dense(self.num_experts, name=f'gating_{}', use_bias=False) for i in
                       range(self.tasks)]

        if self.output_share:
            self.gating_share = tf.keras.layers.Dense(self.num_experts, name=f'gating_share', use_bias=False)

        self.activation_layers = [
            tf.keras.layers.Activation(
                self.output_activation if i == len(self.units) - 1 and self.output_activation else self.activition)
            for i in range(len(self.units))
        ]
        super(CGC, self).build(input_shape)

    def call(self, inputs, **kwargs):
        target_inputs = inputs



