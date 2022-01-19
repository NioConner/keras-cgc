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
from tensorflow.python.keras import backend as K


class CGC(Layer):
    '''
    cgc model
    '''

    def __init__(self, num_tasks=2, num_experts_task=1, num_experts_share=2, units=[256], activation='relu',
                 output_activation='linear', output_share=False, use_bias=True, l2_reg=0, seed=1024, **kwargs):
        self.num_tasks = num_tasks
        self.num_experts_task = num_experts_task
        self.num_experts_share = num_experts_share
        self.units = units
        self.activition = activation
        self.output_activation = output_activation
        self.output_share = output_share
        self.use_bias = use_bias
        self.l2_reg = l2_reg
        self.seed = seed
        self.num_experts = num_tasks * num_experts_task + num_experts_share
        super(CGC, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        Method for creating the layer weight.
        :param input_shape: tensor_shape
        '''
        assert input_shape is not None and len(input_shape) >= 2
        units = [input_shape[-1]] + self.units
        self.experts = []
        self.experts_bais = []

        for n in range(self.num_experts):
            self.experts.append([self.add_weight(name=f'experts_weight_{n}_{i}', shape=(units[i], units[i + 1]),
                                                 initializer=tf.initializers.glorot_normal(seed=self.seed),
                                                 regularizer=l2(self.l2_reg), trainable=True)
                                 for i in range(len(units) - 1)])

            self.experts_bais.append(
                self.add_weight(name=f'expert_bais_weight_{n}_{i}', shape=(units[i]),
                                initializer=tf.initializers.Zeros(), trainable=True) for i in range(len(units) - 1))

        self.gating = [tf.keras.layers.Dense(self.num_experts, name=f'gating_{i}', use_bias=False) for i in
                       range(self.num_tasks)]

        if self.output_share:
            self.gating_share = tf.keras.layers.Dense(self.num_experts, name=f'gating_share', use_bias=False)

        self.activation_layers = [
            tf.keras.layers.Activation(
                self.output_activation if i == len(self.units) - 1 and self.output_activation else self.activition)
            for i in range(len(self.units))
        ]
        super(CGC, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # experts 不共享inputs效果会更好
        target_inputs = inputs
        if K.ndim(target_inputs) == 2:
            x0 = tf.tile(tf.expand_dims(target_inputs, axis=1), [1, self.num_experts, 1])
            x1 = [tf.expand_dims(target_inputs, axis=1)] * (self.num_tasks + 1)
        else:
            raise ValueError(f'Unexpected inputs dims {K.ndim(target_inputs)}')
        x0 = tf.split(x0, num_or_size_splits=self.num_experts, axis=1)
        for n in range(self.num_experts):
            x0[n] = tf.squeeze(x0[n], axis=1)
            for i in range(len(self.units)):
                x0[n] = tf.matmul(x0[0], self.experts[n][i])
                x0[n] = self.activation_layers[i](tf.add(x0[n], self.experts_bais[n][i]))
        ret = []
        for i in range(self.num_tasks):
            gating_score = tf.nn.softmax(self.gating[i](x1[i]))
            output_of_experts = tf.stack(
                x0[i * self.num_experts_task:(i + 1) * self.num_experts_task] + x0[-self.num_experts_share:])
            ret.append(tf.matmul(gating_score, output_of_experts))

        if self.output_share:
            gating_score = tf.nn.softmax(self.gating_share(x1[-1]))
            ret.append(tf.matmul(gating_score, tf.stack(x0, axis=1)))
        return tf.concat(ret, axis=1)

    def get_config(self):
        config = {'units': self.units,
                  'num_experts_task': self.num_experts_task,
                  'num_experts_share': self.num_experts_share,
                  'num_task': self.num_tasks,
                  'activation': self.activition,
                  'output_activation': self.output_activation,
                  'output_share': self.output_share,
                  'use_bias': self.use_bias,
                  'l2_reg': self.l2_reg,
                  'seed': self.seed
                  }
        base_config = super(CGC, self).get_config
        base_config.update(config)
        return base_config

    def compute_output_shape(self, input_shape):
        return None, self.num_tasks, self.units
