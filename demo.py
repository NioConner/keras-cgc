#!/usr/bin/env python3
# encoding: utf-8
'''
@author: NioConner
@contact: 798225589@qq.com
@file: demo.py
@time: 2022-01-19 11:18
@desc:
'''
from model.CGC import CGC
from utils.DataGenerater import data_preparation
from tensorflow.keras.layers import Input, Dense


def train():
    # load data
    train_data, train_label, validation_data, validation_label, test_data, test_label, output_info = data_preparation()
    num_featuers = train_data.shape[1]
    print(num_featuers)

    input_layer = Input(shape=(num_featuers,))

    CGC(num_tasks=2, num_experts_task=3, num_experts_share=3,
        units=[256], output_share=False, name="cgc")(input_layer)

train()
