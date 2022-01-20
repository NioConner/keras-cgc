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
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback


class ROCCallback(Callback):
    def __init__(self, training_data, validation_data, test_data):
        self.train_X = training_data[0]
        self.train_Y = training_data[1]
        self.validation_X = validation_data[0]
        self.validation_Y = validation_data[1]
        self.test_X = test_data[0]
        self.test_Y = test_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        train_prediction = self.model.predict(self.train_X)
        validation_prediction = self.model.predict(self.validation_X)
        test_prediction = self.model.predict(self.test_X)

        # Iterate through each task and output their ROC-AUC across different datasets
        for index, output_name in enumerate(self.model.output_names):
            train_roc_auc = roc_auc_score(self.train_Y[index], train_prediction[index])
            validation_roc_auc = roc_auc_score(self.validation_Y[index], validation_prediction[index])
            test_roc_auc = roc_auc_score(self.test_Y[index], test_prediction[index])
            print(
                'ROC-AUC-{}-Train: {} ROC-AUC-{}-Validation: {} ROC-AUC-{}-Test: {}'.format(
                    output_name, round(train_roc_auc, 4),
                    output_name, round(validation_roc_auc, 4),
                    output_name, round(test_roc_auc, 4)
                )
            )

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def train():
    # load data
    train_data, train_label, validation_data, validation_label, test_data, test_label, output_info = data_preparation()
    num_featuers = train_data.shape[1]
    print(num_featuers)

    input_layer = Input(shape=(num_featuers,))

    cgc_output = CGC(num_tasks=2, num_experts_task=3, num_experts_share=3,
                     units=[4], output_share=False, name="cgc")(input_layer)
    target_outputs = tf.split(cgc_output, num_or_size_splits=2, axis=1)

    output_layers = []

    for index, task_layer in enumerate(target_outputs):
        print(index)
        print(task_layer)
        task_layer = tf.squeeze(task_layer, axis=1)
        tower_layer = Dense(
            units=8,
            activation='relu',
            kernel_initializer=VarianceScaling())(task_layer)
        output_layer = Dense(
            units=output_info[index][0],
            name=output_info[index][1],
            activation='softmax',
            kernel_initializer=VarianceScaling())(tower_layer)
        output_layers.append(output_layer)

    model = Model(inputs=[input_layer], outputs=output_layers)
    adam_optimizer = Adam()
    model.compile(loss={'income': 'binary_crossentropy', 'marital': 'binary_crossentropy'}, optimizer=adam_optimizer,
                  metrics=['accuracy'])
    model.summary()

    model.fit(
        x=train_data,
        y=train_label,
        validation_data=(validation_data, validation_label),
        callbacks=[
            ROCCallback(
                training_data=(train_data, train_label),
                validation_data=(validation_data, validation_label),
                test_data=(test_data, test_label)
            )
        ],
        epochs=100
    )


if __name__ == '__main__':
    train()
