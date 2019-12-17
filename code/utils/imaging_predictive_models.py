import numpy as np
import pickle
import time
import os

import pandas as pd
import sys
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes

from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import preprocessing

import keras
import tensorflow as tf
os.environ["KERAS_BACKEND"] = 'tensorflow'
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from helper import *
from keras_helper import *
from plotting_helper import *

from abc import abstractmethod


class imaging_dataset(dataset):
    def __init__(self,name):
        super().__init__(name)
        self.datatype = 'imaging'

    def load_data(self,datasource,labelsource):
        self.X = np.load(datasource)['all_patients']
        self.y = pd.read_pickle(labelsource)
        self.label = list(self.y)[0]

    def preprocess(self):
        # Normalize:
        for i in range(self.X.shape[0]):
            self.X[i] = (self.X[i]-(np.mean(self.X[i])))/(np.std(self.X[i]))


class CNN_3_layers(model):
    def train(self, val_score):
        params = self.def_params.copy()

        np.random.seed(1)
        tf.set_random_seed(2)

        # Start Gridsearch
        best_AUC = 0.5
        AUC_vals = []

        for tune in ParameterGrid(self.tuning_params):
            params.update(tune)

            callbacks = [EarlyStopping(monitor = val_score, min_delta = 0.01, patience = params['iter_patience'], mode='max')]
            optimizer = eval('keras.optimizers.'+params['optimizer'])(lr = params['learning_rate'])

            model = Sequential()
            model.add(Conv3D(32, (3, 3, 3), activation=params['hidden_activation'], padding=params['padding'], input_shape=(156,192,64,1)))
            #model.add(BatchNormalization())
            model.add(MaxPooling3D(pool_size=(2, 2, 2)))

            model.add(Conv3D(64, (3, 3, 3), activation=params['hidden_activation'], padding=params['padding']))
            #model.add(BatchNormalization())
            model.add(MaxPooling3D(pool_size=(3, 3, 3)))

            model.add(Conv3D(128, (3, 3, 3), activation=params['hidden_activation'], padding=params['padding']))
            #model.add(BatchNormalization())
            model.add(MaxPooling3D(pool_size=(4, 4, 4)))

            model.add(Flatten())
            model.add(Dense(256, activation=params['hidden_activation']))
            #model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(1 , activation=params['out_activation']))

            model.compile(loss= params['loss_func'],
                              optimizer= optimizer,
                              metrics= [eval(val_score)])

            parallel_model = multi_gpu_model(model, params['number_of_gpus'])
            parallel_model.compile(loss= params['loss_func'],
                              optimizer= optimizer,
                              metrics=[eval(val_score)])

            history = parallel_model.fit(X_tr, y_tr, callbacks= callbacks, validation_data = (X_val, y_val), epochs = params['epochs'], batch_size = params['batch_size'], verbose = 0)

            model.set_weights(parallel_model.get_weights())

            AUC_val = history.history['val_'+val_score][-1]
            AUC_vals.append(AUC_val)

            if AUC_val > best_AUC:
                best_AUC = AUC_val
                self.best_model = model
                self.best_params = tune

        self.AUC_val = best_AUC

        return AUC_vals

    def save_best_params(self, directory_to_save='', suffix=''):
        json.dump(dict(self.default_params.items() + self.best_params.items()), open(directory_to_save+'/best_'+self.name+'_params'+suffix+'.json', 'w'))

        