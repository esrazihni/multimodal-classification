import numpy as np
import pandas as pd
import sys
import os
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes
import time
import yaml
import json
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D, BatchNormalization, Activation, Input, concatenate
from keras.callbacks import EarlyStopping
from keras.backend.tensorflow_backend import set_session
from keras.utils import multi_gpu_model
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid


from helper import dataset, model
from imaging_predictive_models import imaging_dataset
from clinical_predictive_models import clinical_dataset, MLP
from multimodal_prediction_helper import multimodal_dataset
from keras_helper import EpochEvaluation

#### ENVIRONMENT AND SESSION SET UP ####################################################################
# set the environment variable
os.environ["KERAS_BACKEND"] = "tensorflow"
# Silence INFO logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# create a configuration protocol
config = tf.ConfigProto()
# set the allow_growth option to true in the protocol
config.gpu_options.allow_growth = True
# define GPU to use
config.gpu_options.visible_device_list = "0,1"

# start a sesstion that uses the configuration protocol
set_session(tf.Session(config=config))

#### READ CONFIGURATION FILE ###########################################################################
def join(loader,node):
    seq = loader.construct_sequence(node)
    return ''.join(str(i) for i in seq)

yaml.add_constructor('!join',join)
cfg = yaml.load(open('config.yml', 'r'))

#### ASSIGN PATHS AND VARIABLES #########################################################################
dataset_name = cfg['dataset name']
img_splits_path = cfg['imaging dataset']['splits path']
img_feat_splits_path = 'data/' + cfg['imaging dataset']['feature splits path']
img_models_path = cfg['imaging dataset']['models path']
img_params_folder = '../TOF-based/modeling_results/1kplus_multimodal/params/'
img_scores_folder = '../TOF-based/modeling_results/1kplus_multimodal/performance_scores/'
clin_splits_path = cfg['clinical dataset']['splits path']
clin_feat_splits_path = 'data/'+ cfg['clinical dataset']['feature splits path']
clin_models_path = cfg['clinical dataset']['models path']
clin_params_folder = '../clinical parameter-based/modeling_results/1kplus_multimodal/params/'
clin_scores_folder = '../clinical parameter-based/modeling_results/1kplus_multimodal/performance_scores/'
num_splits = cfg['number of runs']

#### LOAD BOTH CLINICAL AND IMAGING DATA #################################################################
img_data = imaging_dataset(dataset_name)
img_sets = img_data.assign_train_val_test_sets(img_splits_path)
clin_data = clinical_dataset(dataset_name)
clin_sets = clin_data.assign_train_val_test_sets(clin_splits_path)
features = multimodal_dataset(dataset_name)
features.load_feature_sets(img_feat_splits_path, clin_feat_splits_path)

def train_and_evaluate_CNN(training_data, test_data, params, num_training_runs = 100):
    X_tr, y_tr = training_data
    X_te, y_te = test_data

    AUC_trs = []
    AUC_tes = []
    for i in range(num_training_runs):
        model = Sequential()
        model.add(Conv3D(params['num_filters'][0], params['arc_params']['filter_size'], strides = params['arc_params']['filter_stride'],
                     padding="same",kernel_regularizer= keras.regularizers.l2(params['l2_reg']),input_shape=(156,192,64,1)))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size= params['arc_params']['pool_size']))

        model.add(Conv3D(params['num_filters'][1], params['arc_params']['filter_size'], strides = params['arc_params']['filter_stride'],
                     padding="same",kernel_regularizer= keras.regularizers.l2(params['l2_reg']) ))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=params['arc_params']['pool_size']))

        model.add(Conv3D(params['num_filters'][2], params['arc_params']['filter_size'], strides = params['arc_params']['filter_stride'],
                     padding="same",kernel_regularizer= keras.regularizers.l2(params['l2_reg'])))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=params['arc_params']['pool_size']))

        model.add(Flatten())
        model.add(Dense(params['num_neurons_in_powers']*params['num_filters'][2], activation='relu',kernel_regularizer= keras.regularizers.l2(params['l2_reg'])))
        model.add(Dropout(params['dropout']))
        model.add(Dense(2 , activation='softmax',kernel_regularizer= keras.regularizers.l2(params['l2_reg'])))

        optimizer = keras.optimizers.Adam(lr = params['learning_rate'])

        model.compile(loss='binary_crossentropy',optimizer=optimizer)

        parallel_model = multi_gpu_model(model, 2)
        parallel_model.compile(loss='binary_crossentropy',optimizer=optimizer)

        e_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.02, patience = 2
                           , mode='auto')

        callbacks = [e_stop]

        start = time.time()
        history = parallel_model.fit(X_tr, y_tr,  callbacks = callbacks, validation_data = (X_te,y_te),
                                 batch_size = params['batch_size'], epochs=20,verbose = 0)
        end = time.time()

        model.set_weights(parallel_model.get_weights())

        probs_tr = model.predict(X_tr, batch_size = 8)
        probs_te = model.predict(X_te, batch_size = 8)

        score_tr = roc_auc_score(y_tr, probs_tr)
        score_te = roc_auc_score(y_te, probs_te)

        AUC_trs.append(score_tr)
        AUC_tes.append(score_te)
        print('Training time for run %i was around %i minutes'%(i, np.floor((end-start)/60)))
        keras.backend.clear_session()

    return AUC_trs, AUC_tes

def train_and_evaluate_MLP(training_data, test_data, params, num_training_runs = 100):
    X_tr, y_tr = training_data
    X_te, y_te = test_data

    AUC_trs = []
    AUC_tes = []
    for i in range(num_training_runs):
        e_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.01, patience = 5, mode='min')
        callbacks = [e_stop]

        optimizer = keras.optimizers.Adam(lr = params['learning_rate'])

        model = Sequential()
        model.add(Dense(params['num_neurons'],input_dim = 7, kernel_initializer = 'glorot_uniform', activation = 'relu', kernel_regularizer = keras.regularizers.l2(params['l2_ratio'])))
        model.add(Dropout(params['dropout_rate']))
        model.add(Dense(2, kernel_initializer = 'glorot_uniform', activation = 'softmax', kernel_regularizer = keras.regularizers.l2(params['l2_ratio'])))
        model.compile(loss = 'binary_crossentropy', optimizer = optimizer)

        history = model.fit(X_tr, y_tr, callbacks= callbacks, validation_data = (X_te, y_te), epochs = 100, batch_size = params['batch_size'], verbose = 0)

        probs_tr = model.predict(X_tr, batch_size = 8)
        probs_te = model.predict(X_te, batch_size = 8)

        score_tr = roc_auc_score(y_tr, probs_tr)
        score_te = roc_auc_score(y_te, probs_te)

        AUC_trs.append(score_tr)
        AUC_tes.append(score_te)
        keras.backend.clear_session()

    return AUC_trs, AUC_tes

def train_and_evaluate_end_to_end(img_X_tr, clin_X_tr, y_tr, img_X_te, clin_X_te, y_te, params,num_training_runs = 100):


    AUC_trs = []
    AUC_tes = []

    for i in range(num_training_runs):
        img_input = Input(shape= (156,192,64,1), name='image_input')
        clin_input = Input(shape= (clin_X_tr.shape[1],), name='clinical_input')
        x1 = Conv3D(params['num_filters'][0], (3,3,3), strides = (1,1,1),padding="same",
            kernel_regularizer= keras.regularizers.l2(params['l2_ratio']))(img_input)
        x1 = Activation('relu')(x1)
        x1 = MaxPooling3D(pool_size=(3,3,3))(x1)

        x1 = Conv3D(params['num_filters'][1], (3,3,3), strides = (1,1,1),padding="same",
            kernel_regularizer= keras.regularizers.l2(params['l2_ratio']))(x1)
        x1 = Activation('relu')(x1)
        x1 = MaxPooling3D(pool_size=(3,3,3))(x1)

        x1 = Conv3D(params['num_filters'][2], (3,3,3), strides = (1,1,1),padding="same",
            kernel_regularizer= keras.regularizers.l2(params['l2_ratio']))(x1)
        x1 = Activation('relu')(x1)
        x1 = MaxPooling3D(pool_size=(3,3,3))(x1)

        x1 = Flatten()(x1)
        x1 = Dense(params['num_filters'][2]*2, activation='relu',
           kernel_regularizer= keras.regularizers.l2(params['l2_ratio']))(x1)
        x1 = Dropout(params['dropout_rate'])(x1)
        x1 = Dense(params['num_neurons_embedding'][1], activation='relu',
           kernel_regularizer= keras.regularizers.l2(params['l2_ratio']))(x1)



        x2 = Dense(params['num_neurons_MLP'], activation = 'relu', 
           kernel_regularizer= keras.regularizers.l2(params['l2_ratio']))(clin_input)
        x2 = Dropout(params['dropout_rate'])(x2)
        x2 = Dense(params['num_neurons_embedding'][0], activation='relu',
           kernel_regularizer= keras.regularizers.l2(params['l2_ratio']))(x2)

        x = concatenate([x1, x2])
        x = Dense(params['num_neurons_final'], activation = 'relu', 
          kernel_regularizer= keras.regularizers.l1(params['l2_ratio']))(x)
        x= Dropout(params['dropout_rate'])(x)

        output = Dense(2,activation= 'softmax', kernel_regularizer= keras.regularizers.l2(params['l2_ratio']))(x)

        model = Model(inputs=[img_input, clin_input], outputs=[output])

        optimizer = keras.optimizers.Adam(lr = params['learning_rate'])
        model.compile(loss='binary_crossentropy', optimizer = optimizer)

        e_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.02, patience = 2, mode='auto')
        callbacks = [e_stop]

        start= time.time()
        history = model.fit(
            {'image_input' : img_X_tr,
            'clinical_input' : clin_X_tr},#inputs
            y_tr, #output
            callbacks = callbacks,
            validation_data= ([img_X_te, clin_X_te],y_te),
            epochs=20,
            batch_size= params['batch_size'], 
            verbose=0)
        end= time.time()

        probs_tr = model.predict([img_X_tr,clin_X_tr],batch_size = 8)
        probs_te = model.predict([img_X_te,clin_X_te],batch_size = 8)
        score_tr = roc_auc_score(y_tr, probs_tr)
        score_te = roc_auc_score(y_te, probs_te)

        AUC_trs.append(score_tr)
        AUC_tes.append(score_te)

        print('Training time for run %i was around %i minutes'%(i, np.floor((end-start)/60)))
        keras.backend.clear_session()

    return AUC_trs, AUC_tes

def train_and_evaluate_feat_extract(img_X_tr, clin_X_tr, y_tr, img_X_te, clin_X_te, y_te, params,num_training_runs = 100):


    AUC_trs = []
    AUC_tes = []

    for i in range(num_training_runs):
        img_input = Input(shape= (img_X_tr.shape[1],), name='image_input')
        clin_input = Input(shape= (clin_X_tr.shape[1],), name='clinical_input')

        dense1 = Dense(params['num_neurons_embedding'][0], activation = 'relu', 
                    kernel_regularizer= keras.regularizers.l2(params['l2_ratio']))(clin_input)

        dense2 = Dense(params['num_neurons_embedding'][1], activation = 'relu', 
                    kernel_regularizer= keras.regularizers.l2(params['l2_ratio']))(img_input)

        x = concatenate([dense1, dense2])
        x = Dense(params['num_neurons_final'], activation = 'relu', 
                kernel_regularizer= keras.regularizers.l2(params['l2_ratio']))(x)
        x= Dropout(params['dropout_rate'])(x)

        output = Dense(2, activation= 'softmax', kernel_regularizer= keras.regularizers.l2(params['l2_ratio']))(x)        

        optimizer = keras.optimizers.Adam(lr = params['learning_rate'])

        model = Model(inputs=[img_input, clin_input], outputs=[output])
        model.compile(loss='binary_crossentropy', optimizer = optimizer)

        e_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.01, patience = 5, mode='auto')
        callbacks = [e_stop]

        history = model.fit({'image_input' : img_X_tr,
                             'clinical_input' : clin_X_tr},
                            y_tr, 
                            callbacks = callbacks,
                            validation_data= ([img_X_te, clin_X_te],y_te), 
                            epochs=100, 
                            batch_size= params['batch_size'], 
                            verbose=0)



        probs_tr = model.predict([img_X_tr,clin_X_tr],batch_size = 8)
        probs_te = model.predict([img_X_te,clin_X_te],batch_size = 8)
        score_tr = roc_auc_score(y_tr, probs_tr)
        score_te = roc_auc_score(y_te, probs_te)

        AUC_trs.append(score_tr)
        AUC_tes.append(score_te)

        keras.backend.clear_session()

    return AUC_trs, AUC_tes


# fix seed
np.random.seed(1)
tf.set_random_seed(2)
import random as rn
rn.seed(3)

options = [ 'CNN', 'end-to-end']

if 'MLP' in options:
    for i in range(num_splits):
        X_tr = clin_sets[i]['train_data']
        y_tr = clin_sets[i]['train_labels']
        X_val = clin_sets[i]['val_data']
        y_val = clin_sets[i]['val_labels']
        X_te = clin_sets[i]['test_data']
        y_te = clin_sets[i]['test_labels']
        X_train = np.concatenate((X_tr,X_val))
        y_train = np.concatenate((y_tr,y_val))
        y_tr = pd.get_dummies(y_tr)
        y_val = pd.get_dummies(y_val)
        y_te = pd.get_dummies(y_te)
        y_train = pd.get_dummies(y_train.reshape(250,))

        with open(clin_params_folder+ 'best_MLP_multimodal_tuning_parameters_split_'+str(i+1)+'.json') as json_file:  
            tuning_params = json.load(json_file)
        print(tuning_params)

        AUC_trs, AUC_tes = train_and_evaluate_MLP((X_train,y_train),(X_te,y_te),tuning_params,num_training_runs=100)
        np.savetxt('../clinical parameter-based/modeling_results/1kplus_multimodal/performance_scores/outer_loop_AUC_performance_over_100_runs_model_'+str(i+1)+'.csv', [AUC_trs, AUC_tes], delimiter=",")


if 'CNN' in options:
    for i in range(num_splits):
        X_tr = img_sets[i]['train_data']
        y_tr = img_sets[i]['train_labels']
        X_val = img_sets[i]['val_data']
        y_val = img_sets[i]['val_labels']
        X_te = img_sets[i]['test_data']
        y_te = img_sets[i]['test_labels']
        X_train = np.concatenate((X_tr,X_val))
        y_train = np.concatenate((y_tr,y_val))
        y_tr = pd.get_dummies(y_tr)
        y_val = pd.get_dummies(y_val)
        y_te = pd.get_dummies(y_te)
        y_train = pd.get_dummies(y_train)

        with open(img_params_folder+ 'best_tuning_params_split_'+str(i+1)+'.json') as json_file:  
            tuning_params = json.load(json_file)
        print(tuning_params)

        AUC_trs, AUC_tes = train_and_evaluate_CNN((X_train,y_train),(X_te,y_te),tuning_params,num_training_runs=100)
        np.savetxt('../TOF-based/modeling_results/1kplus_multimodal/performance_scores/outer_loop_AUC_performance_over_100_runs_model_'+str(i+1)+'.csv', [AUC_trs, AUC_tes], delimiter=",")

if 'feature' in options:
    for i in range(num_splits):
        img_X_tr = features.img_sets[i]['train_data']
        img_X_val = features.img_sets[i]['val_data']
        img_X_train = np.concatenate((img_X_tr,img_X_val))
        img_X_te = features.img_sets[i]['test_data']

        clin_X_tr = features.clin_sets[i]['train_data']
        clin_X_val = features.clin_sets[i]['val_data']
        clin_X_train = np.concatenate((clin_X_tr,clin_X_val))
        clin_X_te = features.clin_sets[i]['test_data']

        y_tr = features.img_sets[i]['train_labels']
        y_val = features.img_sets[i]['val_labels']
        y_train = np.concatenate((y_tr,y_val))
        y_te = features.img_sets[i]['test_labels']

        y_tr = pd.get_dummies(y_tr)
        y_val = pd.get_dummies(y_val)
        y_train = pd.get_dummies(y_train)
        y_te = pd.get_dummies(y_te)

        with open('modeling_results/1kplus_multimodal/params/end_to_end/best_tuning_params_split_'+str(i+1)+'.json') as json_file:  
            tuning_params = json.load(json_file)
        print(tuning_params)

        AUC_trs, AUC_tes = train_and_evaluate_feat_extract(img_X_train, clin_X_train, y_train, img_X_te, clin_X_te, y_te,
                                                     tuning_params, num_training_runs=100)
        np.savetxt('modeling_results/1kplus_multimodal/performance_scores/feature_concatenation/outer_loop_AUC_performance_over_100_runs_model_'+str(i+1)+'.csv', [AUC_trs, AUC_tes], delimiter=",")

if 'end-to-end' in options:
    for i in range(num_splits):
        img_X_tr = img_sets[i]['train_data']
        img_X_val = img_sets[i]['val_data']
        img_X_train = np.concatenate((img_X_tr,img_X_val))
        img_X_te = img_sets[i]['test_data']

        clin_X_tr = clin_sets[i]['train_data']
        clin_X_val = clin_sets[i]['val_data']
        clin_X_train = np.concatenate((clin_X_tr,clin_X_val))
        clin_X_te = clin_sets[i]['test_data']

        y_tr = img_sets[i]['train_labels']
        y_val = img_sets[i]['val_labels']
        y_train = np.concatenate((y_tr,y_val))
        y_te = img_sets[i]['test_labels']

        y_tr = pd.get_dummies(y_tr)
        y_val = pd.get_dummies(y_val)
        y_train = pd.get_dummies(y_train)
        y_te = pd.get_dummies(y_te)

        with open('modeling_results/1kplus_multimodal/params/end_to_end/best_tuning_params_split_'+str(i+1)+'.json') as json_file:  
            tuning_params = json.load(json_file)
        print(tuning_params)

        AUC_trs, AUC_tes = train_and_evaluate_end_to_end(img_X_train, clin_X_train, y_train, img_X_te, clin_X_te, y_te,
                                                     tuning_params, num_training_runs=100)
        np.savetxt('modeling_results/1kplus_multimodal/performance_scores/end_to_end/outer_loop_AUC_performance_over_100_runs_model_'+str(i+1)+'.csv', [AUC_trs, AUC_tes], delimiter=",")
