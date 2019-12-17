"""
File name: CNN_gridsearch.py
Author: Esra Zihni
Date created: 04.04.2019


This file trains models on different parameters using grid search. It saves the trained models, AUC metrics on corresdponding 
training, validation and test sets and evolution of loss and AUC metrics over epochs during training.
"""

import numpy as np
import pandas as pd
import sys
import os
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes
import time
import yaml

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D, BatchNormalization, Activation
from keras.callbacks import EarlyStopping
from keras.backend.tensorflow_backend import set_session
from keras.utils import multi_gpu_model
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid

from keras_helper import *
from helper import *
from plotting_helper import *
from imaging_predictive_models import *


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

# fix seed
np.random.seed(1)
tf.set_random_seed(2)


#### READ CONFIGURATION FILE ###########################################################################
def join(loader,node):
    seq = loader.construct_sequence(node)
    return ''.join(str(i) for i in seq)

yaml.add_constructor('!join',join)
cfg = yaml.load(open('config.yml', 'r'))

#### ASSIGN PATHS AND VARIABLES #########################################################################
dataset_name = cfg['dataset name']
split_path = cfg['imaging dataset']['splits path']
test_size = cfg['test size']
val_size = cfg['validation size']
def_params = cfg['hyperparameters']
tuning_params = cfg['tuning parameters']
num_splits = cfg['number of runs']
arc_params = {'arc_params' : [{'filter_size': (3,3,3), 'filter_stride':(1,1,1),'pool_size': (3,3,3)}]}#, {'filter_size': (3,3,3), 'filter_stride':(2,2,2),'pool_size': (2,2,2)},
			  #{'filter_size': (5,5,5), 'filter_stride':(1,1,1),'pool_size': (3,3,3)}, {'filter_size': (5,5,5), 'filter_stride':(2,2,2),'pool_size': (2,2,2)}]}
#arc_params = {'filter_size':[(3,3,3),(5,5,5)], 'filter_stride':[(1,1,1),(2,2,2)], 'pool_size': [(2,2,2),(3,3,3)]}
tuning_params.update(arc_params)

#### LOAD DATA ###########################################################################################
data = imaging_dataset(dataset_name)
sets = data.assign_train_val_test_sets(splits_path,test_size,val_size,num_splits)

for i in range(num_splits):
	#### ASSIGN TRAINING, TEST AND VALIDATION SETS FOR CURRENT SPLIT ##########################################
	current_split_num = i+1
	X_tr = sets[current_split_num-1]['train_data']
	y_tr = sets[current_split_num-1]['train_labels'].values
	X_te = sets[current_split_num-1]['test_data']
	y_te = sets[current_split_num-1]['test_labels'].values
	X_val = sets[current_split_num-1]['val_data']
	y_val = sets[current_split_num-1]['val_labels'].values

	if def_params['out_activation'] == 'softmax':
		y_tr = pd.get_dummies(y_tr)
		y_val = pd.get_dummies(y_val)
		y_te = pd.get_dummies(y_te)


	#### START GRID SEARCH #####################################################################################
	start_grid = time.time()
	comb = 1

	for tune in ParameterGrid(tuning_params):

		for j in range(2):
			model_path = 'final_gridsearch_softmax_2/models/split_'+str(current_split_num)+'/trained_model_params_comb_'+str(comb)+'_test_'+str(j)+'.h5'
			if os.path.isfile(model_path):
				pass

			else:		
				try:

					model = Sequential()
					model.add(Conv3D(tune['num_filters'][0], tune['arc_params']['filter_size'], strides = tune['arc_params']['filter_stride'],padding="same",kernel_regularizer= keras.regularizers.l2(tune['l2_reg']),input_shape=(156,192,64,1)))
					model.add(Activation(def_params['hidden_activation']))
					model.add(MaxPooling3D(pool_size=tune['arc_params']['pool_size']))
					
					model.add(Conv3D(tune['num_filters'][1], tune['arc_params']['filter_size'], strides = tune['arc_params']['filter_stride'],padding="same",kernel_regularizer= keras.regularizers.l2(tune['l2_reg']) ))
					model.add(Activation(def_params['hidden_activation']))
					model.add(MaxPooling3D(pool_size=tune['arc_params']['pool_size']))
					
					model.add(Conv3D(tune['num_filters'][2], tune['arc_params']['filter_size'], strides = tune['arc_params']['filter_stride'],padding="same",kernel_regularizer= keras.regularizers.l2(tune['l2_reg'])))
					model.add(Activation(def_params['hidden_activation']))
					model.add(MaxPooling3D(pool_size=tune['arc_params']['pool_size']))
					
					model.add(Flatten())
					model.add(Dense(tune['num_neurons_in_powers']*tune['num_filters'][2], activation=def_params['hidden_activation'],kernel_regularizer= keras.regularizers.l2(tune['l2_reg'])))
					model.add(Dropout(tune['dropout']))

					if def_params['out_activation'] == 'softmax':
						model.add(Dense(2 , activation=def_params['out_activation'], kernel_regularizer= keras.regularizers.l2(tune['l2_reg'])))
					else:
						model.add(Dense(1 , activation=def_params['out_activation'], kernel_regularizer= keras.regularizers.l2(tune['l2_reg'])))

					#print(tune['num_neurons_in_powers']*tune['num_filters'][2])

					optimizer = keras.optimizers.Adam(lr = tune['learning_rate'])

					model.compile(loss=def_params['loss_func'],optimizer=optimizer)

					parallel_model = multi_gpu_model(model, 2)
					parallel_model.compile(loss=def_params['loss_func'],optimizer=optimizer)

					e_stop = EarlyStopping(monitor = 'val_loss', min_delta = def_params['min_delta'], patience = def_params['iter_patience'], mode='auto')
					ep_val = EpochEvaluation(validation_data = (X_val, y_val),training_data = (X_tr, y_tr), test_data=(X_te,y_te))
					callbacks = [ep_val, e_stop]

					print('Training for comb %i with parameters %s'%(comb,str(tune)))

					start = time.time()
					history = parallel_model.fit(X_tr, y_tr, validation_data = (X_val, y_val),  callbacks = callbacks, batch_size = tune['batch_size'], epochs=def_params['epochs'],verbose = 0)
					end = time.time()

					print('Training completed in around %i minutes'%(np.floor((end-start)/60)))

					model.set_weights(parallel_model.get_weights())

					models_path = 'final_gridsearch_softmax_2/models/split_'+str(current_split_num)+'/'
					if not os.path.exists(models_path):
						os.makedirs(models_path)
					model.save(models_path+ 'trained_model_params_comb_'+str(comb)+'_test_'+str(j)+'.h5')

					loss = history.history['loss']
					val_loss = history.history['val_loss']
					auc= ep_val.roc_auc
					val_auc = ep_val.val_roc_auc
					test_auc = ep_val.test_roc_auc

					scores_path = 'final_gridsearch_softmax_2/scores/split_'+str(current_split_num)+'/'
					if not os.path.exists(scores_path):
						os.makedirs(scores_path)
					np.savetxt(scores_path+'loss and_auc_over epochs_params_comb_'+str(comb)+'_test_'+str(j)+'.csv', [auc,val_auc,test_auc,loss,val_loss], delimiter=",")

					figures_path = 'final_gridsearch_softmax_2/figures/split_'+str(current_split_num)+'/'
					if not os.path.exists(figures_path):
						os.makedirs(figures_path)
					plot_evolution(loss,val_loss,auc,val_auc,test_auc,tune,figures_path+ 'trained_model_params_comb_'+str(comb)+'_test_'+str(j)+'.png')	

					#probs_tr = model.predict(X_tr, batch_size = 8).T[0]
					#probs_te = model.predict(X_te, batch_size = 8).T[0]
					#probs_val = model.predict(X_val, batch_size = 8).T[0]

					#score_tr = roc_auc_score(y_tr, probs_tr)
					#score_te = roc_auc_score(y_te, probs_te)
					#score_val = roc_auc_score(y_val, probs_val)

					#np.savetxt('parameter search/final_gridsearch_softmax_2/scores/split_'+str(current_split_num)+'/final_performance_auc_scores_params_comb_'+str(comb)+'_test_'+str(j)+'.csv', [score_tr, score_val, score_te], delimiter=",")					

					del model

				except ValueError:
					pass

			keras.backend.clear_session()


		comb +=1

	end_grid = time.time()

	print('Whole training completed in %i hours %i minutes.'%(np.floor((end_grid-start_grid)/3600), np.floor(((end_grid-start_grid)%3600)/60)))
