"""
File name: end_to_end_gridsearch.py
Author: Esra Zihni
Date created: 21.05.2019


"""

import numpy as np
import sys
import os
import pandas as pd
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes
import yaml
import pickle
import json
import time
import keras
import tensorflow as tf

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Input, concatenate, Flatten, Conv3D, MaxPooling3D, Activation
from keras.callbacks import EarlyStopping
from keras.utils import multi_gpu_model
from keras.backend.tensorflow_backend import set_session

from helper import dataset, model
from imaging_predictive_models import imaging_dataset
from clinical_predictive_models import clinical_dataset, MLP
from multimodal_prediction_helper import multimodal_dataset
from plotting_helper import plot_evolution


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


#### READ CONFIGURATION FILE ##########
def join(loader,node):
    seq = loader.construct_sequence(node)
    return ''.join(str(i) for i in seq)

yaml.add_constructor('!join',join)
cfg = yaml.load(open('config.yml', 'r'))

#### ASSIGN PATHS AND VARIABLES #########################################################################
dataset_name = cfg['dataset name']
clin_splits_path = cfg['clinical dataset']['splits path']
img_splits_path = cfg['imaging dataset']['splits path']
num_splits = cfg['number of runs']
model_name = cfg['model name']
def_params = cfg['hyperparameters']['end_to_end']
tuning_params = cfg['tuning parameters']['end_to_end']
performance_scores = cfg['final performance measures']
save_models = cfg['save options']['models path']
save_params = cfg['save options']['params path']
save_scores = cfg['save options']['scores path']
save_figures = cfg['save options']['figures path']

img_data = imaging_dataset(dataset_name)
img_sets = img_data.assign_train_val_test_sets(img_splits_path)

clin_data = clinical_dataset(dataset_name)
clin_sets = clin_data.assign_train_val_test_sets(clin_splits_path)

for i in range(num_splits):
	#### ASSIGN TRAINING, TEST AND VALIDATION SETS FOR CURRENT SPLIT ##########################################
	current_split_num = i+1

	img_X_tr = img_sets[i]['train_data']
	img_X_val = img_sets[i]['val_data']
	img_X_te = img_sets[i]['test_data']

	clin_X_tr = clin_sets[i]['train_data']
	clin_X_val = clin_sets[i]['val_data']
	clin_X_te = clin_sets[i]['test_data']

	y_tr = img_sets[i]['train_labels']
	y_val = img_sets[i]['val_labels']
	y_te = img_sets[i]['test_labels']

	if def_params['out_activation'] == 'softmax':
		y_tr = pd.get_dummies(y_tr)
		y_val = pd.get_dummies(y_val)
		y_te = pd.get_dummies(y_te)

	#### START GRID SEARCH #####################################################################################
	start_grid = time.time()
	comb = 1

	for tune in ParameterGrid(tuning_params):

		for j in range(2):
			model_path = 'final_gridsearch/models/split_'+str(current_split_num)+'/trained_model_params_comb_'+str(comb)+'_test_'+str(j)+'.h5'
			if os.path.isfile(model_path):
				pass

			else:		
				try:

					img_input = Input(shape= (156,192,64,1), name='image_input')
					clin_input = Input(shape= (clin_X_tr.shape[1],), name='clinical_input')

					x1 = Conv3D(tune['num_filters'][0], (3,3,3), strides = (1,1,1),padding="same",kernel_regularizer= keras.regularizers.l2(tune['l2_ratio']))(img_input)
					x1 = Activation(def_params['hidden_activation'])(x1)
					x1 = MaxPooling3D(pool_size=(3,3,3))(x1)

					x1 = Conv3D(tune['num_filters'][1],(3,3,3), strides = (1,1,1),padding="same",kernel_regularizer= keras.regularizers.l2(tune['l2_ratio']))(x1)
					x1 = Activation(def_params['hidden_activation'])(x1)
					x1 = MaxPooling3D(pool_size=(3,3,3))(x1)

					x1 = Conv3D(tune['num_filters'][2], (3,3,3), strides = (1,1,1),padding="same",kernel_regularizer= keras.regularizers.l2(tune['l2_ratio']))(x1)
					x1 = Activation(def_params['hidden_activation'])(x1)
					x1 = MaxPooling3D(pool_size=(3,3,3))(x1)

					x1 = Flatten()(x1)
					x1 = Dense(tune['num_filters'][2]*2, activation=def_params['hidden_activation'],kernel_regularizer= keras.regularizers.l2(tune['l2_ratio']))(x1)
					x1 = Dropout(tune['dropout_rate'])(x1)
					x1 = Dense(tune['num_neurons_embedding'][1], activation=def_params['hidden_activation'],kernel_regularizer= keras.regularizers.l2(tune['l2_ratio']))(x1)



					x2 = Dense(tune['num_neurons_MLP'], activation = def_params['hidden_activation'], kernel_regularizer= keras.regularizers.l2(tune['l2_ratio']))(clin_input)
					x2 = Dropout(tune['dropout_rate'])(x2)
					x2 = Dense(tune['num_neurons_embedding'][0], activation=def_params['hidden_activation'],kernel_regularizer= keras.regularizers.l2(tune['l2_ratio']))(x2)

					x = concatenate([x1, x2])
					x = Dense(tune['num_neurons_final'],  activation = def_params['hidden_activation'], kernel_regularizer= keras.regularizers.l1(tune['l2_ratio']))(x)
					x = Dropout(tune['dropout_rate'])(x)

					if def_params['out_activation'] == 'softmax':
						output = Dense(2,activation= def_params['out_activation'], kernel_regularizer= keras.regularizers.l2(tune['l2_ratio']))(x)
					else: 
						output = Dense(1,activation= def_params['out_activation'], kernel_regularizer= keras.regularizers.l2(tune['l2_ratio']))(x)

					optimizer = keras.optimizers.Adam(lr = tune['learning_rate'])

					model = Model(inputs=[img_input, clin_input], outputs=[output])
					model.compile(loss=def_params['loss_func'], optimizer = optimizer)

					parallel_model = multi_gpu_model(model, 2)
					parallel_model.compile(loss=def_params['loss_func'],optimizer=optimizer)

					e_stop = EarlyStopping(monitor = 'val_loss', min_delta = def_params['min_delta'], patience = def_params['iter_patience'], mode='auto')
					callbacks = [e_stop]

					print('Training for comb %i with parameters %s'%(comb,str(tune)))

					start = time.time()
					history = model.fit({'image_input' : img_X_tr,'clinical_input' : clin_X_tr}, y_tr, callbacks = callbacks,validation_data= ([img_X_val, clin_X_val],y_val), 
						epochs=def_params['epochs'], batch_size= tune['batch_size'], verbose=0)
					end = time.time()

					print('Training completed in around %i minutes'%(np.floor((end-start)/60)))

					model.set_weights(parallel_model.get_weights())

					models_path = 'final_gridsearch/models/split_'+str(current_split_num)+'/'
					if not os.path.exists(models_path):
						os.makedirs(models_path)
					model.save(models_path+ 'trained_model_params_comb_'+str(comb)+'_test_'+str(j)+'.h5')

					loss = history.history['loss']
					val_loss = history.history['val_loss']

					probs_tr = model.predict([img_X_tr,clin_X_tr],batch_size = 8)
					probs_val = model.predict([img_X_val,clin_X_val],batch_size = 8)
					probs_te = model.predict([img_X_te,clin_X_te],batch_size = 8)

					score_tr = roc_auc_score(y_tr, probs_tr)
					score_val = roc_auc_score(y_val, probs_val)
					score_te = roc_auc_score(y_te, probs_te)

					scores_path = 'final_gridsearch/scores/split_'+str(current_split_num)+'/'
					if not os.path.exists(scores_path):
						os.makedirs(scores_path)
					np.savetxt(scores_path+'loss_over_epochs_params_comb_'+str(comb)+'_test_'+str(j)+'.csv', [loss,val_loss], delimiter=",")
					np.savetxt(scores_path+ 'auc_scores_params_comb_'+str(comb)+'_test_'+str(j)+'.csv', [score_tr,score_val,score_te], delimiter= ",")

					del model

				except ValueError:
					pass

			keras.backend.clear_session()


		comb +=1

	end_grid = time.time()

	print('Whole training completed in %i hours %i minutes.'%(np.floor((end_grid-start_grid)/3600), np.floor(((end_grid-start_grid)%3600)/60)))
