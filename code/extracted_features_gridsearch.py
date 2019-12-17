"""
File name: extracted_features_gridsearch.py
Author: Esra Zihni
Date created: 29.04.2019



"""

import numpy as np
import sys
import os
import yaml
import pickle
import pandas as pd
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes
import json
import time
import keras
import tensorflow as tf
from keras.models import load_model,Sequential, Model
from keras.layers import Dense, Dropout, Input, concatenate
from keras.callbacks import EarlyStopping
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score

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
data_path = 'data/'
clin_feat_splits_path = data_path+ cfg['clinical dataset']['feature splits path']
img_feat_splits_path = data_path + cfg['imaging dataset']['feature splits path']
num_splits = cfg['number of runs']
model_name = cfg['model name']
def_params = cfg['hyperparameters'][model_name]
tuning_params = cfg['tuning parameters'][model_name]
performance_scores = cfg['final performance measures']
save_models = cfg['save options']['models path']
save_params = cfg['save options']['params path']
save_scores = cfg['save options']['scores path']
save_figures = cfg['save options']['figures path']

##### GET TRAINING,VALIDATION AND TEST DATA ##############################################################
data = multimodal_dataset(dataset_name)
data.load_feature_sets(img_feat_splits_path, clin_feat_splits_path)
#feature_sets = data.combine_features(combining_method = 'concat_and_normalize')


##### TRAIN AND SAVE MODELS #################################################################################
for i in range(num_splits):
	#### ASSIGN TRAINING, TEST AND VALIDATION SETS FOR CURRENT SPLIT ##########################################
	current_split_num = i+1

	img_X_tr = data.img_sets[i]['train_data']
	img_X_val = data.img_sets[i]['val_data']
	img_X_te = data.img_sets[i]['test_data']

	clin_X_tr = data.clin_sets[i]['train_data']
	clin_X_val = data.clin_sets[i]['val_data']
	clin_X_te = data.clin_sets[i]['test_data']

	y_tr = data.img_sets[i]['train_labels']
	y_val = data.img_sets[i]['val_labels']
	y_te = data.img_sets[i]['test_labels']

	if def_params['out_activation'] == 'softmax':
		y_tr = pd.get_dummies(y_tr)
		y_val = pd.get_dummies(y_val)
		y_te = pd.get_dummies(y_te)


	model_path = save_models + '/best_model_on_outer_training_set_split_'+str(current_split_num)+'.h5'
	#params_path = save_params + '/best_parameters_run_'+str(current_split_num)+'.json'
	tune_params_path = save_params + '/best_tuning_parameters_split_'+str(current_split_num)+'.json'


	if os.path.isfile(model_path):
		pass
	else:
		if not os.path.exists(save_models):
			os.makedirs(save_models)
		#### START GRID SEARCH #####################################################################################
		start = time.time()

		best_AUC = 0.5

		i = 0

		for tune in ParameterGrid(tuning_params):
			img_input = Input(shape= (img_X_tr.shape[1],), name='image_input')
			clin_input = Input(shape= (clin_X_tr.shape[1],), name='clinical_input')

			dense1 = Dense(tune['num_neurons_embedding'][0], kernel_initializer = def_params['weight_init'], activation = def_params['hidden_activation'], 
							kernel_regularizer= keras.regularizers.l2(tune['l2_ratio']))(clin_input)

			dense2 = Dense(tune['num_neurons_embedding'][1], kernel_initializer = def_params['weight_init'], activation = def_params['hidden_activation'], 
							kernel_regularizer= keras.regularizers.l2(tune['l2_ratio']))(img_input)

			x = concatenate([dense1, dense2])
			x = Dense(tune['num_neurons_final'], kernel_initializer = def_params['weight_init'], activation = def_params['hidden_activation'], 
						kernel_regularizer= keras.regularizers.l2(tune['l2_ratio']))(x)
			x= Dropout(tune['dropout_rate'])(x)

			if def_params['out_activation'] == 'softmax':
				output = Dense(2,kernel_initializer = def_params['weight_init'],activation= def_params['out_activation'], 
								kernel_regularizer= keras.regularizers.l2(tune['l2_ratio']))(x)
			else: 
				output = Dense(1,kernel_initializer = def_params['weight_init'],activation= def_params['out_activation'], 
								kernel_regularizer= keras.regularizers.l2(tune['l2_ratio']))(x)

			optimizer = keras.optimizers.Adam(lr = tune['learning_rate'])

			model = Model(inputs=[img_input, clin_input], outputs=[output])
			model.compile(loss=def_params['loss_func'], optimizer = optimizer)

			e_stop = EarlyStopping(monitor = 'val_loss', min_delta = def_params['min_delta'], patience = def_params['iter_patience'], mode='auto')
			callbacks = [e_stop]
			history = model.fit({'image_input' : img_X_tr,'clinical_input' : clin_X_tr}, y_tr, callbacks = callbacks,validation_data= ([img_X_val, clin_X_val],y_val), 
								epochs=def_params['epochs'], batch_size= tune['batch_size'], verbose=0)

			probs_val = model.predict([img_X_val,clin_X_val],batch_size = 8)
			score_val = roc_auc_score(y_val, probs_val)

			i +=1
			if i%10 == 0:
				print(i)

			if score_val > best_AUC:
				best_AUC = score_val
				best_params = tune   
				loss_tr = history.history['loss']
				loss_val = history.history['val_loss']   
				model.save(save_models + '/best_model_on_inner_training_set_split_'+str(current_split_num)+'.h5')

			keras.backend.clear_session()

		best_model = load_model(save_models + '/best_model_on_inner_training_set_split_'+str(current_split_num)+'.h5')
		probs_tr = best_model.predict([img_X_tr,clin_X_tr],batch_size = 8)
		probs_val = best_model.predict([img_X_val,clin_X_val],batch_size = 8)
		probs_te = best_model.predict([img_X_te,clin_X_te],batch_size = 8)

		score_tr = roc_auc_score(y_tr, probs_tr)
		score_val = roc_auc_score(y_val, probs_val)
		score_te = roc_auc_score(y_te, probs_te)


		# Save tuning parameters that resulted in the best model:
		if not os.path.exists(save_params):
			os.makedirs(save_params)
		json.dump(best_params,open(tune_params_path,'w'))

		# Save loss and auc scores calculated at each epoch during training:
		if not os.path.exists(save_scores):
			os.makedirs(save_scores)

		np.savetxt(save_scores+'/inner_loop_loss_over_epochs_split_'+str(current_split_num)+'.csv', [loss_tr,loss_val], delimiter=",")
		np.savetxt(save_scores+ "/inner_loop_auc_scores_split_"+str(current_split_num)+".csv", [score_tr, score_val, score_te], delimiter=",")

		end = time.time()

		print('Training time for split %s: %i minutes.'%(str(current_split_num),np.floor(((end-start)%3600)/60)))


	