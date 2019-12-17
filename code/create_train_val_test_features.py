"""
File name: crear_train_val_test_features.py
Author: Esra Zihni
Date created: 24.04.2019



"""

import numpy as np
import sys
import os
import yaml
import pickle
import keras
import tensorflow as tf
from keras.models import load_model, Model 
from keras.backend.tensorflow_backend import set_session

from helper import dataset, model
from imaging_predictive_models import imaging_dataset
from clinical_predictive_models import clinical_dataset


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
clin_splits_path = cfg['clinical dataset']['splits path']
clin_models_path = cfg['clinical dataset']['models path']
img_splits_path = cfg['imaging dataset']['splits path']
img_models_path = cfg['imaging dataset']['models path']
num_runs = cfg['number of runs']


#### LOAD TRAINING, VALIDATION AND TEST SETS ############################################################
img_data = imaging_dataset(dataset_name)
img_sets = img_data.assign_train_val_test_sets(img_splits_path)

clin_data = clinical_dataset(dataset_name)
clin_sets = clin_data.assign_train_val_test_sets(clin_splits_path)

# create empty list to store feature sets
clin_feature_sets = []
img_feature_sets = []

#### EXTRACT FEATURES FROM TRAINED MODELS ###############################################################

for i in range(num_runs):
	# load best imaging and clinical models
	img_model_obj = model('CNN',img_sets[i],default_params={'out_activation':'sigmoid'})
	img_model_obj.best_model = load_model(img_models_path+ 'best_model_on_inner_training_set_split_'+str(i+1)+'.h5')
	clin_model_obj = model('MLP',clin_sets[i],default_params={'out_activation':'sigmoid'})
	clin_model_obj.best_model = load_model(clin_models_path+ 'best_MLP_multimodal_model_on_inner_training_set_split_'+str(i+1)+'.h5')


	# get layer names (since it may be different for each model):
	CNN_layer_names = [layer.name for layer in img_model_obj.best_model.layers]
	MLP_layer_names = [layer.name for layer in clin_model_obj.best_model.layers]
	#print(CNN_layer_names)
	#print(MLP_layer_names)

	# build pipeline to get the one before last layer output given an input

	intermediate_layer_model = Model(inputs= img_model_obj.best_model.input, outputs = img_model_obj.best_model.get_layer(CNN_layer_names[-2]).output)
	img_intermediate_output_tr = intermediate_layer_model.predict(img_model_obj.X_tr, batch_size=8)
	img_intermediate_output_val = intermediate_layer_model.predict(img_model_obj.X_val, batch_size=8)
	img_intermediate_output_te = intermediate_layer_model.predict(img_model_obj.X_te, batch_size=8)
	print(img_intermediate_output_tr.shape)
	#print(img_intermediate_output_val.shape)
	#print(img_intermediate_output_te.shape)
	#print(img_intermediate_output_tr.mean(axis=0))
	#print(img_intermediate_output_tr.std(axis=0))


	intermediate_layer_model = Model(inputs= clin_model_obj.best_model.input, outputs = clin_model_obj.best_model.get_layer(MLP_layer_names[1]).output)
	clin_intermediate_output_tr = intermediate_layer_model.predict(clin_model_obj.X_tr, batch_size=8)
	clin_intermediate_output_val = intermediate_layer_model.predict(clin_model_obj.X_val, batch_size=8)
	clin_intermediate_output_te = intermediate_layer_model.predict(clin_model_obj.X_te, batch_size=8)
	print(clin_intermediate_output_tr.shape)
	#print(clin_intermediate_output_val.shape)
	#print(clin_intermediate_output_te.shape)
	#print(clin_intermediate_output_tr.mean(axis=0))
	#print(clin_intermediate_output_tr.std(axis=0))


	clin_tmp_set = dict.fromkeys(['train_data','train_labels','val_data','val_labels','test_data','test_labels'])

	clin_tmp_set['train_data'] = clin_intermediate_output_tr
	clin_tmp_set['train_labels'] = clin_model_obj.y_tr
	clin_tmp_set['val_data'] = clin_intermediate_output_val
	clin_tmp_set['val_labels'] = clin_model_obj.y_val
	clin_tmp_set['test_data'] = clin_intermediate_output_te
	clin_tmp_set['test_labels'] = clin_model_obj.y_te

	clin_feature_sets.append(clin_tmp_set)

	img_tmp_set = dict.fromkeys(['train_data','train_labels','val_data','val_labels','test_data','test_labels'])

	img_tmp_set['train_data'] = img_intermediate_output_tr
	img_tmp_set['train_labels'] = img_model_obj.y_tr
	img_tmp_set['val_data'] = img_intermediate_output_val
	img_tmp_set['val_labels'] = img_model_obj.y_val
	img_tmp_set['test_data'] = img_intermediate_output_te
	img_tmp_set['test_labels'] = img_model_obj.y_te

	img_feature_sets.append(img_tmp_set)


#### ASSIGN PATH AND SAVE FEATURES ######################################################################### 
folder_to_save = 'data/'
if not os.path.exists(folder_to_save):
	os.makedirs(folder_to_save)

clin_feat_splits_path = cfg['clinical dataset']['feature splits path']
img_feat_splits_path = cfg['imaging dataset']['feature splits path']

pickle.dump(clin_feature_sets, open(folder_to_save+clin_feat_splits_path, 'wb'))
pickle.dump(img_feature_sets, open(folder_to_save+img_feat_splits_path, 'wb'))
