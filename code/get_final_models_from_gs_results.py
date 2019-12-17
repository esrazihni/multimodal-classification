"""
File name: get_final_models_from_gs_results.py
Author: Esra Zihni
Date created: 27.05.2019


This file finds the best trained models on a given split and returns the parameter combinations that give best 
validation AUC results.
"""

import numpy as np
import os
import yaml 
import json
import pandas as pd
import keras 
import tensorflow as tf 
from keras.models import load_model
from sklearn.model_selection import ParameterGrid

#### ENVIRONMENT AND SESSION SET UP ####################################################################
# set the environment variable
os.environ["KERAS_BACKEND"] = "tensorflow"
# Silence INFO logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


#### READ CONFIGURATION FILE ###########################################################################
def join(loader,node):
    seq = loader.construct_sequence(node)
    return ''.join(str(i) for i in seq)

yaml.add_constructor('!join',join)
cfg = yaml.load(open('config.yml', 'r'))


#### ASSIGN PATHS AND VARIABLES #########################################################################
dataset_name = cfg['dataset name']
num_splits = cfg['number of runs']
model_name = cfg['model name']
def_params = cfg['hyperparameters'][model_name]
tuning_params = cfg['tuning parameters'][model_name]
performance_scores = cfg['final performance measures']
save_models = cfg['save options']['models path']
save_params = cfg['save options']['params path']
save_scores = cfg['save options']['scores path']
save_figures = cfg['save options']['figures path']

tmp_scores = np.zeros((num_splits,3))

for current_split_num in range(num_splits):


	#### READ VALIDATION AUC SCORES OF ALL TRAINED MODELS FROM GRID SEARCH ##################################
	comb = 1

	#val_aucs_all_comb = []
	#test_aucs_all_comb = []
	ref_AUC = 0.5

	for tune in ParameterGrid(tuning_params):
		val_aucs_single_comb = np.zeros(2)
		test_aucs_single_comb = np.zeros(2)
		train_aucs_single_comb = np.zeros(2)
		for j in range(2):
			score_path = 'final_gridsearch/scores/split_'+str(current_split_num+1)+'/auc_scores_params_comb_'+str(comb)+'_test_'+str(j)+'.csv'
			if os.path.isfile(score_path):
				scores = np.loadtxt(score_path, delimiter = ',')
				val_aucs_single_comb[j] = scores[1]
				test_aucs_single_comb[j] = scores[2]
				train_aucs_single_comb[j] = scores[0]


		mean_AUC_val = val_aucs_single_comb.mean()
		if mean_AUC_val > ref_AUC:
			ref_AUC = mean_AUC_val
			best_comb_num = comb
			best_tuning_params = tune
			AUC_tr = train_aucs_single_comb.mean()
			AUC_te = test_aucs_single_comb.mean()


		comb +=1

	path_to_params = save_params+ '/best_tuning_params_split_'+str(current_split_num+1)+'.json'
	if not os.path.exists(save_params):
		os.makedirs(save_params)
	json.dump(best_tuning_params, open(path_to_params, 'w'))

	print(AUC_tr)
	print(AUC_te)
	print(ref_AUC)
	print(best_comb_num)

	tmp_scores[current_split_num,0] = AUC_tr
	tmp_scores[current_split_num,1] = ref_AUC
	tmp_scores[current_split_num,2] = AUC_te

	if not os.path.exists(save_models):
		os.makedirs(save_models)
	best_model_inner_loop = load_model('final_gridsearch/models/split_'+str(current_split_num+1)+'/trained_model_params_comb_'+str(best_comb_num)+'_test_1.h5')
	best_model_inner_loop.save(save_models+ '/best_model_on_inner_training_set_split_'+str(current_split_num+1)+'.h5')
	
	#val_aucs_all_comb = np. asarray(val_aucs_all_comb)
	#test_aucs_all_comb = np. asarray(test_aucs_all_comb)

	#print(np.sort(val_aucs_all_comb)[-10:])
	#print(np.argsort(val_aucs_all_comb)[-10:]+1)


if not os.path.exists(save_scores):
	os.makedirs(save_scores)

df_scores = pd.DataFrame(tmp_scores, columns=['training AUC','validation AUC','test AUC'])
df_scores.to_csv(save_scores+ '/inner_loop_AUC_performance_all_splits.csv')
