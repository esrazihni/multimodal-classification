"""
Created on Mon Apr 29 2018


"""

import numpy as np
import pickle
import time
import os
import pandas as pd
import sys
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes

from sklearn import preprocessing

sys.path.append("/home/esra/Desktop/git_repos/outcome-prediction/clinical parameter-based")
from helper import dataset, model




class multimodal_dataset(dataset):
    def __init__(self,name):
        super().__init__(name)
        self.datatype = 'multimodal'

    def load_feature_sets(self,img_datasource, clin_datasource):
        self.img_sets = pickle.load(open(img_datasource,'rb'))
        self.clin_sets = pickle.load(open(clin_datasource,'rb'))


    #def preprocess(self):
        #feature_tr = preprocessing.StandardScaler().fit_transform(feature_tr)
        #feature_val = preprocessing.StandardScaler().fit_transform(feature_val)
        #feature_te = preprocessing.StandardScaler().fit_transform(feature_te)



#lass end_to_end_multimodal(model):
#	def train(self):
