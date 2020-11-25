import os
import copy
import random
import pickle

from src.ehr_utils import ehr_from_dataframe
from src.yc_utils import yc_from_dataframe

class Featureset:
    def __init__(self, base_dir, data_config=None):
        self.base_dir = base_dir
        self.features = None
        self.config = data_config
        self.train_features = None
        self.test_features = None
        

    def load_ehr(self, csvfile):
        csvpath = os.path.join(self.base_dir, csvfile)
        self.features, self.config = ehr_from_dataframe(csvpath, self.config)
        
    def load_yc(self, csvuser, csvitem, csvtarget):
        csvpaths = [os.path.join(self.base_dir, csv) for csv in [csvuser, csvitem, csvtarget]]
        self.features, self.config = yc_from_dataframe(*csvpaths, self.config)
        
    def split_train_test(self, train_split):
        features = copy.deepcopy(self.features)
        ntrain = int(len(features)*train_split)
        random.shuffle(features)
        self.train_features, self.test_features = features[:ntrain], features[ntrain:]
    
    def create_new_dataset(self, train_split, tag):
        if self.train_features is None:
            self.split_train_test(train_split)
        
        for d in ['train_features', 'test_features', 'config']:
            with open(os.path.join(self.base_dir, '{}_{}.pkl'.format(d, tag)), "wb") as f:
                pickle.dump(self.__dict__[d], f)
                
    def load_dataset(self, tag):
        for d in ['train_features', 'test_features', 'config']:
            with open(os.path.join(self.base_dir, '{}_{}.pkl'.format(d, tag)), "rb") as f:
                self.__dict__.update({d:pickle.load(f)})