import torch
import numpy as np
import pandas as pd
from .base_dataset import BaseDataset
import sys
import os

sys.path.append('..')
from utils import to_numeric
from sklearn.model_selection import train_test_split


class Compas(BaseDataset):

    def __init__(self, name='Compas', train_test_ratio=0.2, binary_race=False, single_bit_binary=False, device='cpu', 
                 random_state=42, split_from_file=True):
        super().__init__(name=name, device=device, random_state=random_state)

        self.train_test_ratio = train_test_ratio
        self.binary_race = binary_race
        self.features = Compas.get_features(binary_race=binary_race)

        self.single_bit_binary = single_bit_binary
        self.label = 'two_year_recid'

        self.train_features = {key: self.features[key] for key in self.features.keys() if key != self.label}

        # Prefer CUTS_TRAIN_CSV / CUTS_TEST_CSV env vars (set by wrapper),
        # fall back to gold path for standalone use.
        cuts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(cuts_dir)
        sd_framework_root = os.path.dirname(project_root)
        gold_dir = os.path.join(sd_framework_root, 'data', 'gold', 'compas')
        train_path = os.environ.get('CUTS_TRAIN_CSV') or os.path.join(gold_dir, 'train.csv')
        test_path = os.environ.get('CUTS_TEST_CSV') or os.path.join(gold_dir, 'test.csv')

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Pre-processed data already has columns: 
        # age,sex,race,diff_custody,diff_jail,priors_count,c_charge_degree,v_score_text,two_year_recid
        
        # Ensure column order matches get_features keys
        ordered_cols = list(self.features.keys())
        train_df = train_df[ordered_cols]
        test_df = test_df[ordered_cols]

        # convert to numeric
        train_data = train_df.to_numpy()
        test_data = test_df.to_numpy()
        
        train_data_num = (to_numeric(train_data, self.features, label=self.label, single_bit_binary=self.single_bit_binary)).astype(np.float32)
        test_data_num = (to_numeric(test_data, self.features, label=self.label, single_bit_binary=self.single_bit_binary)).astype(np.float32)

        # split labels and features
        Xtrain, ytrain = train_data_num[:, :-1], train_data_num[:, -1]
        Xtest, ytest = test_data_num[:, :-1], test_data_num[:, -1]
        
        self.num_features = Xtrain.shape[1]

        # convert to torch
        self.Xtrain, self.Xtest = torch.tensor(Xtrain).to(self.device), torch.tensor(Xtest).to(self.device)
        self.ytrain, self.ytest = torch.tensor(ytrain, dtype=torch.long).to(self.device), torch.tensor(ytest, dtype=torch.long).to(self.device)

        # set to train mode as base
        self.train()

        # calculate the standardization statistics
        self._calculate_mean_std()

        # calculate the mins and the maxs
        self._calculate_mins_maxs()

        # calculate the histograms and feature bounds
        self._calculate_categorical_feature_distributions_and_continuous_bounds()

        # fill the feature domain lists
        self.create_feature_domain_lists()

    @staticmethod
    def get_features(binary_race=False):
        """
        Static method such that we can access the features of the dataset without having to instantiate it.
        """
        features = {
            'age': None, 
            'sex': ['Male', 'Female'], 
            'race': ['Other', 'African-American', 'Caucasian', 'Hispanic', 'Asian', 'Native American'], 
            'diff_custody': None, 
            'diff_jail': None, 
            'priors_count': None, 
            'c_charge_degree': ['F', 'M'], 
            'v_score_text': ['Low', 'High', 'Medium'],
            'two_year_recid': [0, 1]
        }

        if binary_race:
            features['race'] = ['African-American', 'Caucasian']
       
        return features
    
    def repeat_split(self, split_ratio=None, random_state=None):
        if random_state is None:
            random_state = self.random_state
        if split_ratio is None:
            split_ratio = self.train_test_ratio
        X = torch.cat([self.Xtrain, self.Xtest], dim=0).detach().cpu().numpy()
        y = torch.cat([self.ytrain, self.ytest], dim=0).detach().cpu().numpy()
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=split_ratio, random_state=random_state,
                                                        shuffle=True)
        # convert to torch
        self.Xtrain, self.Xtest = torch.tensor(Xtrain).to(self.device), torch.tensor(Xtest).to(self.device)
        self.ytrain, self.ytest = torch.tensor(ytrain, dtype=torch.long).to(self.device), torch.tensor(ytest, dtype=torch.long).to(self.device)
        # update the split status as well
        self._assign_split(self.split_status)
