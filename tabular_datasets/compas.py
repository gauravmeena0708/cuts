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

        self.binary_race = binary_race

        if True:
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

            # Gold data already has the columns we want? 
            # Migration used raw `compas-scores-two-years.csv`.
            # The original loader did heavy preprocessing (filtering days, dropping cols).
            # If migration just copied raw, we need to repeat preprocessing here or move preprocessing to migration.
            # Ideally data/processed should be CLEAN.
            # But since I already ran migration as raw copy-paste, I should apply preprocessing here on the loaded DF.
            
            def preprocess(df):
                # Apply same filters as original
                df = df[df['days_b_screening_arrest'] >= -30]
                df = df[df['days_b_screening_arrest'] <= 30]
                df = df[df['is_recid'] != -1]
                df = df[df['c_charge_degree'] != '0']
                df = df[df['score_text'] != 'N/A']

                df['in_custody'] = pd.to_datetime(df['in_custody'])
                df['out_custody'] = pd.to_datetime(df['out_custody'])
                df['diff_custody'] = (df['out_custody'] - df['in_custody']).dt.total_seconds()
                df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
                df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
                df['diff_jail'] = (df['c_jail_out'] - df['c_jail_in']).dt.total_seconds()
                
                if self.binary_race:
                    df = df[df['race'].isin(['African-American', 'Caucasian'])]
                
                # Drop cols
                cols_to_drop = [
                    'id', 'name', 'first', 'last', 'v_screening_date', 'compas_screening_date', 'dob', 'c_case_number',
                    'screening_date', 'in_custody', 'out_custody', 'c_jail_in', 'c_jail_out',
                    'is_recid', 'is_violent_recid', 'violent_recid', 'two_year_recid'
                ]
                # Only drop if exist
                cols_to_drop = [c for c in cols_to_drop if c in df.columns]
                
                y = 1 - df['two_year_recid']
                x = df.drop(cols_to_drop, axis=1)
                
                # Keep only specific features
                x = x[[
                    'age', 'sex', 'race', 'diff_custody', 'diff_jail', 'priors_count', 'c_charge_degree', 'v_score_text'
                ]]
                
                x['two_year_recid'] = y.astype(int) 
                return x

            train_df = preprocess(train_df)
            test_df = preprocess(test_df)

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
        
        :param binary_race: (bool) Toggle if the race feature only contains two possibilities.
        :return: (dict) The features.
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
        """
        As the dataset does not come with a standard train-test split, we assign this split manually during the
        initialization. To allow for independent experiments without much of a hassle, we allow through this method for
        a reassignment of the split.

        :param split_ratio: (float) The desired ratio of test_data/all_data.
        :param random_state: (int) The random state according to which we do the assignment,
        :return: None
        """
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
