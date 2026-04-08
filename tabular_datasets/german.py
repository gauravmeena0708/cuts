import torch
import numpy as np
import pandas as pd
from .base_dataset import BaseDataset
import sys
import os

sys.path.append("..")
from utils import to_numeric
from sklearn.model_selection import train_test_split


class German(BaseDataset):

    def __init__(self, name='German', train_test_ratio=0.2, single_bit_binary=False, device='cpu', random_state=42, split_from_file=True):
        super(German, self).__init__(name=name, device=device, random_state=random_state)

        self.train_test_ratio = train_test_ratio

        self.features = German.get_features()

        self.single_bit_binary = single_bit_binary
        self.label = 'class'

        self.train_features = {key: self.features[key] for key in self.features.keys() if key != self.label}

        if True: # Always load from file for consistency now
            # Prefer CUTS_TRAIN_CSV / CUTS_TEST_CSV env vars (set by wrapper),
            # fall back to gold path for standalone use.
            cuts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            project_root = os.path.dirname(cuts_dir)
            sd_framework_root = os.path.dirname(project_root)
            gold_dir = os.path.join(sd_framework_root, 'data', 'gold', 'german')
            train_path = os.environ.get('CUTS_TRAIN_CSV') or os.path.join(gold_dir, 'train.csv')
            test_path = os.environ.get('CUTS_TEST_CSV') or os.path.join(gold_dir, 'test.csv')

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Map columns if necessary (migration used full names, but get_features uses A1..A20?)
            # The migration script used names:
            # ['status', 'duration', 'history', 'purpose', 'amount', 'savings', 'employment', 
            #  'installment_rate', 'personal_status', 'debtors', 'residence', 'property', 
            #  'age', 'plans', 'housing', 'existing_credits', 'job', 'maintenance', 
            #  'telephone', 'foreign', 'target']
            
            # We need to ensure get_features matches these names OR rename here.
            # Let's rename the dataframe columns to match the legacy A-scheme if we want to keep get_features or update get_features.
            # Updating get_features is better for readability, but more work.
            # Let's map dataframe columns to A1..A20 to match existing get_features logic for now to minimize breakage?
            # Actually, user wants "German Credit", the prompt implies moving towards better support.
            # But changing feature names breaks existing checkpoints potentially.
            # However we are in a refactor.
            # Let's keep A1..A20 mapping for now but safer to load by position or update get_features?
            # If we used `pd.read_csv` in migration with `names=columns`, the CSV has header `status, duration...`.
            
            # Let's update `get_features` in a separate call or here. If I change here, I must change get_features.
            # I will assume we map back for now to minimize diff, or better, just rely on column order?
            # get_features uses keys for dict lookups.
            
            # Let's just update the loader to use the columns from the file and we will update get_features in next step.
            
            # Actually, `to_numeric` relies on `self.features` keys.
            # So I should update `get_features` to match the GOLD dataset names.
            
            # BUT, I can't update get_features in this single block easily if it is static.
            # I'll update the logic here to use the GOLD files, and then do a second pass to update `get_features`.
            
            # Wait, `migration` script used explicit names.
            # So `train_df` has columns "status", "duration", etc.
            # `self.features` expects "A1", "A2".
            # `to_numeric` will look for "A1" in `train_df` and fail.
            
            # So I MUST map `train_df` columns to `AX` OR update `get_features`.
            # Updating `get_features` is cleaner.
            pass
            
            # I will actually replace the whole file or large chunks.
            # Let's do a multi-replace if possible or just replace the loader part and I will immediately follow up with feature update.
            # Or better, just mapping here:
            
            column_mapping = {
                'status': 'A1', 'duration': 'A2', 'history': 'A3', 'purpose': 'A4', 
                'amount': 'A5', 'savings': 'A6', 'employment': 'A7', 'installment_rate': 'A8',
                'personal_status': 'A9', 'debtors': 'A10', 'residence': 'A11', 'property': 'A12',
                'age': 'A13', 'plans': 'A14', 'housing': 'A15', 'existing_credits': 'A16',
                'job': 'A17', 'maintenance': 'A18', 'telephone': 'A19', 'foreign': 'A20',
                'target': 'class'
            }
            train_df = train_df.rename(columns=column_mapping)
            test_df = test_df.rename(columns=column_mapping)
            
            train_data = train_df.to_numpy()
            test_data = test_df.to_numpy()

            data_num = (to_numeric(train_data, self.features, label=self.label, single_bit_binary=self.single_bit_binary)).astype(np.float32)
            test_data_num = (to_numeric(test_data, self.features, label=self.label, single_bit_binary=self.single_bit_binary)).astype(np.float32)

            # split labels and features
            Xtrain, ytrain = data_num[:, :-1], data_num[:, -1]
            Xtest, ytest = test_data_num[:, :-1], test_data_num[:, -1] # test has same format
            
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

    @staticmethod
    def get_features():
        """
        Static method such that we can access the features of the dataset without having to instantiate it.

        :return: (dict) The features.
        """
        features = {
            'A1': ['A1' + str(i) for i in range(1, 5)],  # status of existing checking account
            'A2': None,  # duration
            'A3': ['A3' + str(i) for i in range(0, 5)],  # credit history
            'A4': ['A4' + str(i) for i in range(0, 11)],  # purpose
            'A5': None,  # credit amount
            'A6': ['A6' + str(i) for i in range(1, 6)],  # savings account/bonds
            'A7': ['A7' + str(i) for i in range(1, 6)],  # present employment since
            'A8': None,  # installment rate in percentage of dispsable income
            'A9': ['A9' + str(i) for i in range(1, 6)],  # personal status and sex
            'A10': ['A10' + str(i) for i in range(1, 4)],  # other debtors / guarantors
            'A11': None,  # present residence since
            'A12': ['A12' + str(i) for i in range(1, 5)],  # property
            'A13': None,  # age
            'A14': ['A14' + str(i) for i in range(1, 4)],  # other installment plans
            'A15': ['A15' + str(i) for i in range(1, 4)],  # housing
            'A16': None,  # number of existing credits at this bank
            'A17': ['A17' + str(i) for i in range(1, 5)],  # job
            'A18': None,  # number of people being liable to provide maintanance for
            'A19': ['A19' + str(i) for i in range(1, 3)],  # telephone
            'A20': ['A20' + str(i) for i in range(1, 3)],  # foreign worker
            'class': [1, 2]  # credit risk good or bad
        }
       
        return features