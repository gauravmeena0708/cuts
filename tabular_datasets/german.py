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
        
        # Ensure column names match what get_features expects
        # The processed data in workdir/data/processed/german/train.csv has these names:
        # status,duration,credit_history,purpose,credit_amount,savings,employment,installment_rate,
        # personal_status,debtors,residence_since,property,age,installment_plans,housing,
        # existing_credits,job,maintenance_people,telephone,foreign_worker,class
        
        # If the file names differ from the get_features keys, we rename them here.
        # Here they should already match if we use the names from get_features().
        
        train_data = train_df.to_numpy()
        test_data = test_df.to_numpy()

        data_num = (to_numeric(train_data, self.features, label=self.label, single_bit_binary=self.single_bit_binary)).astype(np.float32)
        test_data_num = (to_numeric(test_data, self.features, label=self.label, single_bit_binary=self.single_bit_binary)).astype(np.float32)

        # split labels and features
        Xtrain, ytrain = data_num[:, :-1], data_num[:, -1]
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
        Values correspond to the Statlog (German Credit Data) dataset encoding.
        """
        features = {
            'status': ['A11', 'A12', 'A13', 'A14'],
            'duration': None,
            'credit_history': ['A30', 'A31', 'A32', 'A33', 'A34'],
            'purpose': ['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A410'],
            'credit_amount': None,
            'savings': ['A61', 'A62', 'A63', 'A64', 'A65'],
            'employment': ['A71', 'A72', 'A73', 'A74', 'A75'],
            'installment_rate': None,
            'personal_status': ['A91', 'A92', 'A93', 'A94', 'A95'],
            'debtors': ['A101', 'A102', 'A103'],
            'residence_since': None,
            'property': ['A121', 'A122', 'A123', 'A124'],
            'age': None,
            'installment_plans': ['A141', 'A142', 'A143'],
            'housing': ['A151', 'A152', 'A153'],
            'existing_credits': None,
            'job': ['A171', 'A172', 'A173', 'A174'],
            'maintenance_people': None,
            'telephone': ['A191', 'A192'],
            'foreign_worker': ['A201', 'A202'],
            'class': [1, 2]
        }
       
        return features
