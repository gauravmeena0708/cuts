import os
import sys
import numpy as np
import pandas as pd
import torch

from .base_dataset import BaseDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import to_numeric


class BankMarketing(BaseDataset):
    """UCI Bank Marketing dataset for CuTS. Binary classification on `y`."""

    def __init__(self, name='BankMarketing', device='cpu', random_state=42):
        super(BankMarketing, self).__init__(name=name, device=device, random_state=random_state)

        self.features = BankMarketing.get_features()
        self.label = 'y'

        cuts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(cuts_dir)
        sd_framework_root = os.path.dirname(project_root)
        gold_dir = os.path.join(sd_framework_root, 'data', 'gold', 'bank_marketing')
        train_path = os.environ.get('CUTS_TRAIN_CSV') or os.path.join(gold_dir, 'train.csv')
        test_path = os.environ.get('CUTS_TEST_CSV') or os.path.join(gold_dir, 'test.csv')

        train_df = pd.read_csv(train_path).dropna()
        test_df = pd.read_csv(test_path).dropna()

        ordered_cols = list(self.features.keys())
        train_df = train_df[ordered_cols]
        test_df = test_df[ordered_cols]

        self.train_features = {k: v for k, v in self.features.items() if k != self.label}

        for col, domain in self.features.items():
            if domain is not None:
                train_df[col] = train_df[col].astype(str)
                test_df[col] = test_df[col].astype(str)
            else:
                train_df[col] = train_df[col].astype(float)
                test_df[col] = test_df[col].astype(float)

        train_data = train_df.to_numpy()
        test_data = test_df.to_numpy()

        train_data_num = to_numeric(train_data, self.features, label=self.label, single_bit_binary=False)
        test_data_num = to_numeric(test_data, self.features, label=self.label, single_bit_binary=False)

        Xtrain = train_data_num[:, :-1].astype(np.float32)
        Xtest = test_data_num[:, :-1].astype(np.float32)
        ytrain = train_data_num[:, -1].astype(np.float32)
        ytest = test_data_num[:, -1].astype(np.float32)

        self.num_features = Xtrain.shape[1]

        self.Xtrain = torch.tensor(Xtrain).to(self.device)
        self.Xtest = torch.tensor(Xtest).to(self.device)
        self.ytrain = torch.tensor(ytrain, dtype=torch.long).to(self.device)
        self.ytest = torch.tensor(ytest, dtype=torch.long).to(self.device)

        self.train()
        self._calculate_mean_std()
        self._calculate_mins_maxs()
        self._calculate_categorical_feature_distributions_and_continuous_bounds()
        self.create_feature_domain_lists()

    @staticmethod
    def get_features():
        return {
            'age': None,
            'job': ['management', 'technician', 'entrepreneur', 'blue-collar', 'unknown',
                    'retired', 'admin.', 'services', 'self-employed', 'unemployed',
                    'housemaid', 'student'],
            'marital': ['married', 'single', 'divorced'],
            'education': ['tertiary', 'secondary', 'unknown', 'primary'],
            'default': ['no', 'yes'],
            'balance': None,
            'housing': ['no', 'yes'],
            'loan': ['no', 'yes'],
            'contact': ['unknown', 'cellular', 'telephone'],
            'day': None,
            'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                      'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
            'duration': None,
            'campaign': None,
            'pdays': None,
            'previous': None,
            'poutcome': ['unknown', 'failure', 'other', 'success'],
            'y': ['no', 'yes'],
        }
