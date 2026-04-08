
import torch
import numpy as np
import pandas as pd
from .base_dataset import BaseDataset
import sys
import os

# Allow importing from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import to_numeric

class Default(BaseDataset):
    def __init__(self, name='Default', device='cpu', random_state=42):
        super(Default, self).__init__(name=name, device=device, random_state=random_state)

        self.features = Default.get_features()
        self.label = 'default payment next month'

        # Load data
        # Assume files are in tabular_datasets/Default/default.train and default.test
        cuts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(cuts_dir)
        sd_framework_root = os.path.dirname(project_root)
        gold_dir = os.path.join(sd_framework_root, 'data', 'gold', 'default')
        train_path = os.environ.get('CUTS_TRAIN_CSV') or os.path.join(gold_dir, 'train.csv')
        test_path = os.environ.get('CUTS_TEST_CSV') or os.path.join(gold_dir, 'test.csv')

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train file not found at {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found at {test_path}")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        self.train_features = {k: v for k, v in self.features.items() if k != self.label}

        # Preprocess: Ensure types match features dict
        for col, domain in self.features.items():
            if col not in train_df.columns:
                raise ValueError(f"Column {col} not found in train data")
                
            if domain is not None:
                # Categorical -> convert to string to match domain values
                train_df[col] = train_df[col].astype(str)
                test_df[col] = test_df[col].astype(str)
            else:
                # Numerical -> convert to float
                train_df[col] = train_df[col].astype(float)
                test_df[col] = test_df[col].astype(float)

        train_data = train_df.to_numpy()
        test_data = test_df.to_numpy()

        # Convert to numeric (one-hot etc using utils.to_numeric)
        train_data_num = to_numeric(train_data, self.features, label=self.label, single_bit_binary=False)
        test_data_num = to_numeric(test_data, self.features, label=self.label, single_bit_binary=False)

        # Split features and labels
        # Last column is label because 'default payment next month' is last in CSV and features dict
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
        # Complete domain specification for Default dataset
        pay_domain = ['-2', '-1', '0', '1', '2', '3', '4', '5', '6', '7', '8']
        
        return {
            'LIMIT_BAL': None,
            'SEX': ['1', '2'],
            'EDUCATION': ['0', '1', '2', '3', '4', '5', '6'],
            'MARRIAGE': ['0', '1', '2', '3'],
            'AGE': None,
            'PAY_0': pay_domain,
            'PAY_2': pay_domain,
            'PAY_3': pay_domain,
            'PAY_4': pay_domain,
            'PAY_5': pay_domain,
            'PAY_6': pay_domain,
            'BILL_AMT1': None,
            'BILL_AMT2': None,
            'BILL_AMT3': None,
            'BILL_AMT4': None,
            'BILL_AMT5': None,
            'BILL_AMT6': None,
            'PAY_AMT1': None,
            'PAY_AMT2': None,
            'PAY_AMT3': None,
            'PAY_AMT4': None,
            'PAY_AMT5': None,
            'PAY_AMT6': None,
            'default payment next month': ['0', '1']
        }
