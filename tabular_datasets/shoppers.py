
import torch
import numpy as np
import pandas as pd
from .base_dataset import BaseDataset
import sys
import os

# Allow importing from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import to_numeric

class Shoppers(BaseDataset):
    def __init__(self, name='Shoppers', device='cpu', random_state=42):
        super(Shoppers, self).__init__(name=name, device=device, random_state=random_state)

        self.features = Shoppers.get_features()
        self.label = 'Revenue'

        # Load data — prefer CUTS_TRAIN_CSV / CUTS_TEST_CSV env vars (set by wrapper),
        # fall back to gold path for standalone use.
        cuts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(cuts_dir)
        sd_framework_root = os.path.dirname(project_root)
        gold_dir = os.path.join(sd_framework_root, 'data', 'gold', 'shoppers')
        train_path = os.environ.get('CUTS_TRAIN_CSV') or os.path.join(gold_dir, 'train.csv')
        test_path = os.environ.get('CUTS_TEST_CSV') or os.path.join(gold_dir, 'test.csv')

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train file not found at {train_path}")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        self.train_features = {k: v for k, v in self.features.items() if k != self.label}

        # Preprocess
        for col, domain in self.features.items():
            if col not in train_df.columns:
                raise ValueError(f"Column {col} not found in train data")
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

        # Revenue is the target (True/False -> 1/0)
        # to_numeric typically handles the label if specified
        
        # Split features and labels - assume label is last column from to_numeric if label passed
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
        # Define features and domains
        return {
            'Administrative': None,
            'Administrative_Duration': None,
            'Informational': None,
            'Informational_Duration': None,
            'ProductRelated': None,
            'ProductRelated_Duration': None,
            'BounceRates': None,
            'ExitRates': None,
            'PageValues': None,
            'SpecialDay': None,
            'Month': ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            'OperatingSystems': ['1', '2', '3', '4', '5', '6', '7', '8'],
            'Browser': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'],
            'Region': ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'TrafficType': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'],
            'VisitorType': ['Returning_Visitor', 'New_Visitor', 'Other'],
            'Weekend': ['False', 'True'],
            'Revenue': ['False', 'True']
        }
