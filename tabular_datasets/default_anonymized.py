
import torch
import numpy as np
import pandas as pd
from .base_dataset import BaseDataset
import sys
import os

# Allow importing from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import to_numeric

class DefaultAnonymized(BaseDataset):
    def __init__(self, name='Default', device='cpu', random_state=42):
        super(DefaultAnonymized, self).__init__(name=name, device=device, random_state=random_state)

        self.label = 'label'

        # Load data from the centralized/relddpm data directory
        # Path relative to sub/cuts/tabular_datasets/
        # Root is ../../.. from here
        # Target is sub/27-11-2025/datasets/
        
        # Using absolute paths is safer if possible, but relative keeps it portable within the repo structure
        # Repository root relative to this file: ../../..
        # Let's try to find the root dynamically or use the known relative path
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        data_dir = os.path.join(project_root, 'sub', '27-11-2025', 'datasets')

        train_path = os.environ.get('CUTS_TRAIN_CSV') or os.path.join(data_dir, 'default_train.csv')
        test_path = os.environ.get('CUTS_TEST_CSV') or os.path.join(data_dir, 'default_test.csv')

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train file not found at {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found at {test_path}")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        self.features = DefaultAnonymized.get_features(train_df=train_df, test_df=test_df)

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
        # Last column is label because 'label' is last in CSV and features dict (usually)
        # But we should rely on features dict order or just taking the last col if we are sure
        # features dict has label at end.
        
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
    def get_features(train_df=None, test_df=None):
        features = {
          "col0": None,
          "col1": ["S0", "S1"],
          "col2": ["E0", "E1", "E2", "E3", "E4", "E5", "E6"],
          "col3": ["M0", "M1", "M2", "M3"],
          "col4": None,
          "col5": ["P0", "P1", "P10", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"],
          "col6": ["P0", "P1", "P10", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"],
          "col7": ["P0", "P1", "P10", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"],
          "col8": ["P0", "P1", "P10", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"],
          "col9": ["P0", "P1", "P10", "P2", "P4", "P5", "P6", "P7", "P8", "P9"],
          "col10": ["P0", "P1", "P10", "P2", "P4", "P5", "P6", "P7", "P8", "P9"],
          "col11": None,
          "col12": None,
          "col13": None,
          "col14": None,
          "col15": None,
          "col16": None,
          "col17": None,
          "col18": None,
          "col19": None,
          "col20": None,
          "col21": None,
          "col22": None,
          "label": ["L0", "L1"]
        }

        if train_df is not None:
            combined = [train_df]
            if test_df is not None:
                combined.append(test_df)
            merged = pd.concat(combined, ignore_index=True)
            for col, domain in list(features.items()):
                if domain is None:
                    continue
                if col in merged.columns:
                    features[col] = sorted(
                        merged[col].dropna().astype(str).unique().tolist()
                    )

        return features
