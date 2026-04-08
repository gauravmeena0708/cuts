
import torch
import numpy as np
import pandas as pd
from .base_dataset import BaseDataset
import sys
import os

# Allow importing from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import to_numeric

class Beijing(BaseDataset):
    """
    Beijing PM2.5 dataset for CuTS.
    This is a regression dataset where pm2.5 is the target.
    
    Note: The base class expects the label to be the LAST feature in self.features
    and uses self.train_features (features without label) for decode_batch.
    """
    
    def __init__(self, name='Beijing', device='cpu', random_state=42):
        super(Beijing, self).__init__(name=name, device=device, random_state=random_state)

        # Features with label at the END (required by base class)
        self.features = Beijing.get_features()
        self.label = 'pm2.5'

        cuts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(cuts_dir)
        sd_framework_root = os.path.dirname(project_root)
        gold_dir = os.path.join(sd_framework_root, 'data', 'gold', 'beijing')
        train_path = os.environ.get('CUTS_TRAIN_CSV') or os.path.join(gold_dir, 'train.csv')
        test_path = os.environ.get('CUTS_TEST_CSV') or os.path.join(gold_dir, 'test.csv')

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Drop NaN rows
        train_df = train_df.dropna()
        test_df = test_df.dropna()

        # Reorder columns to match features dict order (label at end)
        ordered_cols = list(self.features.keys())
        train_df = train_df[ordered_cols]
        test_df = test_df[ordered_cols]

        self.train_features = {k: v for k, v in self.features.items() if k != self.label}

        # Convert categoricals to string for one-hot encoding
        for col, domain in self.features.items():
            if domain is not None:
                train_df[col] = train_df[col].astype(str)
                test_df[col] = test_df[col].astype(str)

        train_data = train_df.to_numpy()
        test_data = test_df.to_numpy()

        train_data_num = to_numeric(train_data, self.features, label=self.label, single_bit_binary=False)
        test_data_num = to_numeric(test_data, self.features, label=self.label, single_bit_binary=False)

        # Label is last column after to_numeric
        Xtrain = train_data_num[:, :-1].astype(np.float32)
        Xtest = test_data_num[:, :-1].astype(np.float32)
        ytrain = train_data_num[:, -1].astype(np.float32)
        ytest = test_data_num[:, -1].astype(np.float32)

        self.num_features = Xtrain.shape[1]

        self.Xtrain = torch.tensor(Xtrain).to(self.device)
        self.Xtest = torch.tensor(Xtest).to(self.device)
        
        # Regression: use float for labels
        self.ytrain = torch.tensor(ytrain, dtype=torch.float32).to(self.device)
        self.ytest = torch.tensor(ytest, dtype=torch.float32).to(self.device)

        self.train()
        self._calculate_mean_std()
        self._calculate_mins_maxs()
        self._calculate_categorical_feature_distributions_and_continuous_bounds()
        self.create_feature_domain_lists()

    def _calculate_mins_maxs(self):
        """
        Override for regression datasets where label is numeric.
        Uses train_features instead of features to avoid index mismatch.
        """
        Xtrain = self.decode_batch(self.Xtrain.clone(), standardized=self.standardized)
        mins = []
        maxs = []
        for i, (feature_name, feature_domain) in enumerate(self.train_features.items()):
            if feature_domain is None:
                train_min = np.min(Xtrain[:, i].astype(float))
                train_max = np.max(Xtrain[:, i].astype(float))
                mins.append(train_min)
                maxs.append(train_max)
        self.mins, self.maxs = mins, maxs

    @staticmethod
    def get_features():
        """
        Returns feature dictionary with label (pm2.5) at the END.
        The order must match the column reordering done in __init__.
        cbwd is the only categorical feature.
        """
        return {
            'year': None,
            'month': None,
            'day': None,
            'hour': None,
            'DEWP': None,
            'TEMP': None,
            'PRES': None,
            'cbwd': ['cv', 'NE', 'NW', 'SE'],  # Wind direction - categorical
            'Iws': None,
            'Is': None,
            'Ir': None,
            'pm2.5': None,  # Label MUST be at end for base class to work
        }
