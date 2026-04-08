
import torch
import numpy as np
import pandas as pd
from .base_dataset import BaseDataset
import sys
import os

# Allow importing from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import to_numeric

class Magic(BaseDataset):
    def __init__(self, name='Magic', device='cpu', random_state=42):
        super(Magic, self).__init__(name=name, device=device, random_state=random_state)

        self.features = Magic.get_features()
        self.label = 'class'

        data_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        gold_dir = os.path.join(os.path.dirname(data_dir), 'data', 'gold', 'magic')
        train_path = os.path.join(gold_dir, 'train.csv')
        test_path = os.path.join(gold_dir, 'test.csv')

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Ensure column names map to what we expect
        # Migration provided explicit names for magic: ["Length", "Width", "Size", "Conc", "Conc1", "Asym", "M3Long", "M3Trans", "Alpha", "Dist", "class"]
        
        self.train_features = {k: v for k, v in self.features.items() if k != self.label}

        # Numeric conversion
        # All columns are float except class which is g/h
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
            'Length': None,
            'Width': None,
            'Size': None,
            'Conc': None,
            'Conc1': None,
            'Asym': None,
            'M3Long': None,
            'M3Trans': None,
            'Alpha': None,
            'Dist': None,
            'class': ['g', 'h']
        }
