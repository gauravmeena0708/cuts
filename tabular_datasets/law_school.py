import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from .base_dataset import BaseDataset

import sys
sys.path.append("..")
from utils import to_numeric


class LawSchool(BaseDataset):
    """LSAC National Longitudinal Bar Passage Study (Wightman 1998).

    Target: bar (TRUE = passed bar exam).
    Protected attributes: race1 (white / non-white), gender (male / female).
    Source: OpenML dataset 43890.
    """

    def __init__(self, name='LawSchool', single_bit_binary=False, device='cpu', random_state=42):
        super(LawSchool, self).__init__(name=name, device=device, random_state=random_state)

        self.features = LawSchool.get_features()
        self.single_bit_binary = single_bit_binary
        self.label = 'bar'
        self.train_features = {k: v for k, v in self.features.items() if k != self.label}

        cuts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(cuts_dir)
        sd_framework_root = os.path.dirname(project_root)
        gold_dir = os.path.join(sd_framework_root, 'data', 'gold', 'law_school')

        train_path = os.environ.get('CUTS_TRAIN_CSV') or os.path.join(gold_dir, 'train.csv')
        test_path = os.environ.get('CUTS_TEST_CSV') or os.path.join(gold_dir, 'test.csv')

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        col_order = list(self.features.keys())
        train_df = self._canonicalize_frame(train_df[[c for c in col_order if c in train_df.columns]])
        test_df = self._canonicalize_frame(test_df[[c for c in col_order if c in test_df.columns]])

        train_data = train_df.to_numpy()
        test_data = test_df.to_numpy()

        train_num = to_numeric(
            train_data, self.features, label=self.label,
            single_bit_binary=self.single_bit_binary
        ).astype(np.float32)
        test_num = to_numeric(
            test_data, self.features, label=self.label,
            single_bit_binary=self.single_bit_binary
        ).astype(np.float32)

        Xtrain, ytrain = train_num[:, :-1], train_num[:, -1]
        Xtest, ytest = test_num[:, :-1], test_num[:, -1]
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
            'lsat': None,
            'ugpa': None,
            'age': None,
            'decile1': None,
            'decile3': None,
            'fam_inc': None,
            'gender': ['male', 'female'],
            'race1': ['white', 'non-white'],
            'cluster': ['1', '2', '3', '4', '5', '6'],
            'fulltime': ['1', '2'],
            'ugpagt3': ['TRUE', 'FALSE'],
            'bar': ['TRUE', 'FALSE'],
        }

    @staticmethod
    def _canonicalize_frame(df):
        """Match pandas-loaded values to the string domains used by CUTS."""
        df = df.copy()
        for col in ('cluster', 'fulltime'):
            if col in df.columns:
                df[col] = df[col].map(lambda v: str(int(v)) if pd.notna(v) else v)
        for col in ('ugpagt3', 'bar'):
            if col in df.columns:
                df[col] = df[col].map(LawSchool._canonicalize_bool_label)
        return df

    @staticmethod
    def _canonicalize_bool_label(value):
        if pd.isna(value):
            return value
        text = str(value).strip().lower()
        if text in {'true', '1', '1.0'}:
            return 'TRUE'
        if text in {'false', '0', '0.0'}:
            return 'FALSE'
        return str(value).strip()
