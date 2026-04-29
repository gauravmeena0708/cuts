import os
import sys
import numpy as np
import pandas as pd
import torch

from .base_dataset import BaseDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import to_numeric


class TelcoChurn(BaseDataset):
    """Telco Customer Churn dataset for CuTS. Binary classification on `Churn`.

    Many domain values contain whitespace (e.g. `Fiber optic`, `Month-to-month`,
    `No internet service`). CuTS escapes whitespace symmetrically in both the
    program and the feature dict during parsing, so constraints on these values
    do round-trip — but the cuts_wrapper still emits a warning per BUG-007.
    """

    def __init__(self, name='TelcoChurn', device='cpu', random_state=42):
        super(TelcoChurn, self).__init__(name=name, device=device, random_state=random_state)

        self.features = TelcoChurn.get_features()
        self.label = 'Churn'

        cuts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(cuts_dir)
        sd_framework_root = os.path.dirname(project_root)
        gold_dir = os.path.join(sd_framework_root, 'data', 'gold', 'telco_churn')
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
            'gender': ['Female', 'Male'],
            'SeniorCitizen': ['0', '1'],
            'Partner': ['No', 'Yes'],
            'Dependents': ['No', 'Yes'],
            'tenure': None,
            'PhoneService': ['No', 'Yes'],
            'MultipleLines': ['No phone service', 'No', 'Yes'],
            'InternetService': ['DSL', 'Fiber optic', 'No'],
            'OnlineSecurity': ['No', 'Yes', 'No internet service'],
            'OnlineBackup': ['No', 'Yes', 'No internet service'],
            'DeviceProtection': ['No', 'Yes', 'No internet service'],
            'TechSupport': ['No', 'Yes', 'No internet service'],
            'StreamingTV': ['No', 'Yes', 'No internet service'],
            'StreamingMovies': ['No', 'Yes', 'No internet service'],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'PaperlessBilling': ['No', 'Yes'],
            'PaymentMethod': ['Electronic check', 'Mailed check',
                              'Bank transfer (automatic)', 'Credit card (automatic)'],
            'MonthlyCharges': None,
            'TotalCharges': None,
            'Churn': ['No', 'Yes'],
        }
