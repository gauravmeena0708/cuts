
import torch
import numpy as np
import pandas as pd
from .base_dataset import BaseDataset
import sys
import os

# Allow importing from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import to_numeric

class News(BaseDataset):
    def __init__(self, name='News', device='cpu', random_state=42):
        super(News, self).__init__(name=name, device=device, random_state=random_state)

        # News is regression task in TabDiff json?
        # "task_type": "regression"
        # CuTS is usually for classification or generation. 
        # Use a dummy label or bin the target if needed, but for generation it doesn't matter much.
        
        self.features = News.get_features() # This needs to define columns.
        self.label = 'shares' # Target in news.csv

        data_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        gold_dir = os.path.join(os.path.dirname(data_dir), 'data', 'gold', 'news')
        train_path = os.path.join(gold_dir, 'train.csv')
        test_path = os.path.join(gold_dir, 'test.csv')

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Remove non-predictive columns if any (url, timedelta usually dropped)
        # TabDiff json num_col_idx starts from 0, maybe already cleaned?
        # Let's inspect columns from df. For now, assuming standard cleaned news.
        # But typically "url" and "timedelta" are in raw.
        if 'url' in train_df.columns:
            train_df = train_df.drop(columns=['url', 'timedelta'])
            test_df = test_df.drop(columns=['url', 'timedelta'])

        self.train_features = {k: v for k, v in self.features.items() if k != self.label}

        train_data = train_df.to_numpy()
        test_data = test_df.to_numpy()

        train_data_num = to_numeric(train_data, self.features, label=self.label, single_bit_binary=False)
        test_data_num = to_numeric(test_data, self.features, label=self.label, single_bit_binary=False)

        Xtrain = train_data_num[:, :-1].astype(np.float32)
        Xtest = test_data_num[:, :-1].astype(np.float32)
        ytrain = train_data_num[:, -1].astype(np.float32)
        ytest = test_data_num[:, -1].astype(np.float32)
        
        # Since regression, ytrain is float. CuTS might expect long for classification?
        # If regression, we should keep as float. 
        # But `to_numeric` logic depends on implementation.
        # If CuTS is strictly categorical/classification generative, we might need to bin `shares`.
        # For now, keep as float/long and let the model handle or crash.
        # But standard `ytrain` cast to `long` suggests classification.
        # We will cast to float for regression if possible, but `self.ytrain` type hint was `long` in examples.
        # Let's verify `adult.py`: `dtype=torch.long`.
        # If `news` is regression, we might face issues.
        # Assuming we just want generation, we can treat it as another feature?
        # But for now, faithfully implementing loader.
        
        self.num_features = Xtrain.shape[1]

        self.Xtrain = torch.tensor(Xtrain).to(self.device)
        self.Xtest = torch.tensor(Xtest).to(self.device)
        
        # Check if regression
        if len(self.features[self.label] or []) == 0:
             self.ytrain = torch.tensor(ytrain, dtype=torch.float32).to(self.device)
             self.ytest = torch.tensor(ytest, dtype=torch.float32).to(self.device)
        else:
             self.ytrain = torch.tensor(ytrain, dtype=torch.long).to(self.device)
             self.ytest = torch.tensor(ytest, dtype=torch.long).to(self.device)

        self.train()
        self._calculate_mean_std()
        self._calculate_mins_maxs()
        self._calculate_categorical_feature_distributions_and_continuous_bounds()
        self.create_feature_domain_lists()

    @staticmethod
    def get_features():
        # Large dataset, returning simplified dict.
        # Most are numerical.
        # 'data_channel_is_...' are categorical (one-hot in raw?)
        return {
            'n_tokens_title': None,
            'n_tokens_content': None,
            'n_unique_tokens': None,
            'n_non_stop_words': None,
            'n_non_stop_unique_tokens': None,
            'num_hrefs': None,
            'num_self_hrefs': None,
            'num_imgs': None,
            'num_videos': None,
            'average_token_length': None,
            'num_keywords': None,
            'data_channel_is_lifestyle': ['0','1'],
            'data_channel_is_entertainment': ['0','1'],
            'data_channel_is_bus': ['0','1'],
            'data_channel_is_socmed': ['0','1'],
            'data_channel_is_tech': ['0','1'],
            'data_channel_is_world': ['0','1'],
            'kw_min_min': None,
            'kw_max_min': None,
            'kw_avg_min': None,
            'kw_min_max': None,
            'kw_max_max': None,
            'kw_avg_max': None,
            'kw_min_avg': None,
            'kw_max_avg': None,
            'kw_avg_avg': None,
            'self_reference_min_shares': None,
            'self_reference_max_shares': None,
            'self_reference_avg_sharess': None, 
            'weekday_is_monday': ['0','1'],
            'weekday_is_tuesday': ['0','1'],
            'weekday_is_wednesday': ['0','1'],
            'weekday_is_thursday': ['0','1'],
            'weekday_is_friday': ['0','1'],
            'weekday_is_saturday': ['0','1'],
            'weekday_is_sunday': ['0','1'],
            'is_weekend': ['0','1'],
            'LDA_00': None,
            'LDA_01': None,
            'LDA_02': None,
            'LDA_03': None,
            'LDA_04': None,
            'global_subjectivity': None,
            'global_sentiment_polarity': None,
            'global_rate_positive_words': None,
            'global_rate_negative_words': None,
            'rate_positive_words': None,
            'rate_negative_words': None,
            'avg_positive_polarity': None,
            'min_positive_polarity': None,
            'max_positive_polarity': None,
            'avg_negative_polarity': None,
            'min_negative_polarity': None,
            'max_negative_polarity': None,
            'title_subjectivity': None,
            'title_sentiment_polarity': None,
            'abs_title_subjectivity': None,
            'abs_title_sentiment_polarity': None,
            'shares': None # Regression target
        }
