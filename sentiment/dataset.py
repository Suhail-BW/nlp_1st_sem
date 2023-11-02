import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ReviewsDataset(Dataset):
    def __init__(self, csv_path, tokenizer, config):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path, dtype=str, na_filter=False)
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return self.df.shape[0]

    def _get_from_row(self, row):
        text = row["Review"]
        target = int(row["Rating"])
        assert target in range(1, 6)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.config.max_length,
            return_token_type_ids=False,
            truncation=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        # print(np.sum([1 for i in ids if i > 0]), np.sum(mask))

        # positive or negative
        sentiment_target = 1
        if target in [1, 2]:
            sentiment_target = 0
        elif target in [4, 5]:
            sentiment_target = 2

        # strong or weak
        strength_target = 0
        strength_weight = 0 if target == 3 else 1
        if target in [1, 5]:
            strength_target = 1

        # index starts with 0
        target = target - 1
        # unsqueeze
        target_onehot = np.eye(self.config.num_classes)[target]
        sentiment_target_onehot = np.eye(3)[sentiment_target]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "masks": torch.tensor(mask, dtype=torch.long),
            "targets": torch.tensor(target_onehot, dtype=torch.float),
            "sentiment_targets": torch.tensor(sentiment_target_onehot, dtype=torch.float),
            "strength_targets": torch.tensor(strength_target, dtype=torch.float),
            "strength_weights": torch.tensor(strength_weight, dtype=torch.float),
        }

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return self._get_from_row(row)


class RandomReviewsDataset(ReviewsDataset):
    def __init__(self, csv_paths, tokenizer, config, num_samples=1600):
        super(RandomReviewsDataset, self).__init__(csv_paths[0], tokenizer, config)
        self.num_samples = num_samples
        self.dfs = [
            pd.read_csv(csv_path, dtype=str, na_filter=False)
            for csv_path in csv_paths
        ]
        # print(csv_paths)
        assert len(self.dfs) == 5

    def __len__(self):
        return self.num_samples * len(self.dfs)

    def __getitem__(self, index):
        df_index = index // self.num_samples
        df = self.dfs[df_index]
        item_index = random.randrange(0, len(df))
        row = df.iloc[item_index]
        assert row["Rating"] == str(df_index + 1)
        return self._get_from_row(row)
