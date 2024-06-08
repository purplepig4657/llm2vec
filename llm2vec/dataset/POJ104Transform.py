from .dataset import DataSample, TrainSample, Dataset
from accelerate.logging import get_logger
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

logger = get_logger(__name__, log_level="INFO")

class POJ104Transform(Dataset):
    def __init__(
        self,
        dataset_name: str = "POJ104Transform",
        split: str = "validation",
        file_path: str = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.data = []
        self.val_data = []
        if file_path is None:
            file_path = "cache/output5.csv"
        self.load_data(file_path)

    def __len__(self):
        if self.split == "train":
            return len(self.data)
        if self.split == "validation":
            return len(self.val_data)


    def load_data(self, file_path: str = None):
        data = pd.read_csv(file_path)
        data = data.sample(frac=1, random_state=4).reset_index(drop=True)
        train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
        for idx, row in train_data.iterrows():
            self.data.append(
                DataSample(
                    id_=idx,
                    query=row["code1"],
                    positive=row["code2"],
                    label=row["label"]
                )
            )
        for idx, row in val_data.iterrows():
            self.val_data.append(
                DataSample(
                    id_=idx,
                    query=row["code1"],
                    positive=row["code2"],
                    label=row["label"]
                )
            )
        logger.info(f"Loaded {len(self.data)} samples.")

    def __getitem__(self, index):
        if self.split == "train":
            sample = self.data[index]
            return TrainSample(
                texts=[sample.query, sample.positive], label=sample.label
            )
        elif self.split == "validation":
            sample = self.val_data[index]
            return TrainSample(
                texts=[sample.query, sample.positive], label=sample.label
            )
