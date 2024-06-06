from .dataset import DataSample, TrainSample, Dataset
from accelerate.logging import get_logger
from datasets import load_dataset
import pandas as pd

logger = get_logger(__name__, log_level="INFO")

class POJ104(Dataset):
    def __init__(
        self,
        dataset_name: str = "POJ104",
        split: str = "validation",
        file_path: str = "cache/wiki1m_for_simcse.txt",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.data = []
        self.load_data()

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str = None):
        data = pd.DataFrame(load_dataset('google/code_x_glue_cc_clone_detection_poj104', split='train'))
        for idx, row in data.iterrows():
            self.data.append(
                DataSample(
                    id_=idx,
                    query=row["code"],
                    positive=row["code"],
                )
            )
        logger.info(f"Loaded {len(self.data)} samples.")

    def __getitem__(self, index):
        sample = self.data[index]
        if self.split == "train":
            return TrainSample(
                texts=[sample.query, sample.positive], label=1.0
            )
        elif self.split == "validation":
            assert False, "POJ104 does not have a validation split."
