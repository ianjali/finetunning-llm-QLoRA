from datasets import load_dataset
from src.entity import DataIngestionConfig
from src.logging import logger
class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config=config

    def get_data(self):
        dataset = load_dataset(self.config.dataset_name, split="train")
        return dataset
