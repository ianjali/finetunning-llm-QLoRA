# import os
# import torch
# from datasets import load_dataset
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     HfArgumentParser,
#     TrainingArguments,
#     pipeline,
#     logging,
# )
from src.pipelines.stage_1_data_pipeline import DataIngestionTrainingPipeline
from src.pipelines.stage_2_model_training import ModelTrainerPipeline
# from peft import LoraConfig, PeftModel
# from trl import SFTTrainer

STAGE_1="Data Preparation"
data_ingestion_pipeline = DataIngestionTrainingPipeline()
train_df = data_ingestion_pipeline.initiate_data_ingestion()


STAGE_2="Model Training"
model_trainer_pipeline = ModelTrainerPipeline()
model_trainer_pipeline.initiate_model_training(train_df)
