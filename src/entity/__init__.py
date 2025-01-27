from dataclasses import dataclass
from pathlib import Path
import os
from src.logging import logger
@dataclass
class DataIngestionConfig:
    dataset_name: str

@dataclass
class BitsBytesConfig:
    use_4bit: bool
    bnb_4bit_compute_dtype: str
    bnb_4bit_quant_type: str
    use_nested_quant: bool
 
@dataclass          
class QLoraConfig:
    lora_r:int
    lora_alpha: int
    lora_dropout: float

@dataclass          
class ModelConfig:
    model_name:str
    output_dir:Path
    fine_tune_model:Path

@dataclass
class SFTTrainerConfig:
    max_seq_length: int
    packing:bool
    
@dataclass
class TrainingArgumentsConfig:
    num_train_epochs:int
    fp16:bool
    bf16:bool
    per_device_train_batch_size:int
    per_device_eval_batch_size:int
    gradient_accumulation_steps:int
    gradient_checkpointing:bool
    max_grad_norm:float
    learning_rate:float
    weight_decay:float
    optim:str
    lr_scheduler_type:str
    max_steps:int
    warmup_ratio:float
    group_by_length:bool
    save_steps:int
    logging_steps:int