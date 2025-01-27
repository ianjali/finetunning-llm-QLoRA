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
# # Load the entire model on the GPU 0
# device_map = {"": 0}
# # Load tokenizer and model with QLoRA configuration
# compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=use_4bit,
#     bnb_4bit_quant_type=bnb_4bit_quant_type,
#     bnb_4bit_compute_dtype=compute_dtype,
#     bnb_4bit_use_double_quant=use_nested_quant,
# )
# # Check GPU compatibility with bfloat16
# if compute_dtype == torch.float16 and use_4bit:
#     major, _ = torch.cuda.get_device_capability()
#     if major >= 8:
#         print("=" * 80)
#         print("Your GPU supports bfloat16: accelerate training with bf16=True")
#         print("=" * 80)
        
# # # Load base model
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     device_map=device_map
# )

# # Load LLaMA tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# # Load LoRA configuration
# peft_config = LoraConfig(
#     lora_alpha=lora_alpha,
#     lora_dropout=lora_dropout,
#     r=lora_r,
#     bias="none",
#     task_type="CAUSAL_LM",
# )

# # Set training parameters
# training_arguments = TrainingArguments(
#     output_dir=output_dir,
#     num_train_epochs=num_train_epochs,
#     per_device_train_batch_size=per_device_train_batch_size,
#     gradient_accumulation_steps=gradient_accumulation_steps,
#     optim=optim,
#     save_steps=save_steps,
#     logging_steps=logging_steps,
#     learning_rate=learning_rate,
#     weight_decay=weight_decay,
#     fp16=fp16,
#     bf16=bf16,
#     max_grad_norm=max_grad_norm,
#     max_steps=max_steps,
#     warmup_ratio=warmup_ratio,
#     group_by_length=group_by_length,
#     lr_scheduler_type=lr_scheduler_type,
#     report_to="tensorboard"
# )

# # Set supervised fine-tuning parameters
# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     peft_config=peft_config,
#     dataset_text_field="text",
#     max_seq_length=max_seq_length,
#     tokenizer=tokenizer,
#     args=training_arguments,
#     packing=packing,
# )
