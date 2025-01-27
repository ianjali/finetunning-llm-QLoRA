from datasets import load_dataset
from src.entity import BitsBytesConfig, ModelConfig, QLoraConfig, TrainingArgumentsConfig,SFTTrainerConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
import torch
from src.logging import logger
class ModelTrainer:
    def __init__(self):
        self.bnb_config=None
        self.peft_config=None
        # self.model_config=None
        self.train_args=None
        self.device_map = {"": 0}
    def set_config(self,bits_and_bytes_config:BitsBytesConfig,lora_config:QLoraConfig,training_arguments_config:TrainingArgumentsConfig):
        compute_dtype = getattr(torch, self.bits_and_bytes_config.compute_dtype)
        self.bnb_config = BitsAndBytesConfig(
                            load_in_4bit=self.bits_and_bytes_config.use_4bit,
                            bnb_4bit_quant_type=self.bits_and_bytes_config.bnb_4bit_quant_type,
                            bnb_4bit_compute_dtype=compute_dtype ,
                            bnb_4bit_use_double_quant=self.bits_and_bytes_config.use_nested_quant,
                        )
        if compute_dtype == torch.float16 and self.bits_and_bytes_config.use_4bit:
            major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)
            
        self.peft_config = LoraConfig(
                lora_alpha=self.lora_config.lora_alpha,
                lora_dropout=self.lora_config.lora_dropout,
                r=self.lora_config.lora_r,
                bias="none",
                task_type="CAUSAL_LM",
            )
        self.train_args = TrainingArguments(
                output_dir=self.training_arguments_config.output_dir,
                num_train_epochs=self.training_arguments_config.num_train_epochs,
                per_device_train_batch_size=self.training_arguments_config.per_device_train_batch_size,
                gradient_accumulation_steps=self.training_arguments_config.gradient_accumulation_steps,
                optim=self.training_arguments_config.optim,
                save_steps=self.training_arguments_config.save_steps,
                logging_steps=self.training_arguments_config.logging_steps,
                learning_rate=self.training_arguments_config.learning_rate,
                weight_decay=self.training_arguments_config.weight_decay,
                fp16=self.training_arguments_config.fp16,
                bf16=self.training_arguments_config.bf16,
                max_grad_norm=self.training_arguments_config.max_grad_norm,
                max_steps=self.training_arguments_config.max_steps,
                warmup_ratio=self.training_arguments_config.warmup_ratio,
                group_by_length=self.training_arguments_config.group_by_length,
                lr_scheduler_type=self.training_arguments_config.lr_scheduler_type,
                report_to="tensorboard"
            )
    
    def load_base_model_tokenizer(self,model_config:ModelConfig):
        #load base model
        base_model = AutoModelForCausalLM.from_pretrained(
                model_config.model_name,
                quantization_config=self.bnb_config,
                device_map=self.device_map
                )
        # Load LLaMA tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_config.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
        return base_model,tokenizer
    
    def finetune_model(self,sft:SFTTrainerConfig, model_config:ModelConfig,model:AutoModelForCausalLM,tokenizer:AutoTokenizer,train_df):
        # Set supervised fine-tuning parameters
        trainer = SFTTrainer(
            model=model,
            train_dataset=model_config,
            peft_config=self.peft_config,
            dataset_text_field="text",
            max_seq_length=self.sft.max_seq_length,
            tokenizer=tokenizer,
            args=self.train_args,
            packing=self.sft.packing,
        )

        # Train model
        # trainer.train()
        # trainer.model.save_pretrained(model_config.fine_tune_model)
        
