from src.constants import CONFIG_FILE_PATH, TRAINING_CONFIG_PATH
from src.utils.common import read_yaml

from src.entity import DataIngestionConfig,BitsBytesConfig,QLoraConfig,ModelConfig,TrainingArgumentsConfig,SFTTrainerConfig

class ConfigurationManager:
    def __init__(self,
                 config_path=CONFIG_FILE_PATH,
                 params_filepath=TRAINING_CONFIG_PATH):
        self.config=read_yaml(config_path)
        self.params=read_yaml(params_filepath)

        # create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self)-> DataIngestionConfig:
        config=self.config.data_ingestion
        data_ingestion_config=DataIngestionConfig(
            dataset_name=config.dataset_name,

        )

        return data_ingestion_config
    
    def get_bits_and_bytes_config(self)-> BitsBytesConfig:
        config=self.params.bit_bytes_params
        bits_and_bytes_config=BitsBytesConfig(
            use_4bit=config.use_4bit,
            bnb_4bit_compute_dtype=config.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            use_nested_quant=config.use_nested_quant
        )

        return bits_and_bytes_config
    
    def get_qlora_config(self) -> QLoraConfig:
        params=self.params.QLoRA_params
        qlora_config=QLoraConfig(
            lora_r=params.lora_r,
            lora_alpha=params.lora_alpha,
            lora_dropout = params.lora_dropout,
        )
        return qlora_config
    
    def get_model_config(self) -> ModelConfig:
        params = self.params.model_params
        model_config = ModelConfig(
            model_name=params.model_name,
            output_dir=params.output_dir,
            fine_tune_model=params.fine_tune_model
        )
        return model_config
    
    def get_training_args_config(self) -> TrainingArgumentsConfig:
        params = self.params.training_params
        training_config = TrainingArgumentsConfig(
            num_train_epochs=params.num_train_epochs,
            fp16=params.fp16,
            bf16=params.bf16,
            per_device_train_batch_size=params.per_device_train_batch_size,
            per_device_eval_batch_size=params.per_device_eval_batch_size,
            gradient_accumulation_steps=params.gradient_accumulation_steps,
            gradient_checkpointing=params.gradient_checkpointing,
            max_grad_norm=params.max_grad_norm,
            learning_rate=params.learning_rate,
            weight_decay=params.weight_decay,
            optim=params.optim,
            lr_scheduler_type=params.lr_scheduler_type,
            max_steps=params.max_steps,
            warmup_ratio=params.warmup_ratio,
            group_by_length=params.group_by_length,
            save_steps=params.save_steps,
            logging_steps=params.logging_steps
        )
        return training_config
    def get_sft_trainer_config(self)->SFTTrainerConfig:
        params=self.params.sft_params
        sft_config = SFTTrainerConfig(
            max_seq_length=params.max_seq_length,
            packing=params.packing
        )
        return sft_config
        



    

    
