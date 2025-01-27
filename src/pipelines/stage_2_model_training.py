from src.config.configuration import ConfigurationManager
from src.components.model_training import ModelTrainer
from src.logging import logger


class ModelTrainerPipeline:
    def __init__(self):
        pass

    def initiate_model_training(self,train_df):
        config=ConfigurationManager()
        
        bits_and_bytes_config=config.get_bits_and_bytes_config()
        lora_config=config.get_lora_config()
        model_config=config.get_model_config()
        training_arguments_config=config.get_training_arguments_config()
        sft=config.get_sft_trainer_config()
        
        
        model_trainer = ModelTrainer()
        model_trainer.set_config(bits_and_bytes_config,lora_config,model_config,training_arguments_config)
        base_model,tokenizer = model_trainer.load_base_model_tokenizer(model_config)
        model_trainer.finetune_model(sft, model_config,base_model,tokenizer,train_df)
        
        
        
        
