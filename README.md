## Read More

For a detailed explanation of fine-tuning Llama-2-7b-chat with LoRA and QLoRA, check out this [Medium article](https://medium.com/@mudgal.anjali.am/efficient-fine-tuning-with-lora-and-qlora-reducing-parameters-and-memory-usage-for-large-models-facf2ee60af4).

Project Structure
```

finetunning-llm-QLoRA/
│
├── config/                
│   ├── config.yaml        # Configuration settings for the project
│   └── tranining_args.yaml # configuration setting for training parameters 
│
├── models /             # Folder for base and finetuned models
│   ├── dataset/
│       ├── base_model
│       └── finetuned_model
│
├── src/                   # Source code for model training and deployment
│   └── MobilePriceClassification/
│       ├── components/     # Scripts for different model components
│       │   ├── data_ingestion.py   # Data ingestion logic
│       │   └── model_trainer.py    # Model training logic
│       │
│       ├── config/          # Configuration objects 
│       │   └── configuration.py
│       │
│       ├── constants/       # Constant variables and enums
│       │   └── __init__.py
│       │
│       ├── entity/          # Entity files (such as data schema)
│       │   └── __init__.py
│       │
│       ├── logging/         # Logging utilities
│       │   └── __init__.py
│       │
│       ├── pipeline/        # Pipeline for model stages
│       │   ├── stage_1_data_ingestion_pipeline.py  # Data ingestion pipeline
│       │   └── stage_2_model_training.py           # Model training pipeline
│       │
│       └── utils/           # Utility functions
│           └── common.py
│
├── logs/                  # Logs for tracking model activities
│   └── continuos_logs.log # Logs file
│
├── main.py                # Main script to run the project
├── .env                   # Environment variables for the project
├── requirements.txt       # Dependencies for the project
├── README.md              # Project documentation
└── .gitignore             # Git ignore file
```
