import os
import sys 
import logging

# log directory
log_dir = "logs"
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

#logging file path 
log_filepath = os.path.join(log_dir, "continous_logs.log")
 
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, 
                    format=logging_str,
                    handlers=[
                        logging.FileHandler(log_filepath), # log file handler - file
                        logging.StreamHandler(sys.stdout) # console handler - terminal
                    ]
                    )
logger = logging.getLogger("summarizer_logger")
