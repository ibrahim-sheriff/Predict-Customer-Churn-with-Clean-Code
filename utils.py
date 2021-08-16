import os
import logging


def setup_logger(name, log_file, level=logging.INFO):
    "To setup as many loggers as you want"
    formatter = logging.Formatter('%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')
    
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def check_file_exists(file_path, file_name):
    return os.path.exists(os.path.join(os.path.abspath(file_path), file_name))
