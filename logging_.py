import logging
import os
from rich.logging import RichHandler

def adding_logs(file_name):
    log_dir = 'logs'
    os.makedirs(log_dir,exist_ok=True)

    
    logger = logging.getLogger(f'{file_name}')
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        console_handler = RichHandler(rich_tracebacks=True)
        console_handler.setLevel(logging.DEBUG)

        log_file_path = os.path.join(log_dir,f'{file_name}.log')
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger