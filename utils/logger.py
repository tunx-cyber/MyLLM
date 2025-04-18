import logging
import os
 
def setup_logger(name, log_file, level=logging.INFO):
    
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    directory = os.path.dirname(log_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger