import logging
import os
from general_config import COMMON_LOG_PATH, LOG_LEVEL


class FlushableFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def get_general_logger(log_name, date_str):

    logger = logging.getLogger(log_name)
    logger.setLevel(LOG_LEVEL)

    file_handler = logging.FileHandler(
        os.path.join(COMMON_LOG_PATH, f'{log_name}_{date_str}.log'))
    
    file_handler.setLevel(LOG_LEVEL)

    # Create a formatter and set the format for the logs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger