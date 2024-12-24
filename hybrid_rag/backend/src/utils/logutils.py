import sys
import os
import logging
import logging.config
from pathlib import Path

def Logger(file_name):

    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s %(levelname)s %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        filename = f"logs/{file_name}"
    )

    log_object = logging.getLogger(file_name.split('.')[0])
    return log_object