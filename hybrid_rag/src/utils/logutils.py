"""
Module Name: hybrid_search.py
Author: Samiksha Kolhe
Version: 0.1.0
"""
import logging
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel


class Logger:
    def __init__(self, file_name: str | None = None, log_dir: str = "logs") -> None:
        """
        Initializes the Logger class.

        :param file_name: Name of the log file. If None, it generates a timestamped log file name.
        :param log_dir: Directory where log files will be stored.
        """
        # Ensure the logs directory exists
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Generate a default file name if not provided
        if not file_name:
            file_name = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        self.file_path = self.log_dir / file_name

        # Configure logging
        self._configure_logger()

    def _configure_logger(self) -> None:
        """
        Configures the logger with basic settings.
        """
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=str(self.file_path),
        )
        self.logger = logging.getLogger(self.file_path.stem)

    def get_logger(self) -> logging.Logger:
        """
        Returns the logger instance.

        :return: Configured logger object.
        """
        return self.logger
