import logging

# import sys
from logging import Logger

import coloredlogs


class LoggerFactory:
    FIELD_STYLES = {
        "asctime": {"color": "white"},
        "levelname": {"color": "green"},
        "filename": {"color": "cyan"},
        "funcName": {"color": "cyan"},
        "lineno": {"color": "yellow"},
        "message": {"color": "white"},
    }

    @classmethod
    def get_logger(cls, name: str = __name__, level: int = logging.DEBUG) -> Logger:
        # Create logger with the specified name
        local_logger = logging.getLogger(name)
        local_logger.setLevel(level)

        # Create handler and set provided level
        handler = logging.StreamHandler()
        handler.setLevel(level)

        # Create formatter and add it to handler
        formatter = coloredlogs.ColoredFormatter(
            "%(asctime)s - [%(levelname)s] - (%(filename)s:%(funcName)s:%(lineno)d) - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            field_styles=cls.FIELD_STYLES,
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        local_logger.addHandler(handler)

        return local_logger


logger = LoggerFactory().get_logger(level=logging.INFO)
