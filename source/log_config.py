# -*- coding: utf-8 -*-

"""
log_config.py contains a function for the standard configuration of the logging-module.

created: 30.06.2020
author: kornel
"""

import logging.config
import os.path
import json


def config_logging(config_path='logging.json', default_level=logging.INFO,
                   default_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
    """
    loads logging configuration from json

    Parameters
    ----------
    config_path: str
        path of the json with logger-configuration
    default_level: int
        int representing logging-level of default logger (in case json can not be loaded)
    default_format: str
        logger-format-str of default logger (in case json can not be loaded)

    Returns
    -------
    None
    """
    if os.path.exists(config_path):
        with open(config_path, 'rt') as fin:
            config = json.load(fin)

        # TODO: prepend a default-logdir to file-handlers with relative pathes

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=default_level,
            format=default_format,
        )
        logger = logging.getLogger(__name__)
        logger.warning(
            'logging config file "%s" not found. Logging runs with default logger', config_path
        )
