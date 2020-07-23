# -*- coding: utf-8 -*-

"""
log_config.py contains a function for the standard configuration of the logging-module.
Currently a logger for stdout-stream and a default for logging into files and stdout are configured.

created: 30.06.2020
author: kornel
"""

import logging
from logging.config import dictConfig
import os.path


def get_configured_logger(name, filename=None, verbose=True):
    """
    function configures logging and returns the requested logger

    Parameters:
    -----------
    name: str
        name of the logger to be returned
    filename: str
        path of the file where the logging shall take place
    verbose: bool
        which formatter to use (verbose messages or taciturn)

    Returns:
    --------
    logging.Logger

    """

    verbosity = 'verbose' if verbose else 'taciturn'
    if not filename:
        filename = os.path.join(os.getcwd(), 'log.txt')

    config_dict = {
        'version': 1,
        'formatters': {
            'verbose': {'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        'datefmt': '%Y-%m-%d %H:%M:%S'},
            'taciturn': {'format': '%(message)s'},
        },
        'handlers': {
            'stdout': {
                'level': 'DEBUG',
                'formatter': verbosity,
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',
            },
            'file_handler': {
                'level': 'WARNING',
                'formatter': verbosity,
                'filename': filename,
                'mode': 'a',
                'class': 'logging.handlers.RotatingFileHandler',
                'maxBytes': 102400,
                'backupCount': 3
            }
        },
        'loggers': {
            'default': {
                'level': 'INFO',
                'handlers': ['stdout', 'file_handler'],
            },
            'stdout': {
                'level': 'INFO',
                'handlers': ['stdout']
            },
        },
        'disable_existing_loggers': False
    }

    dictConfig(config_dict)
    return logging.getLogger(name)
