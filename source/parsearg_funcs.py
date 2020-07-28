# -*- coding: utf-8 -*-

"""
parsearg_funcs.py

created: 17:13 - 23.07.20
author: kornel 
"""

# --------------------------------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------------------------------

# stamdard lib imports
import os.path
import argparse
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# 3rd party imports

# project imports


# --------------------------------------------------------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------------------------------------------------------

class CsvFileAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # norming path:
        pth = os.path.abspath(values)
        # checking if existing file and correct filetype
        if not os.path.isfile(pth):
            err = '%s is not a path to an existing file' % pth
            logger.error(err)
            raise argparse.ArgumentTypeError(err)
        if os.path.splitext(pth)[-1] != '.csv':
            err = '%s is not a path to an existing file' % pth
            logger.error(err)
            raise argparse.ArgumentTypeError(err)
        # append to namespace
        logger.debug('appending %s to namespace' % pth)
        setattr(namespace, self.dest, pth)


def check_sqlite3(pth):
    """ checks if user supplied a path to an existing sql3-database and norms it """
    # norming path:
    pth = os.path.abspath(pth)
    # checking if existing file and correct filetype
    if not os.path.isfile(pth):
        err = '%s is not a valid path to an existing file' % pth
        logger.error(err)
        raise argparse.ArgumentTypeError(err)
    # SQLite database file header is 100 bytes
    if os.path.getsize(pth) < 100:
        err = '%s is not a valid sqlite3-database' % pth
        logger.error(err)
        raise argparse.ArgumentTypeError(err)
    with open(pth, 'rb') as fin:
        header = fin.read(100)
        if not header[:16] == b'SQLite format 3\x00':
            err = '%s is not a valid sqlite3-database' % pth
            logger.error(err)
            raise argparse.ArgumentTypeError(err)
    return pth


def check_csv(pth):
    """ checks if user supplied a path to an existing file with a csv-extension and norms it """
    logger.info('test')
    # norming path:
    apth = os.path.abspath(pth)
    # checking if existing file and correct filetype
    if not os.path.isfile(apth):
        err = '%s is not a path to an existing file' % apth
        logger.error(err)
        raise argparse.ArgumentTypeError(err)
    # simple (not sufficient) check for the correct file-extension
    # it is assumed that if the user supplies a file with .csv it is actually one
    if os.path.splitext(apth)[-1] != '.csv':
        err = '%s is not a a valid csv-file' % apth
        logger.error(err)
        raise argparse.ArgumentTypeError(err)
    logger.debug('transformed %s to %s' % (pth, apth))
    return apth


def check_dir_existing(pth):
    """ checks if path is or contains an existing directory and norms it """
    # norming path:
    pth = os.path.abspath(pth)
    # checking if existing file and correct filetype
    if not os.path.isdir(os.path.dirname(pth)):
        err = '%s is not a valid path to an existing directory' % pth
        logger.error(err)
        raise argparse.ArgumentTypeError(err)
    return pth
