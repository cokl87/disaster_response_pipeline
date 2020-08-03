# -*- coding: utf-8 -*-

"""
etl_pipeline.py - ETL pipeline which extracts data from two csv-files, transform and cleans this
data and loads it in a database.

created: 12:38 - 23.07.20
author: kornel
"""

# --------------------------------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------------------------------

# standard lib imports
import sqlite3
import argparse
import sys
import logging
import os.path

# 3rd party imports
import pandas as pd
from numpy import nan

# project imports
import parsearg_funcs
from log_config import config_logging


# --------------------------------------------------------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------------------------------------------------------

def main():
    """ main routine of script """
    logger.info(__file__)

    logger.info('parsing arguments...')
    args = parse_args(sys.argv[1:])

    logger.info('reading data...')
    messages, categories = extract_data(args.csv1, args.csv2)

    logger.info('transforming data...')
    data = transform_data(messages, categories)

    logger.info('loading data into database "%s"', args.db)
    load_df2db(data, args.db, args.table)


def parse_args(args):
    """
    functions where argument parser is defined and arguments are parsed

    Parameters
    ----------
    args: list
        list with arguments (e.g. from sys.argv)

    Returns
    -------
    argparse.namespace
    """
    description = __doc__.strip().split('\n\n')[0]
    parser = argparse.ArgumentParser(description)
    parser.add_argument('csv1', metavar='messages.csv', type=parsearg_funcs.check_csv,
                        help='path to a csv-file with disaster messages')
    parser.add_argument('csv2', type=parsearg_funcs.check_csv, metavar='categories.csv',
                        help='path to csv-file with categories')
    parser.add_argument('db', type=parsearg_funcs.check_dir_existing, metavar='database',
                        help='name of the sqlite3-database where transformed data will be stored')
    parser.add_argument('-t', '--table', nargs=1, dest='table', metavar='tablename',
                        type=str, default='categorized',
                        help='name of the table where transformed data will be stored')
    return parser.parse_args(args)


def transform_data(messages, categories):
    """
    transformes data in form of two dataframes to be ready for loading into a database

    Parameters
    ----------
    messages: pandas.DataFrame
        data with messages
    categories: pandas.DataFrame
        data with categories of disasters

    Returns
    -------
    pandas.DataFrame
        transformed data
    """
    logger.info('merging data of disaster messages to category data')
    data = pd.merge(messages, categories, on='id')
    categories = data.categories.str.split(';', expand=True)

    # extract the category names:
    # select the first row of the categories dataframe
    row = categories.iloc[0, :]
    # row used to extract a list of new column names for categories. The form is colname-bool
    category_colnames = [val.split('-')[0] for val in row]
    categories.columns = category_colnames

    # the last character contains the boolean in form of 0 or 1 if not nan will be returned
    categories = categories.applymap(lastchar_to_01)

    # concatenate the original dataframe with the new `categories` dataframe
    data.drop('categories', axis=1, inplace=True)
    data = pd.concat([data, categories], axis=1)

    # drop all datapoints with missing categories or categories with invalid content
    logger.info(
        'extracted categories contain %i invalid entries which will be dropped',
        data[categories.columns].isna().sum().sum()
    )
    logger.debug(data.shape)
    data.dropna(axis=0, how='any', subset=categories.columns, inplace=True)
    logger.debug(data.shape)
    # dropping duplicate data
    return data.drop_duplicates(inplace=False)


def lastchar_to_01(string):
    """
    function extracts 0 or 1 from end of string. If last character is not 0 or 1 numpy.nan will be
    returned to indicate invalid data.

    Parameters
    ----------
    string: str
        str to extract integer from

    Returns
    -------
    numpy.nan, 0 or 1
    """
    string = string.strip()
    try:
        # extract last character and transform to int
        i_01 = int(string[-1])
    except (IndexError, ValueError):
        logger.debug('%s is empty string or not a number', string)
        return nan
    if i_01 not in [0, 1]:
        logger.debug('%s has invalid number %i', string, i_01)
        return nan
    return i_01


def extract_data(csv1, csv2):
    """
    extract routine which extract message and category data from two csv files

    Parameters
    ----------
    csv1: str
        pth to csv with text of disaster-messages
    csv2: str
        pth to csv with text of disaster-categories

    Returns
    -------
    pandas.DataFrame
        containing content csv-content with messages-text
    pandas.DataFrame
        containing content of csv with message-categories
    """
    logger.info('reading %s', csv1)
    messages = pd.read_csv(csv1)

    logger.info('reading %s', csv2)
    categories = pd.read_csv(csv2)
    return messages, categories


def load_df2db(data, db_name, table_name, if_exists='replace', use_index=False):
    """
    writes dataframe into database

    Parameters
    ----------
    data: pd.DataFrame
    db_name: str
    table_name: str
    if_exists: {'replace', 'fail', 'append'}
    use_index: bool

    Returns
    -------
    None

    """
    # create db-connection
    conn = sqlite3.connect(db_name)
    # write into db
    data.to_sql(table_name, conn, if_exists=if_exists, index=use_index)
    # save (commit) the changes and close db-connection
    conn.commit()
    conn.close()


if __name__ == '__main__':
    # configure logging
    config_logging(
        os.path.join(os.path.dirname(__file__), './logging.json')
    )
    logger = logging.getLogger(__name__)
    # call main routine
    logger.info('Starting ETL-Pipeline...')
    main()
