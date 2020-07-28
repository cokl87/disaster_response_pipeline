# -*- coding: utf-8 -*-

"""
etl_pipeline.py

created: 12:38 - 23.07.20
author: kornel
"""

# --------------------------------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------------------------------

# stamdard lib imports
import os.path
import sqlite3
import argparse
import sys

# 3rd party imports
import pandas as pd

# project imports
import parsearg_funcs
from log_config import get_configured_logger
logger = get_configured_logger('default', level='DEBUG')

# --------------------------------------------------------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------------------------------------------------------


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

    description = '''
    ETL pipeline which extracts data from two csv-files, transform and cleans this 
    data and loads it in a database.
    '''
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

    # the last character contains the boolean in form of 0 or 1
    categories = categories.applymap(lambda x: int(x[-1]))

    # concatenate the original dataframe with the new `categories` dataframe
    data.drop('categories', axis=1, inplace=True)
    data = pd.concat([data, categories], axis=1)

    # dropping duplicate data
    return data.drop_duplicates(inplace=False)


def extract_data(csv1, csv2):
    """
    extract routine which extract message and category data from two csv files
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


def main():
    """ main routine of script """
    logger.info(__file__)

    logger.info('parsing arguments...')
    args = parse_args(sys.argv[1:])

    logger.info('reading data...')
    messages, categories = extract_data(args.csv1, args.csv2)

    logger.info('transforming data...')
    data = transform_data(messages, categories)

    logger.info('loading data into database...')
    load_df2db(data, args.db, args.table)


if __name__ == '__main__':
    main()
