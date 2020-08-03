# -*- coding: utf-8 -*-

"""
train.py - NLP and ML pipeline which trains a predictor on data loaded in a database.

created: 15:03 - 28.07.20
author: kornel
"""

# --------------------------------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------------------------------

# stamdard lib imports
import sqlite3
import argparse
import sys
import logging
import os.path
import re
import pickle

# 3rd party imports
import pandas as pd
from numpy import nan

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report  # confusion_matrix

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# project imports
import parsearg_funcs
from log_config import config_logging

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


# --------------------------------------------------------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------------------------------------------------------

def main():
    """ main routine of script """
    logger.info(__file__)

    logger.info('parsing arguments...')
    args = parse_args(sys.argv[1:])

    logger.info('Loading data from database: "%s.%s"', args.db, args.table)
    parameters, categories, category_names = load_data(args.db, args.table)
    x_train, x_test, y_train, y_test = train_test_split(parameters, categories, test_size=0.2)

    logger.info('Building model...')
    model = build_model()

    logger.info('Training model andfinding best parameter set...')
    model.fit(x_train, y_train)

    logger.info('Evaluating model...')
    evaluate_model(model, x_test, y_test, category_names)

    logger.info('Saving model on: "%s"', args.model)
    save_model(model.best_estimator_, args.model)


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
    parser.add_argument(
        'db', type=parsearg_funcs.check_sqlite3, metavar='database',
        help='path to the sqlite3-database where input-data for ml-pipeline is stored'
    )
    parser.add_argument(
        'model', type=parsearg_funcs.check_dir_existing, metavar='model',
        help='path and name where trained model will be stored'
    )
    parser.add_argument(
        '-t', '--table', nargs=1, dest='table', metavar='tablename', type=str,
        default='categorized', help='name of the table where transformed data will be loaded from')
    return parser.parse_args(args)


def load_data(db_pth, table, split=True):
    """
    loads data from a sqlite3-database

    Parameters
    ----------
    db_pth: str
        path to the sqlite3-database
    table: str
        name of the table to load
    split: bool
        return table as is or split in parameters and categories

    Returns
    -------
    text-messages: pandas.DataFrame
    categories: pandas.DataFrame
    category-names: numpy.array

    """
    con = sqlite3.connect(db_pth)
    data = pd.read_sql_query("SELECT * FROM %s" % table, con)
    con.close()
    if not split:
        return data

    parameters = data.message
    categories = data.drop(['id', 'message', 'original', 'genre'], axis=1)
    return parameters, categories, categories.columns.values


def tokenize(text):
    """
    tokenizes, norms and removes punctuation and stopwords from english text-messages.

    Parameters
    ----------
    text: str
        text-message to be tokenized

    Returns
    -------
    cleaned: list
        list with tokenized words.
    """
    # norm and tokenize
    normed = re.sub('[^a-zA-Z0-9]', ' ', text.lower())
    tokens = word_tokenize(normed)
    # remove stopwords
    cleaned = [word for word in tokens if word not in stopwords.words("english")]
    # TODO: deal with patterns e.g. urls:
    return cleaned


def stem(text):
    """
    transforms english messages to their stemmed-form.

    Parameters
    ----------
    text: str
        text messages to be stemmed

    Returns
    -------
    list
        list of stemmed words
    """
    post = PorterStemmer()
    return [post.stem(word) for word in tokenize(text)]


def lemmatize(text):
    """
    tokenize and lemmatize english messages

    Parameters
    ----------
    text: str
        text messages to be lemmatized

    Returns
    -------
    list
        list with lemmatized forms of words
    """
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        if treebank_tag.startswith('V'):
            return wordnet.VERB
        if treebank_tag.startswith('N'):
            return wordnet.NOUN
        if treebank_tag.startswith('R'):
            return wordnet.ADV
        # try to transfer to Noun else
        # else:
        return wordnet.NOUN

    # lemmatize
    wordpos = nltk.pos_tag(tokenize(text))
    lmtzer = WordNetLemmatizer()
    return [lmtzer.lemmatize(word, pos=get_wordnet_pos(pos)) for word, pos in wordpos]


class VerbAtStartExtractor(BaseEstimator, TransformerMixin):
    """
    Class that preprocess text-messages for usage in a ml-pipeline. Checks if in any sentence a
    verb is at first possition.

    Attributes
    ----------

    Methods
    -------
    fit
    starting_verb
    transform
    """
    def fit(self, x, y):
        """ implementation of fit """
        return self

    @staticmethod
    def starting_verb(text):
        """
        checks if in text a verb is at first possition.

        Parameters
        ----------
        text: str

        Returns
        -------
        bool
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            # omit sentences with only stop words or punctuations
            if not pos_tags:
                continue
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def transform(self, data):
        """
        implementation of transform-method. Applies starting-verb on Data and returns a DataFrame
        with boolean data whether a message contains a sentence with a verb at first possition.

        Parameters
        ----------
        data: iterable
            iterable with text-messages to be analysed

        Returns
        -------
        pandas.DataFrame
            DataFrame with boolean values
        """
        x_tagged = pd.Series(data).apply(self.starting_verb)
        return pd.DataFrame(x_tagged)


def build_model():
    """
    builds a nlp- and ml-pipeline which classifies disaster-response messages to disaster-categories

    Returns
    -------
    sklearn.model_selection.GridSearchCV
    """
    pipeline = Pipeline([
        ('feature_prep', FeatureUnion([
            ('msg_pipeline1', Pipeline([
                ('countvec', CountVectorizer(
                    max_df=0.95, max_features=10000, min_df=1, tokenizer=lemmatize)),
                ('tfidf', TfidfTransformer()),
            ])),
            ('msg_pipeline2', Pipeline([
                ('countvec', CountVectorizer(
                    max_df=0.95, max_features=10000, min_df=1, tokenizer=stem)),
                ('tfidf', TfidfTransformer()),
            ])),
            ('ext', VerbAtStartExtractor()),
        ], transformer_weights={'msg_pipeline1': 1, 'ext': 0.5, 'msg_pipeline2': 1})),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    parameters = {
        #'feature_prep__msg_pipeline1__countvec__ngram_range': ((1, 1), (1, 2)),
        #'feature_prep__msg_pipeline1__countvec__max_df': (0.5, 0.75, 1.0),
        #'feature_prep__msg_pipeline1__countvec__max_features': (None, 5000, 10000),
        #'feature_prep__msg_pipeline1__tfidf__use_idf': (True, False),
        #'clf__estimator__n_estimators': [100, 120],
        #'clf__estimator__min_samples_split': [2, 3, 4],
        'feature_prep__transformer_weights': (
            #{'msg_pipeline1': 1, 'ext': 0.5, 'msg_pipeline2': 0},
            {'msg_pipeline1': 1, 'ext': 0.5, 'msg_pipeline2': 1},
            {'msg_pipeline1': 1, 'ext': 0.2, 'msg_pipeline2': 0.5},
            {'msg_pipeline1': 1, 'ext': 0.5, 'msg_pipeline2': 0.75},
            #{'msg_pipeline1': 0.8, 'ext': 1, 'msg_pipeline2': 1},
            #{'msg_pipeline1': 1, 'ext': 0.8, 'msg_pipeline2': 1},
        )
    }
    logger.debug('\n'.join(pipeline.get_params().keys()))
    cv = GridSearchCV(pipeline, param_grid=parameters, error_score=nan, verbose=10, n_jobs=1)
    return cv


def evaluate_model(model, x_test, y_test, category_names):
    """
    evaluates model by calulation of f1-score, precisions and recall-rates of the different classes.
    Results will be logged.

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        trained model to be evaluated
    x_test: array-like, shape (n_samples, n_features)
        parameters (disaster-messages) of test-data
    y_test: array, shape (n_samples,)
        categories of test-data
    category_names: iterable

    Returns
    -------
    None
    """
    logger.info('optimal parameters: %s; score: %s', model.best_params_, model.best_score_)
    y_pred = model.predict(x_test)
    pred_col_list = y_pred.transpose()
    for idx, (name, vals) in enumerate(y_test.iteritems()):
        logger.info(
            '%s:\n%s', name, classification_report(vals, pred_col_list[idx])
        )


def save_model(model, mod_pth):
    """
     saves model as pickle for later usage

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
    mod_pth: str
        output-path

    Returns
    -------
    None
    """
    with open(mod_pth, 'wb') as fout:
        pickle.dump(model, fout)


if __name__ == '__main__':
    # configure logging
    config_logging(
        os.path.join(os.path.dirname(__file__), './logging.json')
    )
    logger = logging.getLogger(__name__)
    # call main routine
    logger.info('Starting NLP-ML-Pipeline...')
    main()
