# -*- coding: utf-8 -*-

"""
run.py script for running flask-app
"""

# --------------------------------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------------------------------

# stamdard lib imports
import json
import os.path
import sys
import logging

# 3rd party imports
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Pie, Bar
import joblib

# project imports
sys.path.append('../source')
from log_config import config_logging
from train import load_data, tokenize, stem, lemmatize, VerbAtStartExtractor


# --------------------------------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------------------------------

DB_PTH = '../data/data.db'
TABLE_NAME = 'categorized'
MODEL_PTH = '../models/model.pkl'
DEBUG = False
HOST = '0.0.0.0'
PORT = 3001


# --------------------------------------------------------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------------------------------------------------------

app = Flask(__name__)

# load data
df = load_data(DB_PTH, TABLE_NAME, split=False)

# load model
model = joblib.load(MODEL_PTH)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_df = df.groupby('genre').count().reset_index('genre')

    categories = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    csums = (categories.sum()*100 / categories.shape[0]).sort_values(ascending=False)
    
    # create visuals
    graphs = [
        {
            'data': [
                Pie(
                    values=genre_df['message'],
                    labels=genre_df['genre'],
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    y=list(csums.values),
                    x=[label.replace('_', ' ') for label in csums.index],
                )
            ],
            'layout': {
                'title': 'Categories of messages sorted by frequency',
                'yaxis': {
                    'title': "Percent of messages with category"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 30,
                },
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """ main routine """
    app.run(host=HOST, port=PORT, debug=DEBUG)


if __name__ == '__main__':
    # configure logging
    config_logging(
        os.path.join(os.path.dirname(__file__), '../source/logging.json')
    )
    logger = logging.getLogger(__name__)
    # call main routine
    logger.info('Starting web-app...')
    main()
