# Disaster Response Pipeline

### Table of Contents
1. [Installation](#installation)
2. [Project Description](#project-description)
2. [File Descriptions](#file-descriptions)
3. [Results](#results)
4. [Licensing, Authors, Acknowledgements](#licensing,-authors,-acknowledgements)


## Installation
For the analysis python3.8 was used. For getting the pipelines running, install the required packages listed in [requirements.txt](./requirements.txt) with `pip install -r requirements.txt`
For running the webapp additional packages like flask are needed. The requirements can be found in [./app/requirements.txt](./app/requirements.txt)

## Project Description
The project is divided into three parts:
1. An ETL-Pipeline, which reads data from csv-files, transforms the data so that it suffices the needs for a NLP- and ML-pipeline and stores this data into a database for later use.
2. An NLP- and ML-pipeline, which loads the messages from the database, process these messages so that they can be used for machine learning alogithmns and then trains a model on the data.
3. A web-app which displays some statistics about the data used for training the data and deploys the trained model to a flask-app where users can insert and classify individual disaster messages.

## File Descriptions
One can find 4 directories in the project.
#### source:
* [etl_pipeline.py](./source/train.py) - etl-pipeline which creates the cleaned and aggregated data for the ml-pipeline
* [train.py](./source/train.py) - nlp-ml-pipeline which processes the disaster-messages and traines a model
* [log_config.py](./source/log_config.py) - includes function for configuration of logging module
* [logging.json](./source/logging.json) - json file where loggers, handlers and formatters are defined
* [parsearg_funcs.py](./source/parsearg_funcs.py) - basic functions used for checking user-input
#### data:
This folder contains two csv files which are read, transformed and loaded into a database by the etl-pipeline.
The so transformed data is stored in the data.db.
#### models:
The trained models were too large in order to include them in the github repo. A trained model for running the flask-app can be created by running train.py or downloaded [here](https://www.dropbox.com/sh/6pnezik72wtqkhz/AADofsAGoEG-rQ3_K74CyltDa?dl=0)
#### app:
contains the files for the webapp. The run.py runs the flask-app and defines the views. In templates you can find the HTML-templates.

## Results
The output of the etl-pipeline can be found in the data-folder:
The trained model in form of a pickle-file of the nlp-ml-pipeline was too large to include it in the repository and can be therefor found [here](https://www.dropbox.com/sh/6pnezik72wtqkhz/AADofsAGoEG-rQ3_K74CyltDa?dl=0) or created by yourself by running the script train.py.

## Licensing, Authors, Acknowledgements
The project was part of the [Udacity's DataScientist program]('https://www.udacity.com/course/data-scientist-nanodegree--nd025'). The html-templates for the web-app are strongly based on resources given by Udacity. You may use the code of this porject as you like.
