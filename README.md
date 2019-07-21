# Disaster Response Pipeline Project

### Description
This project is part of Udaicty Data Science Nanodegree. The aim of the project is to build a Natural Language Processing tool that categorize messages.

The Project is divided in the following Sections:

Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper databse structure
Machine Learning Pipeline to train a model able to classify text message in categories
Web App to show model results in real time.

### Files:
process_data.py : ETL script write a data cleaning pipeline that:
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

train_classifier.py : script write a machine learning pipeline that:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

run.py : Main file to run Flask app that classifies messages based on the model and shows data visualizations.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
