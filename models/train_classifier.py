import sys
import pickle
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

import pandas as pd
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    '''
    Load data from database as dataframe
    Input:
        database_filepath: File path of sql database
    Output:
        X: Message data 
        Y: Categories data
        category_names: Labels for 36 categories
    '''
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('data', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names=Y.columns.values
    return X,Y,category_names

def tokenize(text):
    '''
    Tokenize and clean text
    Input:
        text: original message text
    Output:
        words: Tokenized text
    '''
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    words=word_tokenize(text)
    # Reduce words to their stems
    stemmer = PorterStemmer()
    # Remove stop words
    words = [stemmer.stem(w) for w in words if w not in stopwords.words("english")]
    return words


def build_model():
    '''
    Build a ML pipeline using ifidf, random forest, and gridsearch
    Input: None
    Output:
        Results of GridSearchCV
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'clf__estimator__min_samples_split': [2, 3]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model performance using test data
    Input: 
        model: Model to be evaluated
        X_test: Test data
        Y_test: True lables for Test data
        category_names: Labels for 36 categories
    Output:
        Print accuracy and classfication report for each category
    '''
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test.iloc[:,1:].values, np.array([x[1:] for x in Y_pred]), target_names=category_names))

def save_model(model, model_filepath):
    '''
    Save model as a pickle file 
    Input: 
        model: Model to be saved
        model_filepath: path of the output pick file
    Output:
        A pickle file of saved model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()