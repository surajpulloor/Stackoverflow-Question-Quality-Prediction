from flask import Flask
app = Flask(__name__)

from flask import render_template
from flask import request
import re

import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import stats

# Load Vectorizer
f = open('../models/vectorizer', 'rb')
vectorizer = pickle.load(f)

# LOad all the models
f = open('../models/non_gridcv/bagging', 'rb')
bagging = pickle.load(f)

f = open('../models/non_gridcv/decision_tree', 'rb')
decisionTree = pickle.load(f)

f = open('../models/non_gridcv/logistic_reg', 'rb')
logisticReg = pickle.load(f)

f = open('../models/non_gridcv/mlp', 'rb')
mlp = pickle.load(f)

f = open('../models/non_gridcv/naive_bayes', 'rb')
naiveBayes = pickle.load(f)

f = open('../models/non_gridcv/random_forest', 'rb')
randomForest = pickle.load(f)

f = open('../models/non_gridcv/stack', 'rb')
stack = pickle.load(f)

f = open('../models/non_gridcv/svc', 'rb')
svc = pickle.load(f)

f = open('../models/non_gridcv/voting', 'rb')
voting = pickle.load(f)

f = open('../models/non_gridcv/xgboost', 'rb')
xgboost = pickle.load(f)

@app.route('/')
def hello(name=None):
    return render_template('index.html', name=name)


def cleanData(title, body):

    if title and body:
        text = title + ' ' + body
        text = text.lower()
        text = re.sub(r'[^(a-zA-Z)\s]', '', text)
        
        return text

    return None


@app.route('/', methods=['POST'])
def login():
    if request.method == 'POST':
        
        text = cleanData(request.form['questionTitle'], request.form['questionBody'])

        X = vectorizer.transform(text.split()).toarray()

        # Remove predictions 

        predictions = [
            stats.mode(bagging.predict(X))[0][0],
            stats.mode(decisionTree.predict(X))[0][0],
            stats.mode(logisticReg.predict(X))[0][0],
            stats.mode(mlp.predict(X))[0][0],
            stats.mode(naiveBayes.predict(X))[0][0],
            stats.mode(randomForest.predict(X))[0][0],
            stats.mode(stack.predict(X))[0][0],
            stats.mode(svc.predict(X))[0][0],
            stats.mode(voting.predict(X))[0][0],
            stats.mode(xgboost.predict(X))[0][0],
        ]
        


        # Build a dictionary of the models
        models = {
            'Bagging': [bagging.best_score, predictions[0]],

            'DecisionTree': [ decisionTree.best_score,  predictions[1]],

            'LogisticReg': [ logisticReg.best_score, predictions[2]],

            'MLP': [ mlp.best_score,  predictions[3]],

            'NaiveBayes': [naiveBayes.best_score, predictions[4]],

            'RandomForest': [randomForest.best_score, predictions[5]],

            'Stacking': [stack.best_score, predictions[6]],

            'SVC': [svc.best_score, predictions[7]],

            'Voting': [voting.best_score, predictions[8]],

            'XGBoost': [xgboost.best_score,  predictions[9]]       
        }

        content = {
            'text': text,
            'models': models
        }

        return render_template('index.html', **content)