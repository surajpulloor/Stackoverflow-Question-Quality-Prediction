# Stackoverflow-Question-Quality-Prediction

## Abstract
 Stackoverflow.com is a popular Q/A platform for programmers to solve their programming queries.Here the community can rate a question as high/low quality.This enables stackoverflow to maintain high standard of questions.But a user posting a new question has no clue whether the question will be accepted or rejected.Therefore the aim of this project is to predict the quality of the question using ML models so that the user can make a calculated guess before posting the question.
 
 
## File Structure
* data_analysis.ipynb \[ Jupyter Notebook for performing cleaning & data analysis on the stackoverflow data \]
* stackoverflow_question_quality.ipynb \[ Jupyter Notebook for performing ML model training and evaluation. We select the best model here \]
* interface \[ Flask app to enable easy access to our models for question quality prediction for the user \]
* data \[ Stackoverflow.com question dataset from Kaggle \]
 
 
## Instructions
Before running the flask app using the start.sh(linux/mac)/start.bat(windows) in the interface directory, make sure you run stackoverflow_question_quality.ipynb file and save the models as pickle file. They are necessary for the proper functionality of the app.
 

## Software Requirements:
* Windows/Linux/Mac
* Python -3.9.1
* VSCode - With Python Extension
* Libraries:
   * Numpy -1.18.2
   * Pandas -1.2.1
   * Matplotlib -3.3.3
   * Scikit-learn -0.24.1
   * Flask -1.1.2
   * Jupyterlab
