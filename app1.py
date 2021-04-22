# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 20:20:22 2021

@author: Hp
"""
import numpy as np
import pickle
from flask import Flask, request, render_template

# Load ML model
model = pickle.load(open('model6_Tuned.pkl', 'rb')) 

# Create application
app = Flask(__name__)

# Bind home function to URL
@app.route('/')
def home():
    return render_template('Heart Disease.html')

# Bind predict function to URL
@app.route('/predict', methods =['POST'])
def predict():
    
    # Put all form entries values in a list 
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction = model.predict(array_features)
    
    output = prediction
    
    # Check the output values and retrive the result with html tag based on the value
    if output == 0:
        return render_template('Heart Disease.html', 
                               result = 'The patient is not likely to have heart disease!')
    else:
        return render_template('Heart Disease.html', 
                               result = 'The patient is likely to have heart disease!')

if __name__ == '__main__':
#Run the application
    app.run()
    
    
    return parameters

@app.route(‘/predict’,methods=[‘POST’])
def predict():
    model = open(“data/Heart_model.pkl”,”rb”)
    clfr = joblib.load(model)
    if request.method == ‘POST’:
        parameters = getParameters()
        inputFeature = np.asarray(parameters).reshape(1,-1)
        my_prediction = clfr.predict(inputFeature)
    return render_template(‘result.html’,prediction = int(my_prediction[0]))


if __name__ == ‘__main__’:
app.run(debug=True)