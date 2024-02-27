import pickle
import numpy as np
import pandas as pd

from flask import Flask, request, render_template, url_for ,app,jsonify

app= Flask(__name__)

#laod model
regmodel = pickle.load(open('linear_reg.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict():
    data = request.json["data"]
    print(f" inserted data {data}")
    print(f" correct data {np.array(list(data.values())).reshape(1,-1)}")
    new_data= scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0])
    
    return jsonify (output[0])
    
    
if __name__ == "__main__":
    app.run(debug=True)
   
    
    
    