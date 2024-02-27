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
def predict_api():
    data = request.json["data"]
    print(f" inserted data {data}")
    print(f" correct data {np.array(list(data.values())).reshape(1,-1)}")
    new_data= scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify (output[0])


@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    print('Form data')
    
    new_data = scalar.transform(np.array(data).reshape(1,-1))
    output =  regmodel.predict(new_data)
    print(output)
    
    return render_template('home.html', predict_value = 'The predict value is {}'.format(round(output[0],2)))
    
    
if __name__ == "__main__":
    app.run(debug=True)
   
    
    
    