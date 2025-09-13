import pickle 
from flask import Flask, request, app,jsonify, url_for,render_template
import numpy as np
import pandas as pd


app = Flask(__name__)
## Load the model
regmodel = pickle.load(open('regmodel.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

# def predict_api():
#     data=request.json['data']
#     print(data)
#     print(np.array(list(data.values())).reshape(1,-1))
#     new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1) )
#     output = regmodel.predict(new_data)
#     print(output[0])
#     return jsonify(output[0])


def predict_api():
    data = request.get_json(force=True)   # dict with key "data"
    record = data['data'][0]              # get the first item from the list
    features = np.array(list(record.values())).reshape(1, -1)
    
    new_data = scaler.transform(features)
    print(features)
    output = regmodel.predict(features)[0]
    return jsonify({'prediction': output})



if __name__=="__main__":
    app.run(debug=True)