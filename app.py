import pickle
from flask import Flask,request,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
dtmodel = pickle.load(open('dtmodel.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(list(data.values()))
    new_data = np.array(list(data.values())).reshape(1,-1)
    output= dtmodel.predict(new_data)
    print(output[0])
    return jsonify(float(output[0]))

if __name__=="__main__":
    app.run(debug=True)
