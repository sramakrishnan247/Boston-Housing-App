from flask import Flask, render_template
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, train_test_split,learning_curve, validation_curve
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from flask import request

app = Flask(__name__)


@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods =['POST'])
def predict():
    rm = request.form.get('RM')
    lstat = request.form.get('LSTAT')
    ptratio = request.form.get('PTRATIO')

    values = str(rm)+str(lstat)+str(ptratio)

    filename = 'model_final.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    ans = str(loaded_model.predict([[int(rm),int(lstat),int(ptratio)]]))

    return "Price: "+ans


if __name__ == "__main__":
    app.run(debug=True)




