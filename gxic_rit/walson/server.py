# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import voluptuous as vol
from gxic_rit.walson import Walson
from copy import deepcopy


app = Flask(__name__)
walson = Walson()


@app.route('/retrain_svm')
def retrain():
    return "hello"


PredictParams = vol.Schema([
    {
        "seq": int,
        "eigen": [int],
    }
], required=True)


@app.route('/predict')
def predict():
    params = PredictParams(request.get_json())
    for info in params:
        info['people'] = walson.predict(info['eigen'])
    return jsonify(params)
