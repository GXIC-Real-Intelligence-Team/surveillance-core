# -*- coding: utf-8 -*-
import voluptuous as vol
from flask import Flask, request, jsonify

from gxic_rit import db
from gxic_rit.walson import Walson


app = Flask(__name__)
walson = Walson()
walson.trainSVM()


@app.route('/retrain_svm', methods=['POST'])
def retrain():
    walson.trainSVM()
    return jsonify(success=True)


PredictParams = vol.Schema([
    {
        "seq": int,
        "eigen": [float],
    }
], required=True)


@app.before_request
def log_request_info():
    app.logger.debug('Headers: %s', request.headers)
    app.logger.debug('Body: %s', request.get_data())


@app.route('/predict', methods=['POST'])
def predict():
    params = PredictParams(request.get_json())
    app.logger.info('get {} eigens to classify'.format(len(params)))
    for info in params:
        info['people'] = walson.predict(info['eigen'])
    return jsonify(params)


AlarmParams = vol.Schema({"visitor_id": int, "camera_id": int}, required=True)


@app.route('/add_alarm', methods=['POST'])
def add_alarm():
    params = AlarmParams(request.get_json())
    return jsonify(
        db.add_alarm(visitor_id=params['visitor_id'],
                     camera_id=params['camera_id']))
