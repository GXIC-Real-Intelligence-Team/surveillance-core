# -*- coding: utf-8 -*-
import cPickle as pickle
import traceback
import os

import time
import voluptuous as vol
from flask import Flask, request, jsonify

from gxic_rit.konan import image, faceapi
from gxic_rit.common import smart_log


app = Flask(__name__)


GenerateEigenParams = vol.Schema({
    "data_url": basestring,
    "people": {
        "name": basestring,
    }
}, required=True)

module_dir = os.path.realpath(os.path.dirname(__file__))


@app.route('/generate_eigen', methods=["POST"])
def generate_eigen():
    try:
        params = GenerateEigenParams(request.get_json(()))
        #pickle.dump(params, open(os.path.join(
        #    module_dir, '..', 'tmp', '{}.pickle'.format(time.time())), 'w'))

        frame = image.data_uri_to_cv2_img(params['data_url'])
        bb = faceapi.largest_face_bounding_boxes(frame)
        if bb is None:
            return jsonify(success=False, data_url=params['data_url'])

        landmarks = faceapi.findLandmarks(frame, bb)
        face = faceapi.align(frame, bb, landmarks=landmarks)
        eigen = faceapi.getEigen(face)
        image.printFaceBox(frame, 1, bb, image.HIGH_COLOR)
        image.printName(frame, params["people"]['name'], 1, bb, image.HIGH_COLOR)
        result = {
            "data_url": image.cv2_img_to_data_uri(frame),
            "success": True,
            "eigen": eigen.tolist()
        }
        return jsonify(result)

    except Exception as exc:
        result = {
            "success": False,
            "error": str(exc.__class__.__name__),
            "traceback": traceback.format_exc()
        }
        print(result)
        resp = jsonify(result)
        resp.status_code = 400
        return resp
