# -*- coding: utf-8 -*-
import cPickle as pickle
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
    params = GenerateEigenParams(request.get_json(()))
    #print("receive params: {}".format(smart_log(params)))
    #pickle.dump(params, open(os.path.join(
    #    module_dir, '..', 'tmp', '{}.pickle'.format(time.time())), 'w'))

    frame = image.data_uri_to_cv2_img(params['data_url'])
    bb = faceapi.largest_face_bounding_boxes(frame)
    if bb is None:
        return jsonify(success=False, data_url=params['data_url'])

    landmarks = faceapi.findLandmarks(frame, bb)
    face = faceapi.align(frame, bb, landmarks=landmarks)
    eigen = faceapi.getEigen(face)
    image.printFaceBox(frame, 1, bb)
    image.printName(frame, params["people"]['name'], 1, bb)
    result = {
        "data_url": image.cv2_img_to_data_uri(frame),
        "success": True,
        "eigen": eigen.tolist()
    }
    #print("response with body: {}".format(smart_log(result)))
    return jsonify(result)
