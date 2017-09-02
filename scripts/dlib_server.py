# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import voluptuous as vol
from gxic_rit.konan import image
from gxic_rit.konan import faceapi

app = Flask(__name__)


GenerateEigenParams = vol.Schema({
    "data_url": basestring,
    "people": {
        "name": basestring,
    }
}, required=True)


@app.route('/generate_eigen')
def generate_eigen():
    params = GenerateEigenParams(request.get_json(()))
    frame = image.data_uri_to_cv2_img(params['data_url'])
    bb = faceapi.largest_face_bounding_boxes(frame)
    if bb is None:
        return jsonify(success=False, data_url=params['data_url'])

    landmarks = faceapi.findLandmarks(frame, bb)
    face = faceapi.align(frame, bb, landmarks=landmarks)
    eigen = faceapi.getEigen(face)
    image.printFaceBox(frame, 1, bb)
    image.printName(frame, params["people"]['name'], 1, bb)
    return jsonify({
        "data_url": image.cv2_img_to_data_uri(frame),
        "success": True,
        "eigen": eigen
    })


