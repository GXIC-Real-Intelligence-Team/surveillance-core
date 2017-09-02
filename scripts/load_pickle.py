# -*- coding: utf-8 -*-
import cPickle as pickle
from argparse import ArgumentParser
import numpy as np
import cv2

parser = ArgumentParser()
parser.add_argument('file_path')
options = parser.parse_args()


def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    np_arr = np.fromstring(encoded_data.decode('base64'), np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


obj = pickle.load(open(options.file_path))
print(obj['data_url'])
#data_uri_to_cv2_img(obj)

#import ipdb
#ipdb.set_trace()
