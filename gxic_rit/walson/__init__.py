# -*- coding: utf-8 -*-
import logging
import os

import imagehash
import numpy as np
import openface
from PIL import Image
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from . import db


logger = logging.getLogger(__name__)

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


class Walson(object):
    """
    Face 数据库
    """

    def __init__(self, landmarks='outerEyesAndNose',
                 dlib_predictor_db="shape_predictor_68_face_landmarks.dat",
                 face_network_model="nn4.small2.v1.t7",
                 face_size=96, skip_multi=False, cuda=False):
        self.faces = {}
        self.svm = None
        self.people_db = None

        self.face_size = face_size
        landmarkMap = {
            'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
            'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
        }
        if landmarks not in landmarkMap:
            raise Exception("Landmarks unrecognized: {}".format(landmarks))
        self.landmarkIndices = landmarkMap[landmarks]

        dlibFacePredictor = os.path.join(dlibModelDir, dlib_predictor_db)
        if not os.path.isfile(dlibFacePredictor):
            raise Exception("Dlib Face Predictor Model not found at path {}".format(dlibFacePredictor))
        self.alignFilter = openface.AlignDlib(dlibFacePredictor)

        face_network_model_path = os.path.join(openfaceModelDir, face_network_model)
        if not os.path.isfile(face_network_model_path):
            raise Exception("Face network Model not found at path {}".format(face_network_model_path))
        self.face_network = openface.TorchNeuralNet(face_network_model_path,
                                                    imgDim=self.face_size,
                                                    cuda=cuda)

    def trainSVM(self):
        """
        根据已有的 self.faces 内容进行 training
        """
        # FIXME: take data from db
        logger.info("Training SVM on {} labled images.".format(len(self.faces)))
        d = self.getData()
        if d is None:
            self.svm = None
            self.le = None
            return

        (X, y) = d
        numIdentities = len(set(y + [-1]))
        if numIdentities <= 1:
            return
        param_grid = [
            {'C': [1, 10, 100, 1000],
             'kernel': ['linear']},
            {'C': [1, 10, 100, 1000],
             'gamma': [0.001, 0.0001],
             'kernel': ['rbf']}
        ]
        self.le = LabelEncoder().fit(y)
        self.svm = GridSearchCV(SVC(C=1), param_grid, cv=5).fit(X, y)

    def getData(self):
        """
        从数据库中拿出 visitors, 生成用于监督训练的结构

        Returns:
            X: list, 人脸特征值列表, 每一个元素对应一个人
            Y: list, 特征值打标数据, 监督打标用
        """
        self.people_db = {}
        X, y = [], []
        for people in db.get_all_visitors(db.get_engine()):
            self.people_db[people.pid] = people
            for eigen in people.eigens:
                X.append(eigen)
                y.append(people.pid)

        if len(set(y)) == 0:
            return None

        X = np.vstack(X)
        y = np.array(y)
        return (X, y)

    def predict(self, eigen):
        """
        Args:
            imgObject:
        """
        predictions = self.svm.predict_proba(eigen).ravel()
        max_index = np.argmax(predictions)
        people = self.find_people(max_index)
        confidence = predictions[max_index]
        result = {
            'id': people.pid,
            'name': people.name,
            'confidence': confidence,
        }
        return result

    def find_people(self, index):
        """
        根据 index 找出 people

        returns:
            konan.people.People
        """
        person_id = self.le.inverse_transform(index)
        return self.people_db[person_id]
