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

    def add_people(self, alignedFaceRgb, pid):
        """
        数据库中添加一个包含头像的图片

        Args:
            alignedFaceRgb: 人脸并且 aligned 过的 rgb 三维矩阵
        """
        rep = self.face_network.forward(alignedFaceRgb)
        phash = str(imagehash.phash(Image.fromarray(alignedFaceRgb)))
        face = (rep, alignedFaceRgb, pid)
        self.faces[phash] = face

    def trainSVM(self):
        """
        根据已有的 self.faces 内容进行 training
        """
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
        生成用于监督训练的结构

        Returns:
            X: list, 人脸特征值列表, 每一个元素对应一个人
            Y: list, 特征值打标数据, 监督打标用
        """
        X, y = []
        for face in self.faces.values():
            X.append(face.rep)
            y.append(face.people.id)

        if len(set(y)) == 0:
            return None

        X = np.vstack(X)
        y = np.array(y)
        return (X, y)

    def predict(self, reps, multi=True):
        """
        Args:
            imgObject:

        Returns:
            [[bb, people, confidence], ...]
        """
        results = []
        for bb, rep in reps:
            predictions = self.svm.predict_proba(rep).ravel()
            max_index = np.argmax(predictions)
            people = self.find_people(max_index)
            confidence = predictions[max_index]
            results.append([bb, people, confidence])

        return results

    def find_people(self, index):
        """
        根据 index 找出 people
        """
        person_id = self.le.inverse_transform(index)
        # FIXME how to find people
