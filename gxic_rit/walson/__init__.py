# -*- coding: utf-8 -*-
import logging
import os
import time

import imagehash
import numpy as np
import openface
from PIL import Image
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

from .face import Face


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

    def add_people(self, imgObject, peopleObject):
        """
        数据库中添加一个包含头像的图片

        Args:
            imgList: list of imgObject
        """
        rgb = imgObject.getRGB()
        if rgb is None:
            raise Exception("unable to load image")

        alignedFaceRgb = self.alignFilter.align(
            self.face_size, rgb, landmarkIndices=self.landmarkIndices,
            skipMulti=self.skip_multi)

        if alignedFaceRgb is None:
            raise Exception('unable to align')

        rep = self.face_network.forward(alignedFaceRgb)
        phash = str(imagehash.phash(Image.fromarray(alignedFaceRgb)))
        face = Face(rep, alignedFaceRgb, peopleObject)
        self.faces[phash] = face

    def trainSVM(self):
        """
        根据已有的 self.faces 内容进行 training
        """
        logger.info("Training SVM on {} labled images.".format(len(self.faces)))
        d = self.getData()
        if d is None:
            self.svm = None
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

    def predict(self, imgObject, multi=True):
        """
        Args:
            imgObject:

        Returns:
            [[bb, people, confidence], ...]
        """
        rgbImg = imgObject.getRgb()
        reps = self.getReps(rgbImg, multi)
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
        pass

    def getReps(self, rgbImg, multi):
        """
        寻找脸后，提取特征

        Returns:
            [bb, rep]
        """
        start = time.time()
        if multi:
            bbs = self.alignFilter.getAllFaceBoundingBoxes(rgbImg)
        else:
            bbs = [self.alignFilter.getLargestFaceBoundingBox(rgbImg)]

        if len(bbs) == 0:
            logger.info("unable to find a face")

        logger.debug("face detector took {} seconds.".format(time.time() - start))

        reps = []
        for bb in bbs:
            start = time.time()
            alignedFace = self.alignFilter.align(
                self.face_size, rgbImg, bb,
                landmarkIndices=self.landmarkIndices
            )
            logger.debug('alignment took {} seconds.'.format(time.time() - start))
            if alignedFace is None:
                logger.info("unable to align face at {} {} {} {}".foramt(
                    bb.left(), bb.bottom(), bb.right(), bb.top()))
                continue

            start = time.time()
            rep = self.face_network.forward(alignedFace)
            logger.debug("Neural network forward pass took {} seconds.".format(time.time() - start))
            reps.append((bb, rep))
        sreps = sorted(reps, key=lambda x: x[0].center().x)
        return sreps
