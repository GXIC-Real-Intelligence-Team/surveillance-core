# -*- coding: utf-8 -*-
import logging
import os

import time
import openface


logger = logging.getLogger(__name__)

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


class DlibDetector(object):

    def __init__(self, landmarks='outerEyesAndNose',
                 dlib_predictor_db="shape_predictor_68_face_landmarks.dat",
                 face_network_model="nn4.small2.v1.t7",
                 face_size=96, skip_multi=False, cuda=False):

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
        self.face_network = openface.TorchNeuralNet(
            face_network_model_path, imgDim=self.face_size, cuda=cuda)

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
