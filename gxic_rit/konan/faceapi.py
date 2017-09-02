import logging
import os
import time

import openface
from openface import AlignDlib, TorchNeuralNet


# model directory
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..',
    'models'
)

DLIB_MODEL_DIR = os.path.join(MODELS_DIR, 'dlib')
OPENFACE_MODEL_DIR = os.path.join(MODELS_DIR, 'openface')

DLIB_PREDICTOR = 'shape_predictor_68_face_landmarks.dat'
NETWORK_MODEL = 'nn4.small2.v1.t7'

# face img is 96x96
IMG_DIM = 96

ALIGN = AlignDlib(os.path.join(DLIB_MODEL_DIR, DLIB_PREDICTOR))
NET = TorchNeuralNet(os.path.join(OPENFACE_MODEL_DIR, NETWORK_MODEL))

logger = logging.getLogger(__name__)

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


def findLandmarks(rgbImg, bb):
    """ find landmarks
    Find the landmarks of a face.
    """
    return ALIGN.findLandmarks(rgbImg, bb)


def align(rgbImg, bb, landmarks=None):
    """
    Transform and align a face in an image.
    """
    assert rgbImg is not None
    assert bb is not None

    return ALIGN.align(IMG_DIM, rgbImg, bb=bb, landmarks=landmarks, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


def allFaceBoundingBoxes(img):
    """
    Find all face bounding boxes in an image.

    returns:
        bb
    """
    assert img is not None

    try:
        return ALIGN.getAllFaceBoundingBoxes(img)
    except Exception as e:
        print("Warning: {}".format(e))
        # In rare cases, exceptions are thrown.
        return []


def getEigen(face):
    return NET.forward(face)


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

    def getEigen(self, rgbImg, multi):
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
