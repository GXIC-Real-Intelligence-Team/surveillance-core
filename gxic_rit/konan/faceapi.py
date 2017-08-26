import os
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

ALIGN = AlignDlib(os.path.join(DLIB_MODEL_DIR, DLIB_MODEL_DIR))
NET = TorchNeuralNet(os.path.join(OPENFACE_MODEL_DIR, NETWORK_MODEL))

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

    return ALIGN.align(IMG_DIM, rgbImg
        , bb=bb
        , landmarks=landmarks
        , landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
    
def allFaceBoundingBoxes(rgbImg):
    """
    Find all face bounding boxes in an image.
    """
    assert rgbImg is not None

    try:
        return ALIGN.getAllFaceBoundingBoxes(rgbImg)
    except Exception as e:
        print("Warning: {}".format(e))
        # In rare cases, exceptions are thrown.
        return []

def getEigen(face):
    return NET.forward(face)