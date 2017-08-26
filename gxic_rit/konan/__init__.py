from __future__ import absolute_import
from . import image
from . import faceapi

class Face:
    """ Face class """
    def __init__(self, eigen, people=None):
        self.eigen = eigen
        self.people = people

class Detector:
    """ Detector class """
    def __init__(self, svm, width, heigth, scale):
        self.svm = svm
        # display width and height
        self.width = width
        self.height = heigth
        self.scale = scale

    def detect(self, frame):
        """
        Detect faces of a opencv frame, and return frame
        with boxes and peoples recognized
        """

        frame, small_frame = image.format(frame, self.width, self.height, self.scale)

        bbs = faceapi.allFaceBoundingBoxes(small_frame)

        peoples = []

        if len(bbs) == 0:
            return frame, peoples

        for bb in bbs:
            landmarks = faceapi.findLandmarks(small_frame, bb)
            face = faceapi.align(small_frame, bb, landmarks=landmarks)

            if face is None:
                continue

            eigen = faceapi.geteigen(face)
            people = self.svm.predict(eigen)[0]

            image.printFaceBox(frame, self.width, self.height, self.scale, bb)

            if people:
                name = people.name
            else:
                name = 'Unknown'

            image.printName(frame, name, self.scale, bb)

        return frame
