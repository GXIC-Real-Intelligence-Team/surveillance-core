from __future__ import absolute_import
from . import image
from . import faceapi
import time

class Face:
    """ Face class """
    def __init__(self, eigen, people=None):
        self.eigen = eigen
        self.people = people

class Detector:
    """ Detector class """
    def __init__(self, svm, scale):
        self.svm = svm
        self.scale = scale

    def detect(self, frame):
        """
        Detect faces of a opencv frame, and return frame
        with boxes and peoples recognized
        """
        start = time.time()
        print('start at:', start)
        frame, small_frame = image.format(frame, self.scale)
       
        bbs = faceapi.allFaceBoundingBoxes(small_frame)
        peoples = []

        if len(bbs) == 0:
            return frame, peoples

        for bb in bbs:
            landmarks = faceapi.findLandmarks(small_frame, bb)
            face = faceapi.align(small_frame, bb, landmarks=landmarks)

            if face is None:
                continue

            eigen = faceapi.getEigen(face)
            people = None #self.svm.predict(eigen)[0]

            image.printFaceBox(frame, self.scale, bb)

            if people:
                name = people.name
                peoples.append()
            else:
                name = 'Unknown'

            image.printName(frame, name, self.scale, bb)

        end = time.time()
        print('end at:', end)
        print('cost:', end - start)

        return frame, peoples
