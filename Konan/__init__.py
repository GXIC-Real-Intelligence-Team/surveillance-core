import image
import faceapi

class Face:
    """ Face class """
    def __init__(self, eigen, people=None):
        self.eigen = eigen
        self.people = people

class Detector:
    """ Detector class """
    def __init__(self, showUnknown=False, svm, width, heigth, scale):
        self.svm = svm
        self.showUnknown = showUnknown
        
    def detect(self, frame):
        """ 
        Detect faces of a opencv frame, and return frame 
        with boxes and peoples recognized
        """ 

        frame, smallFrame = image.format(frame, self.width, self.height, self.scale)

        bbs = faceapi.allFaceBoundingBoxes(smallFrame)

        peoples = []

        if len(bbs) == 0:
            return frame, peoples

        for bb in bbs:
            landmarks = faceapi.findLandmarks(smallFrame, bb)
            alignedFace = faceapi.align(smallFrame, bb, landmarks=landmarks)

            if alignedFace is None:
                continue

            eigen = faceapi.geteigen(alignedFace)
            people = self.svm.predict(eigen)[0] 
            
            image.printFaceBox(frame, self.width, self.height, self.scale, bb)

            if people:
                name = people.name
            else:
                name = 'Unknown'
            
            image.printName(frame, name, self.scale, bb)

        
