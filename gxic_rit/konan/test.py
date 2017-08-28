import os
import cv2
from gxic_rit.konan import Detector 


TEST_IMAGE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'example.jpg'
)


if __name__ == '__main__':
    detector = Detector(None, 0.2)
    image = cv2.imread(TEST_IMAGE)
    frame, peoples = detector.detect(image)
    cv2.imwrite('example.result.jpg', frame)