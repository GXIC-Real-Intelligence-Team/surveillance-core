import cv2
# import imagehash
import numpy as np
from matplotlib import cm
# from PIL import Image

# def phash(rgbImg):
#    return str(imagehash(Image.fromarray(rgbImg)))

def format(frame, scale):
    # frame = flip(frame)
    height, width = getSize(frame)
    frameSmall = cv2.resize(frame, (int(width * scale),
                                    int(height * scale)))
    return frame, frameSmall
                                
def printFaceBox(frame, scale, bb):
    center = bb.center()
    height, width = getSize(frame)
    centerI = 0.7 * center.x * center.y / \
        (scale * scale * width * height)
    color_np = cm.Set1(centerI)
    color_cv = list(np.multiply(color_np[:3], 255))

    bl = (int(bb.left() / scale), int(bb.bottom() / scale))
    tr = (int(bb.right() / scale), int(bb.top() / scale))
    cv2.rectangle(frame, bl, tr, color=color_cv, thickness=3)

def printName(frame, name, scale, bb):
    left = int(bb.left() / scale)
    top = int(bb.top() / scale)

    cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                        color=(152, 255, 204), thickness=2)

def flip(frame):
    return cv2.flip(frame, 1)

def getSize(frame):
    return frame.shape[0], frame.shape[1]

def gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)