# -*- coding: utf-8 -*-
import base64

import cv2
# import imagehash
import numpy as np
from matplotlib import cm
# from PIL import Image

# def phash(rgbImg):
#    return str(imagehash(Image.fromarray(rgbImg)))


def resize(frame, scale):
    """
    放大缩小图片

    args:
        frame: opencv 格式图像
        scale: 放大缩小倍数
    """
    # frame = flip(frame)
    height, width = getSize(frame)
    frameSmall = cv2.resize(frame, (int(width * scale),
                                    int(height * scale)))
    return frame, frameSmall


def printFaceBox(frame, scale, bb):
    """
    打印脸上的方框

    args:
        frame: opencv 格式图像
        scale: 同 resize:scale
        bb: 头像定位对象, 通过 faceapi.allFaceBoundingBoxes 可以得到
    """
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
    """
    在脸旁打名字

    args:
        frame: opencv 格式图像
        name: 打印的名字
        scale: 同 resize:scale
        bb: 头像定位对象, 通过 faceapi.allFaceBoundingBoxes 可以得到
    """
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


def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    np_arr = np.fromstring(encoded_data.decode('base64'), np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def cv2_img_to_data_uri(img):
    cnt = cv2.imencode('.jpg', img)[1]
    b64 = base64.encodestring(cnt)
    return "data:image/jpeg;base64,{}".format(b64)
