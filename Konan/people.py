import numpy as np

""" People class"""
class People:
    def __init__(self, pid, name, eigen, face_img):
        self.pid = pid
        self.name = name
        self.img = face_img
        self.eigen = eigen

""" compare two people eigen """
def compare(p1, p2):
    distance = p1.eigen - p2.eigen
    return np.dot(distance, distance)

""" people factory """
def create(pid, name, eigen, face_img):
    return People(pid, name, eigen, face_img)