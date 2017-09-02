# -*- coding: utf-8 -*-
import numpy as np


class People:
    """ People class"""
    def __init__(self, pid, name, eigen, face_img):
        self.pid = pid
        self.name = name
        self.img = face_img
        self.eigen = eigen


def compare(p1, p2):
    """ compare two people eigen """
    distance = p1.eigen - p2.eigen
    return np.dot(distance, distance)


def create(pid, name, eigen, face_img):
    """ people factory """
    return People(pid, name, eigen, face_img)
