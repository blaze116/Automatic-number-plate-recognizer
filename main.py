import cv2
from obj_det import get_lps_yolo # YOLO one
from ptransform import get_LP # For tranforming LP
import glob
import cv2
import numpy as np
import os
from wpod_net import *
from seg_rec import *
import sys


def draw_box(image, x, y, w, h):
    return cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
    

def show_img(image, win_name="IMG"):
    cv2.imshow(win_name, image)
    cv2.waitKey(1000)

def PlateRecognition(image):


    ret, img = get_LP(image)

    if not ret:
        print(f"DIDN'T FIND ANY LICENSE PLATE IN FILE -> {filename} WHILE LICENSE PLATE PERSPECTIVE TRANSFORMING")
        img = image
        
    if img.shape[1] < 3*img.shape[0]:
        img = cv2.resize(img, (img.shape[0]*3, img.shape[0]))
        print("RESIZING LP DUE TO ASPECT RATIO ISSUE")

    try:
        answer = get_chars(img)
        print(f"{filename} : {answer}")
        cv2.imwrite("HELLO.jpg", img)
    except IndexError:
        pass

def from_car(filename):

    image = cv2.imread(filename)

    wpodnet_imgs = []
    yolo_imgs = []
    try:
        wpodnet_imgs = get_plate_from_car(image)
    except AssertionError:
        pass
    yolo_imgs = get_lps_yolo(image)
    
    images = yolo_imgs
    if len(wpodnet_imgs) >= len(yolo_imgs):
        images = wpodnet_imgs 
    if len(images) == 0:
        print(f"DIDN'T FIND LICENSE PLATE IN {filename}")
        return

    for el in images:
        PlateRecognition(el)

def from_video(filename):

    cap = cv2.VideoCapture(filename)

    while(cap.isOpened()):
        ret, image = cap.read()

        wpodnet_imgs = []
        yolo_imgs = []
        try:
            wpodnet_imgs = get_plate_from_car(image)
        except AssertionError:
            pass
        yolo_imgs = get_lps_yolo(image)
        
        images = yolo_imgs
        if len(wpodnet_imgs) >= len(yolo_imgs):
            images = wpodnet_imgs 
        if len(images) == 0:
            continue

        for el in images:
            PlateRecognition(el)

from glob import glob

if __name__ == "__main__":

    image_path = glob("*webp")

    for filename in image_path:
        
        from_car(filename)
        # image = cv2.imread(filename)
        # PlateRecognition(image)