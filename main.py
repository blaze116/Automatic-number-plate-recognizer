import cv2
import glob
import numpy as np
import os
import sys

from obj_det import get_lps_yolo # LP(License Plate) detection with YOLO
from ptransform import get_LP # Perspective transformation of LP
from wpod_net import * # LP detection with WPOD-NET
from seg_rec import * # Character segmentation and recognition using detectron


# This function takes image of LP as input and performs character segmentation and recognition.
def PlateRecognition(image):

    ret, img = get_LP(image)

    if not ret:
        # print(f"DIDN'T FIND ANY LICENSE PLATE IN FILE -> {filename} WHILE LICENSE PLATE PERSPECTIVE TRANSFORMING")
        img = image
        
    if img.shape[1] < 3*img.shape[0]:
        img = cv2.resize(img, (img.shape[0]*3, img.shape[0]))
        # print("RESIZING LP DUE TO ASPECT RATIO ISSUE")

    try:
        answer = get_chars(img)
        print(answer)
        cv2.imwrite("HELLO.jpg", img)
    except IndexError:
        pass

# This function takes image which contains cars as input and performs LP detection, 
# character segmentation and recognition.
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
        # print(f"DIDN'T FIND LICENSE PLATE IN {filename}")
        return

    for el in images:
        PlateRecognition(el)

# This function takes a video which contains cars as input and performs LP detection, 
# character segmentation and recognition on each frame.
def from_video(filename):

    cap = cv2.VideoCapture(filename)
    fps_count = 0

    while(cap.isOpened()):

        ret, image = cap.read()
        fps_count += 1

        if fps_count % 15 == 0:

            wpodnet_imgs = []
            yolo_imgs = []
            try:
                print(image.shape)
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

    image_path = glob("cars/*")
    FROM_CAR = False

    for filename in image_path:
        
        if FROM_CAR:
            from_car(filename)
        else:
            image = cv2.imread(filename)
            PlateRecognition(image)