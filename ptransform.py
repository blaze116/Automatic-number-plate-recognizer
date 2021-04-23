import cv2
import imutils
import numpy as np
from skimage.filters import threshold_local
import argparse

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
# obtain a consistent order of the points and unpack them
# individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def persp_transform(image, mat):
    ratio = 1
    orig = image.copy()
    # mat=[[x,y],[x+w,y],[x+w,y+h],[x,y+h]]
    mat = np.array(mat)
    warped = four_point_transform(orig, mat * ratio)
    # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("T", warped)
    # cv2.waitKey(0)
    return warped

# This function takes image which may contain a License plate tilted or warped as an input 
# and performs perspective transformation on the input image.
def get_LP(img):

    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray1, 13, 15, 15)

    # Thresholding
    kernel = np.ones((3, 3), np.uint8)
    nb1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 103, 7)
    nb1 = cv2.dilate(nb1, kernel)
    _, nb2 = cv2.threshold(gray, (np.max(nb1)-np.min(nb1))//2, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    nb = cv2.bitwise_and(nb1, nb2)

    # masking out LP
    contours = cv2.findContours(nb.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours,key=cv2.contourArea, reverse = True)[:10]

    approx = cv2.convexHull(contours[0])
    mask = np.zeros(nb.shape,np.uint8)
    new_image = cv2.drawContours(mask, [approx], 0,255,-1,)
    masked_image = cv2.bitwise_and(nb,nb,mask=mask)

    # Converting quadrilateral into rectangle
    contours=cv2.findContours(masked_image.copy(),cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours,key=cv2.contourArea, reverse = True)[:10]
    screenCnt = None

    for el in np.arange(0.012, 0.50, 0.002):
        peri = cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], el * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            # print("GOT APPROX = 4")
            break
    else:
        # print("DIDN'T GET")
        return False, None

    # print(screenCnt)

    # Perspective transformation
    return True, persp_transform(img, [screenCnt[0][0], screenCnt[1][0], screenCnt[2][0], screenCnt[3][0]])

