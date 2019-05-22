import cv2 as cv2
import numpy as np
import glob, os
import copy
from skimage.exposure import rescale_intensity

kernel_size = 21
assert kernel_size % 2 == 1

flist = glob.glob("graphs/*_*.jpg")
flist.sort()

for file in flist:
    print(file)
    type = 0
    if 'fhr' in file:
        type = 1
    elif 'uc' in file:
        type = 2
    if type == 0:
        continue
    print('processing', ('FHR' if type == 1 else 'UC'))

    img = cv2.imread(file)
    h, w, _ = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', cv2.pyrDown(gray))
    cv2.waitKey(0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(h // 15, w // 15))
    cl_gray = clahe.apply(gray)
    # cv2.imshow('clahe_gray', cv2.pyrDown(cl_gray))
    # cv2.waitKey(0)

    kernel = np.zeros((kernel_size, kernel_size), np.uint8)
    kernel[kernel_size // 2] = np.ones(kernel_size, np.uint8)

    dilate = cv2.dilate(cl_gray, kernel, iterations=1)
    # cv2.imshow('dilate', cv2.pyrDown(dilate))
    # cv2.waitKey(0)

    bg_kernel = np.ones((15, 15), np.uint8)
    bg = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, bg_kernel)
    # cv2.imshow('bg', cv2.pyrDown(bg))
    # cv2.waitKey(0)

    dilate += ~bg
    # cv2.imshow('dilate_clean', cv2.pyrDown(dilate))
    # cv2.waitKey(0)

    norm_num_blocks = 4
    norm_kernel_w = w//norm_num_blocks
    norm_kernel_h = h//norm_num_blocks
    norm = np.zeros((h, w), np.uint8)

    for rows in range(0, w, norm_kernel_w):
        for cols in range(0, h, norm_kernel_h):
            tmp = copy.deepcopy(dilate)
            cv2.rectangle(tmp, (rows, cols), (rows+norm_kernel_w, cols+norm_kernel_h), (0, 255, 0), 2)
            roi = dilate[cols:cols+norm_kernel_h, rows:rows+norm_kernel_w]
            norm[cols:cols+norm_kernel_h, rows:rows+norm_kernel_w] = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX)
            # cv2.imshow('dil', dilate[cols:cols+norm_kernel_h, rows:rows+norm_kernel_w])
            # cv2.imshow('norm', norm[cols:cols+norm_kernel_h, rows:rows+norm_kernel_w])
            # cv2.imshow('roi', cv2.pyrDown(tmp))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    # norm = cv2.normalize(dilate, None, 0, 255, cv2.NORM_MINMAX)
    # cv2.imshow('norm', cv2.pyrDown(norm))
    # cv2.waitKey(0)

    thresh = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 30)
    # cv2.imshow('thresh', cv2.pyrDown(thresh))
    # cv2.waitKey(0)

    erode = cv2.erode(thresh, np.ones((3, 3), np.uint8))
    # cv2.imshow('erode', cv2.pyrDown(erode))
    # cv2.waitKey(0)

    linesP = cv2.HoughLinesP(~erode, 1, np.pi / 2, 10, None, w // 20, 30)

    output = np.ones((h, w), np.uint8) * 255
    for x1, y1, x2, y2 in linesP[:, 0]:
        if abs(x1 - x2) > abs(y1 - y1):
            cv2.line(output, (x1, y1), (x2, y2), 0, 2)

    cv2.imshow('lines', cv2.pyrDown(output))
    cv2.waitKey(0)

    # lines = cv2.HoughLinesP(output, 1, np.pi / 360, w//2, minLineLength=w//2, maxLineGap=10)
    # interpolate = np.ones((h, w), np.uint8)*255
    # for x1, y1, x2, y2 in lines[:, 0]:
    #     cv2.line(interpolate, (x1, y1), (x2, y2), 0, 1)
    #
    # cv2.imshow('interpolate', cv2.pyrDown(interpolate))
    # cv2.waitKey(0)

    cv2.destroyAllWindows()
