import cv2 as cv2
import numpy as np
import glob, os
import copy
from statistics import mean
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
    print("w:", w, "h:", h)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', cv2.pyrDown(gray))
    # cv2.waitKey(0)

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

    bilateral = cv2.bilateralFilter(dilate, 10, 10, 10)
    # cv2.imshow('bilateral', cv2.pyrDown(bilateral))
    # cv2.waitKey(0)

    norm_num_blocks = 3
    norm_kernel_w = w//norm_num_blocks
    norm_kernel_h = h//norm_num_blocks
    norm = np.zeros((h, w), np.uint8)

    for rows in range(0, w, norm_kernel_w//2):
        for cols in range(0, h, norm_kernel_h//2):
            tmp = copy.deepcopy(bilateral)
            cv2.rectangle(tmp, (rows, cols), (rows+norm_kernel_w, cols+norm_kernel_h), (0, 255, 0), 2)
            roi = bilateral[cols:cols+norm_kernel_h, rows:rows+norm_kernel_w]
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

    line_output = np.ones((h, w), np.uint8) * 255
    img_tmp = copy.deepcopy(img)
    for x1, y1, x2, y2 in linesP[:, 0]:
        if abs(x1 - x2) > abs(y1 - y1):
            cv2.line(img_tmp, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.line(line_output, (x1, y1), (x2, y2), 0, 2)

    # cv2.imshow('lines', cv2.pyrDown(img_tmp))
    # cv2.imshow('line_output', cv2.pyrDown(line_output))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    lines_final = cv2.HoughLinesP(~line_output, 1, np.pi /720, w//20, None, w//20, w//10)

    slopes = []
    line_extend_length = w//20

    for x1, y1, x2, y2 in lines_final[:, 0]:
        if abs(x1 - x2) > abs(y1 - y1):
            slope = (y1-y2)/(x1-x2)
            slopes.append(slope)
            m = slope
            # print('m', m)

            X1 = (x1 if x1 > x2 else x2) + line_extend_length * np.cos(np.arctan(m))
            Y1 = (y1 if x1 > x2 else y2) + line_extend_length * np.sin(np.arctan(m))
            X2 = (x2 if x1 > x2 else x1) - line_extend_length * np.cos(np.arctan(m))
            Y2 = (y2 if x1 > x2 else y1) - line_extend_length * np.sin(np.arctan(m))

            # print(x1, X1)
            # print(x2, X2)
            # print(y1, Y1)
            # print(y2, Y2)

            cv2.line(img_tmp, (int(X1), int(Y1)), (int(X2), int(Y2)), (0, 127, 0), 1)
            cv2.line(line_output, (int(X1), int(Y1)), (int(X2), int(Y2)), 63, 1)

            cv2.line(img_tmp, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.line(line_output, (x1, y1), (x2, y2), 0, 3)

    rotation = mean(slopes)
    # print('rotation', rotation)
    angle = np.arctan(rotation) * 180 / np.pi
    print('angle', angle)
    # cv2.imshow('lines', cv2.pyrDown(img_tmp))
    # cv2.imshow('line_output', cv2.pyrDown(line_output))
    # cv2.waitKey(0)

    M = cv2.getRotationMatrix2D((h/2, w/2), angle, 1)
    # print(M)
    img_fixed = ~cv2.warpAffine(~img_tmp, M, (w, h))
    lines_fixed = ~cv2.warpAffine(~line_output, M, (w, h))

    cv2.imshow(file+'img_fixed', cv2.pyrDown(img_fixed))
    cv2.imshow('lines_fixed', lines_fixed)
    cv2.waitKey(0)

    cls = cv2.morphologyEx(lines_fixed, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=3)
    cv2.imshow('cls', cls)
    cv2.waitKey(0)

    # lines = cv2.HoughLinesP(line_output, 1, np.pi / 1440, w//5, minLineLength=w//5, maxLineGap=50)
    # interpolate = np.ones((h, w), np.uint8)*255
    # for x1, y1, x2, y2 in lines[:, 0]:
    #     cv2.line(interpolate, (x1, y1), (x2, y2), 254, 1)
    #
    # cv2.imshow('interpolate', cv2.pyrDown(interpolate))
    # cv2.waitKey(0)

    # edges = cv2.Canny(line_output, 50, 150, apertureSize=3)
    # cv2.imshow('edges', cv2.pyrDown(edges))
    # cv2.waitKey(0)
    #
    # line_output_hough = np.ones((h, w), np.uint8) * 255
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 250)
    # for rho, theta in lines[:, 0]:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + w * (-b))
    #     y1 = int(y0 + h * (a))
    #     x2 = int(x0 - w * (-b))
    #     y2 = int(y0 - h * (a))
    #
    #     cv2.line(line_output_hough, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #
    # cv2.imshow('line_output_hough', cv2.pyrDown(line_output_hough))
    # cv2.waitKey(0)

    cv2.destroyAllWindows()
