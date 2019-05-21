import cv2 as cv2
import numpy as np
import glob, os


for file in glob.glob("graphs/*_*.jpg"):
    print(file)
    # img = cv2.imread('graphs/img1_1.jpg')
    img = cv2.imread(file, cv2.COLORMAP_HSV)
    cv2.imshow('img', img)
    cv2.waitKey(0)

    w, h, _ = img.shape

    blur = cv2.GaussianBlur(img, (5, 5), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 0])
    upper = np.array([255, 50, 150])

    mask = cv2.inRange(hsv, lower, upper)
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)

    # erode_size, dilate_size = 3, 5
    #
    # erode_kernel = np.ones((erode_size, 1), np.uint8)
    # erode_h = cv2.erode(mask, erode_kernel)
    # # cv2.imshow("erode", erode_h)
    # # cv2.waitKey(0)
    #
    # dilate_kernel = np.ones((dilate_size, 1), np.uint8)
    # dilate_h = cv2.dilate(erode_h, dilate_kernel)
    # # cv2.imshow("dilate", dilate_h)
    # # cv2.waitKey(0)
    #
    # erode_kernel = np.ones((1, erode_size), np.uint8)
    # erode_v = cv2.erode(dilate_h, erode_kernel)
    # # cv2.imshow("erode", erode_v)
    # # cv2.waitKey(0)
    #
    # dilate_kernel = np.ones((1, dilate_size), np.uint8)
    # dilate_v = cv2.dilate(erode_v, dilate_kernel)
    # # cv2.imshow("dilate", dilate_v)
    # # cv2.waitKey(0)

    kernel_size = 2
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opn = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('opn', opn)
    # cv2.waitKey(0)

    bg = np.ones((w, h, 3), np.uint8)
    bg = cv2.bitwise_not(bg, mask=~opn)
    # cv2.imshow('bg', bg)
    # cv2.waitKey(0)

    res = cv2.bitwise_or(img, img, mask=opn)
    res = cv2.bitwise_or(res, bg)

    cv2.imshow("res", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
