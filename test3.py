import cv2 as cv2
import numpy as np
import glob, os
import copy

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

    # img = cv2.imread('graphs/img1_1.jpg')
    img = cv2.imread(file, cv2.COLORMAP_HSV)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    w, h, _ = img.shape

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # cv2.imshow('blur', blur)
    # cv2.imshow('blur/2', cv2.divide(blur, (2, 2, 2, 2)))
    # cv2.waitKey(0)

    blur_chan = np.ones((w, h), np.uint8)

    # for row in range(w):
    #     for col in range(h):
    #         blur_chan[row, col] = [max(blur_chan[row, col]) for _ in blur_chan[row, col]]

    blur_red = blur[:, :, 0].flat[:]
    blur_blue = blur[:, :, 1].flat[:]
    blur_green = blur[:, :, 2].flat[:]

    for p, r, g, b in np.nditer([blur_chan.flat, blur_red, blur_green, blur_blue], op_flags=['readwrite']):
        p[...] = max(r, g, b)

    blur_chan = blur_chan.reshape(w, h, 1)
    blur_chan = cv2.cvtColor(blur_chan, cv2.COLOR_GRAY2BGR)
    # cv2.imshow('blur_chan', blur_chan)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print('blur', blur.shape)
    # print('blch', blur_chan.shape)

    # blur_normalized = cv2.add(cv2.divide(blur, (2, 2, 2, 2)), cv2.divide(blur_chan, (1, 1, 1, 1)))
    blur_normalized = cv2.addWeighted(blur, 0.2, blur_chan, 0.8, 0)
    # blur_normalized/=255
    # cv2.imshow('blur_normalized', blur_normalized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)

    thresh = 140

    lower = np.array([0, 0, 0])
    upper = np.array([255, thresh, 50])

    lower_rgb = np.array([0, 0, 0])
    upper_rgb = np.array([thresh, thresh, thresh])

    mask1 = cv2.inRange(hls, lower, upper)
    mask2 = cv2.inRange(blur_chan, lower_rgb, upper_rgb)
    # cv2.imshow("mask1", mask1)
    # cv2.waitKey(0)
    # cv2.imshow("mask2", mask2)
    # cv2.waitKey(0)

    mask = cv2.bitwise_and(mask1, mask2)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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

    kernel_size = 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opn = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('opn', opn)
    # cv2.waitKey(0)

    bg = np.ones((w, h, 3), np.uint8)
    bg = cv2.bitwise_not(bg, mask=~mask)
    # cv2.imshow('bg', bg)
    # cv2.waitKey(0)

    res = cv2.bitwise_or(img, img, mask=mask)
    res = cv2.bitwise_or(res, bg)

    # cv2.imshow("res", res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
