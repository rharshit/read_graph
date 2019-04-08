from cv2 import cv2
import numpy as np
import copy


def extract(img, left, right):
    dim_y = len(img)
    dim_x = len(img[0])
    print(dim_x, dim_y)
    # cv2.imshow('', img)
    # cv2.waitKey(0)

    final_mask = np.zeros((dim_y, dim_x, 3), np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    # cv2.waitKey(0)

    _, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('thresh', thresh)
    # cv2.waitKey(0)

    l, r = 0, 0
    t = 0
    max_cnt = []
    print('right left', right, left)
    print('rl', r, l)
    while r - l < right - left:
        print('rl', r, l)
        t += 1
        cnt = copy.deepcopy(thresh)
        # cv2.imshow('cnt', cnt)
        # cv2.waitKey(0)

        contours, hierarchy = cv2.findContours(cnt, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(cnt, contours, -1, 255, t)
        # cv2.imshow('cnt', cnt)
        # cv2.waitKey(0)

        contours, hierarchy = cv2.findContours(cnt, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        max_cnt = max(contours, key=cv2.contourArea)
        l = min(max_cnt[:, :, 0])
        r = max(max_cnt[:, :, 0])

    cv2.drawContours(final_mask, [max_cnt], 0, (255, 255, 255), -1)
    # cv2.imshow('final_mask', final_mask)
    # cv2.waitKey(0)

    bit_and = cv2.bitwise_and(img, final_mask)
    bit_not = cv2.bitwise_not(final_mask)
    final = cv2.bitwise_or(bit_and, bit_not)
    # cv2.imshow('final', final)
    # cv2.waitKey(0)

    contrast = cv2.addWeighted(final, 1.5, final, 0, 0)
    # cv2.imshow('contrast', contrast)
    # cv2.waitKey(0)

    return contrast
