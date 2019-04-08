from cv2 import cv2
import numpy as np


def get_dim(img):
    l, r, t, b = 10000, 0, 10000, 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 100)
    # cv2.imshow('edges', edges)
    # cv2.waitKey(0)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, minLineLength=50)
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(x1 - x2) < 20:
                x = int((x1 + x2) / 2)
                l, r = min(l, x), max(r, x)
            if abs(y1 - y2) < 20:
                y = int((y1 + y2) / 2)
                t, b = min(t, y), max(b, y)

    # cv2.line(img, (l, t), (l, b), (0, 255, 0), 2)
    # cv2.line(img, (r, t), (r, b), (0, 255, 0), 2)
    # cv2.line(img, (l, t), (r, t), (0, 255, 0), 2)
    # cv2.line(img, (l, b), (r, b), (0, 255, 0), 2)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    return 60 * 25, 160, l, r, t, b
