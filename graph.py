from cv2 import cv2
import numpy as np
import copy
import extract_graph
import get_graph_dimentions


def pixel_val(p):
    return sum(p)


img = cv2.imread('sample_graph.jpg')
graph = extract_graph.extract(img)
# cv2.imshow('graph', graph)
# cv2.waitKey(0)

x_domain, y_domain, left, right, top, bottom = get_graph_dimentions.get_dim(img)

# cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))
# cv2.imshow('img', img)
# cv2.waitKey(0)

readings = {}
xy = {}

for time in range(x_domain):
    bpm = 0
    x = int((time / x_domain) * (right - left))
    col = list(graph[:, x])
    pixel = min(col, key=pixel_val)
    for y in range(len(col)):
        if col[y][0] == pixel[0] and col[y][1] == pixel[1] and col[y][2] == pixel[2]:
            bpm = 50 + ((bottom - y) / (bottom - top)) * y_domain
            readings[time] = bpm
            xy[x] = y

for x, y in xy.items():
    cv2.circle(graph, (x, y), 2, (0, 255, 0))

cv2.imshow('graph', graph)
cv2.waitKey(0)
cv2.destroyAllWindows()

time = int(input("Enter time: "))
print(readings[time], 'bpm')