from cv2 import cv2
import numpy as np
import copy
import extract_graph
import get_graph_dimentions

img = cv2.imread('sample_graph.jpg')

x_domain, y_domain, left, right, top, bottom = get_graph_dimentions.get_dim(img)

graph = extract_graph.extract(img, left, right)
# cv2.imshow('graph', graph)
# cv2.waitKey(0)

# cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))
# cv2.imshow('img', img)
# cv2.waitKey(0)

readings = {}
xy = {}

gray = cv2.cvtColor(graph, cv2.COLOR_BGR2GRAY)

for time in range(x_domain):
    bpm = 0
    x = int((time / x_domain) * (right - left))
    col = list(gray[:, x])
    pixel = min(col)
    for y in range(len(col)):
        if col[y] == pixel:
            bpm = 50 + ((bottom - y) / (bottom - top)) * y_domain
            readings[time] = bpm
            xy[x] = y

# Display points
for x, y in xy.items():
    cv2.circle(graph, (x, y), 2, (0, 255, 0))

cv2.imshow('graph', graph)
cv2.waitKey(0)
cv2.destroyAllWindows()

time = int(input("Enter time: "))
print(readings[time], 'bpm')
