from cv2 import cv2
import numpy as np
import copy
import extract_graph

img = cv2.imread('sample_graph.jpg')
graph = extract_graph.extract(img)
cv2.imshow('graph', graph)
cv2.waitKey(0)