import cv2
import numpy as np

img = cv2.imread('sample_graph.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
cv2.imshow('edges', edges)
cv2.waitKey(0)

h, w, c = img.shape

lines = cv2.HoughLines(edges,1,np.pi/180,500)

for rho,theta in lines[:,0]:
    print(rho, theta)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + w*(-b))
    y1 = int(y0 + h*(a))
    x2 = int(x0 - w*(-b))
    y2 = int(y0 - h*(a))

    # print(rho, theta)
    # print(x1, x2, y1, y2)

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imshow('fin', img)
cv2.waitKey(0)