import cv2 as cv2
import numpy as np
import copy
from statistics import mean

fname = 'graphs/img9_1.jpg'

num_blocks_h = 10
num_blocks_v = 6

img = cv2.imread(fname)
h, w, _ = img.shape
print("w:", w, "h:", h)

blur = cv2.GaussianBlur(img, (3,3), 0)
cv2.imshow('blur', blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

block_size_w = w//num_blocks_h
block_size_h = h//num_blocks_v
print("block size", block_size_h, block_size_w)

tmp = copy.deepcopy(img)
grid = np.ones((h, w), np.uint8) * 255
for row in range(0, w, block_size_w//2):
    for col in range(0, h, block_size_h//2):
        block = blur[col:col+block_size_h, row:row+block_size_w]
        norm = cv2.normalize(block, None, 0, 255, cv2.NORM_MINMAX)
        # cv2.rectangle(tmp, (row, col), (row+block_size_w, col+block_size_h), (255, 0, 255), 2)
        # cv2.imshow("block:"+str(row)+","+str(col), block)
        # cv2.waitKey(0)
        block_gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
        # block_hls = cv2.cvtColor(block, cv2.COLOR_BGR2HLS)
        # lower_hls = np.array([0, 75, 0])
        # upper_hls = np.array([255, 255, 255])
        # mask = cv2.inRange(block_hls, lower_hls, upper_hls)
        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)

        block_edges = cv2.Canny(norm, 25, 150)
        # cv2.imshow('edges', block_edges)
        # cv2.waitKey(0)

        block_close = cv2.morphologyEx(block_edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        # cv2.imshow('close', block_close)
        # cv2.waitKey(0)

        lines = cv2.HoughLinesP(block_close, 1, np.pi/360, block_size_h//2, None, block_size_h//2, block_size_h//2)
        # print(len(lines[:,0]))
        # print()
        angles_h = []
        angles_v = []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                angle = np.arctan((y2-y1)/(x2-x1)) * 180 / np.pi
                if abs(angle) >=80:
                    angles_v.append(abs(angle))
                if abs(angle) <=10:
                    angles_h.append(abs(angle))
            h_lines_exist = len(angles_h) != 0
            v_lines_exist = len(angles_v) != 0

            mean_h, mean_v = 0, 0
            if h_lines_exist:
                mean_h = mean(angles_h)
                print('mean h', mean_h)
            if v_lines_exist:
                mean_v = mean(angles_v)
                print('mean v', mean_v)

            deviation = 2
            if v_lines_exist or h_lines_exist:
                for x1, y1, x2, y2 in lines[:, 0]:
                    # print(x1, y1, x2, y2)
                    angle = np.arctan((y2-y1)/(x2-x1)) * 180 / np.pi
                    # angle += 180 if angle<0 else 0
                    if h_lines_exist and mean_h-deviation <= abs(angle) <= mean_h+deviation:
                        # print("noise", angle)
                        print('h', angle)
                        cv2.line(norm, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        cv2.line(grid, (row + x1, col + y1), (row + x2, col + y2), 0, 1)
                        continue
                    if v_lines_exist and mean_v-deviation <= abs(angle) <= mean_v+deviation:
                        # print("noise", angle)
                        print('v', angle)
                        cv2.line(norm, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.line(grid, (row + x1, col + y1), (row + x2, col + y2), 0, 1)
                        continue
                    print('n', angle)
                    cv2.line(norm, (x1, y1), (x2, y2), (0, 0, 255), 2)

        tmp[col:col+block_size_h, row:row+block_size_w] = norm
        cv2.imshow('tmp', cv2.pyrDown(tmp))
        cv2.imshow('grid', cv2.pyrDown(grid))
        # cv2.imshow('lines', norm)
        cv2.waitKey(0)

cv2.destroyAllWindows()
cv2.imshow('img', cv2.pyrDown(img))
cv2.imshow('grid', cv2.pyrDown(grid))
cv2.imshow('tmp', cv2.pyrDown(tmp))
cv2.waitKey(0)
cv2.destroyAllWindows()