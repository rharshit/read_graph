import cv2 as cv2
import numpy as np
import copy
import glob, os
from statistics import mean, median, mode

# flist = glob.glob("graphs/*_*.jpg")
flist = glob.glob("graphs/sample_graph.jpg")
flist.sort()

for file in flist:
    print(file)
    image_type = 0
    if 'fhr' in file:
        image_type = 1
    elif 'uc' in file:
        image_type = 2
    if image_type == 0:
        print('sample')
        # continue

    print('processing', ('FHR' if image_type == 1 else 'UC'))

    # fname = 'graphs/sample_graph.jpg'

    num_blocks_h = 10
    num_blocks_v = 6

    angle_threshold_v = 75
    angle_threshold_h = 15

    img = cv2.imread(file)
    h, w, _ = img.shape
    print("w:", w, "h:", h)

    blur = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.imshow('blur', blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    block_size_w = w // num_blocks_h
    block_size_h = h // num_blocks_v
    print("block size", block_size_h, block_size_w)

    tmp = copy.deepcopy(img)
    grid = np.ones((h, w), np.uint8) * 255
    for row in range(0, w, block_size_w // 2):
        for col in range(0, h, block_size_h // 2):
            block = blur[col:col + block_size_h, row:row + block_size_w]
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

            lines = cv2.HoughLinesP(block_close, 1, np.pi / 360, block_size_h // 2, None, block_size_h // 2,
                                    block_size_h // 2)
            # print(len(lines[:,0]))
            # print()
            angles_h = []
            angles_v = []
            if lines is not None:
                for x1, y1, x2, y2 in lines[:, 0]:
                    angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
                    if abs(angle) >= angle_threshold_v:
                        angles_v.append(abs(angle))
                    if abs(angle) <= angle_threshold_h:
                        angles_h.append(abs(angle))
                h_lines_exist = len(angles_h) != 0
                v_lines_exist = len(angles_v) != 0

                median_h, median_v = 0, 0
                if h_lines_exist:
                    median_h = median(angles_h)
                    # print('mean h', mean_h)
                if v_lines_exist:
                    median_v = median(
                        angles_v)
                    # print('mean v', mean_v)

                angle_deviation_threshold = 2
                if v_lines_exist or h_lines_exist:
                    for x1, y1, x2, y2 in lines[:, 0]:
                        # print(x1, y1, x2, y2)
                        angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
                        # angle += 180 if angle<0 else 0
                        if h_lines_exist and median_h - angle_deviation_threshold <= abs(
                                angle) <= median_h + angle_deviation_threshold:
                            # print("noise", angle)
                            # print('h', angle)
                            cv2.line(norm, (x1, y1), (x2, y2), (255, 0, 255), 2)
                            cv2.line(grid, (row + x1, col + y1), (row + x2, col + y2), 0, 1)
                            continue
                        if v_lines_exist and median_v - angle_deviation_threshold <= abs(
                                angle) <= median_v + angle_deviation_threshold:
                            # print("noise", angle)
                            # print('v', angle)
                            cv2.line(norm, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.line(grid, (row + x1, col + y1), (row + x2, col + y2), 0, 1)
                            continue
                        # print('n', angle)
                        cv2.line(norm, (x1, y1), (x2, y2), (0, 0, 255), 2)

            tmp[col:col + block_size_h, row:row + block_size_w] = norm
            # cv2.imshow('tmp', cv2.pyrDown(tmp))
            # cv2.imshow('grid', cv2.pyrDown(grid))
            # cv2.imshow('lines', norm)
            # cv2.waitKey(0)

    cv2.destroyAllWindows()
    cv2.imshow('img', cv2.pyrDown(img))
    cv2.imshow('grid', cv2.pyrDown(grid))
    cv2.imshow('tmp', cv2.pyrDown(tmp))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    max_num_blocks = 10

    block_size = int(max(w, h) // max_num_blocks)

    tmp = copy.deepcopy(img)
    tmp_grid = copy.deepcopy(grid)
    tmp_grid = cv2.cvtColor(tmp_grid, cv2.COLOR_GRAY2BGR)
    grid_clean = np.ones((h, w), np.uint8) * 255

    rho_diff_h = []
    rho_diff_v = []

    step = int(block_size // 2)

    print('step', step)

    for row in range(int(0), int(w), int(step)):
        # print('row', row)
        for col in range(int(0), int(h), int(step)):
            # print('col', col)
            # print("rc", row, col)
            block = tmp[col:col + block_size, row:row + block_size]
            block_grid = grid[col:col + block_size, row:row + block_size]
            block_grid_bgr = cv2.cvtColor(block_grid, cv2.COLOR_GRAY2BGR)
            block_grid_clean = np.ones((block_grid.shape[0:2]), np.uint8) * 255
            block_w, block_h = block_grid.shape[0:2]
            # print(block_h, block_w)

            block_rho_h = []
            block_rho_v = []

            lines = cv2.HoughLines(~block_grid, 1, np.pi / 360, int(block_size * 0.9))

            if lines is not None:

                separate_rho_h = []
                separate_rho_v = []
                tmp_v = []
                tmp_h = []
                prev_h = None
                prev_v = None
                rho_threshold = 5
                rho_v = []
                rho_h = []

                for rho, theta in lines[:, 0]:
                    # print(rho, theta)
                    angle = theta * 180 / np.pi
                    while (angle > 90):
                        angle -= 180
                    while (angle < -90):
                        angle += 180
                    # print(rho, angle)
                    if abs(angle) > angle_threshold_v:
                        # print('append v')
                        rho_v.append(rho)
                        continue
                    elif abs(angle) < angle_threshold_h:
                        # print('append h')
                        rho_h.append(rho)
                        continue

                rho_v.sort()
                rho_h.sort()

                # print('v', rho_v)
                # print('h', rho_h)

                for rho in rho_v:
                    if prev_v is None:
                        prev_v = rho
                    diff = rho - prev_v
                    # print("dv", rho, prev_v, diff)

                    if diff > rho_threshold:
                        tmp_v.sort()
                        med_v = tmp_v[len(tmp_v) // 2]
                        # print('v', v)
                        separate_rho_v.append(med_v)
                        tmp_v = []
                    prev_v = rho
                    tmp_v.append(rho)
                if len(tmp_v) != 0:
                    med_v = median(tmp_v)
                    # print('v', v)
                    separate_rho_v.append(med_v)

                for rho in rho_h:
                    if prev_h is None:
                        prev_h = rho
                    diff = rho - prev_h
                    # print("dh", rho, prev_h, diff)

                    if diff > rho_threshold:
                        tmp_h.sort()
                        med_h = tmp_h[len(tmp_h) // 2]
                        # print('h', h)
                        separate_rho_h.append(med_h)
                        tmp_h = []
                    prev_h = rho
                    tmp_h.append(rho)
                if len(tmp_h) != 0:
                    med_h = median(tmp_h)
                    # print('h', h)
                    separate_rho_h.append(med_h)

                # print('sv', separate_rho_v)
                # print('sh', separate_rho_h)

                for rho, theta in lines[:, 0]:
                    # print(rho, theta)
                    angle = theta * 180 / np.pi
                    while (angle > 90):
                        angle -= 180
                    while (angle < -90):
                        angle += 180
                    # print(rho, angle)
                    c = np.cos(theta)
                    s = np.sin(theta)
                    x0 = rho * c
                    y0 = rho * s
                    x1 = int(x0 + 1000 * s)
                    y1 = int(y0 - 1000 * c)
                    x2 = int(x0 - 1000 * s)
                    y2 = int(y0 + 1000 * c)

                    # print(x1, y1, x2, y2)

                    if rho in separate_rho_h or rho in separate_rho_v:
                        if abs(angle) > angle_threshold_v and rho in separate_rho_v:
                            # print('v_coord', x1, y1, x2, y2)
                            block_rho_v.append(rho)
                            cv2.line(block_grid_bgr, (x1, y1), (x2, y2), (0, 255, 0), 1)
                            cv2.line(block_grid_clean, (x1, y1), (x2, y2), 0, 1)
                            cv2.line(tmp_grid, (col + x1, row + y1), (col + x2, row + y2), 255, 1)
                            continue
                        elif abs(angle) < angle_threshold_h and rho in separate_rho_h:
                            # print('v_coord', x1, y1, x2, y2)
                            block_rho_h.append(rho)
                            cv2.line(block_grid_bgr, (x1, y1), (x2, y2), (255, 0, 0), 1)
                            cv2.line(block_grid_clean, (x1, y1), (x2, y2), 0, 1)
                            cv2.line(tmp_grid, (col + x1, row + y1), (col + x2, row + y2), 255, 1)

                            continue
                    else:
                        continue
                        # print("", end="")
                        # print('extras', x1, y1, x2, y2)
                        # cv2.line(block_grid_bgr, (x1, y1), (x2, y2), (0, 0, 255), 1)

                for i in range(1, len(separate_rho_v)):
                    # print(i)
                    rho_diff_v.append(separate_rho_v[i] - separate_rho_v[i - 1])
                for i in range(1, len(separate_rho_h)):
                    # print(i)
                    rho_diff_h.append(separate_rho_h[i] - separate_rho_h[i - 1])

            else:
                continue
                # print('no lines found')

            block_rho_v.sort()
            block_rho_h.sort()

            # print()
            # print("rho v", block_rho_v)
            # print("rho h", block_rho_h)
            # print()

            tmp_grid[col:col + block_size, row:row + block_size] = block_grid_bgr
            grid_clean[col:col + block_size, row:row + block_size] &= block_grid_clean
            # cv2.imshow('grid', cv2.pyrDown(tmp_grid))
            # cv2.imshow('block_grid', block_grid_bgr)
            # cv2.imshow('grid_clean', cv2.pyrDown(grid_clean))
            # cv2.waitKey(0)

        # print('rho_diff_v', rho_diff_v)
        # print('rho_diff_h', rho_diff_h)

    grid_w = median(rho_diff_v)
    grid_h = median(rho_diff_h)

    print("grid_size", grid_w,grid_h)

    cv2.destroyAllWindows()

cv2.destroyAllWindows()
