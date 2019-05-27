import cv2 as cv2
import numpy as np
import copy
import glob
import time
from statistics import mean, median, mode

# flist = glob.glob("graphs/*_*.jpg")
flist = glob.glob("graphs/sample_graph.jpg")
flist.sort()

for file in flist:
    start_time = time.time()
    print()
    print()
    print()
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
    cv2.imshow('img', cv2.pyrDown(img))

    h, w, _ = img.shape
    print("w:", w, "h:", h)

    blur = cv2.GaussianBlur(img, (3, 3), 0)
    # cv2.imshow('blur', cv2.pyrDown(blur))
    # cv2.waitKey(0)

    block_size_w = w // num_blocks_h
    block_size_h = h // num_blocks_v
    # print("block size", block_size_h, block_size_w)

    angle_deviation_threshold = 2

    tmp = copy.deepcopy(img)
    grid = np.ones((h, w), np.uint8) * 255
    for row in range(0, w, block_size_w // 2):
        for col in range(0, h, block_size_h // 2):
            block = blur[col:col + block_size_h, row:row + block_size_w]
            # cv2.imshow('block', block)
            # cv2.waitKey(0)
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


    # cv2.imshow('img', cv2.pyrDown(img))
    # cv2.imshow('grid', cv2.pyrDown(grid))
    # cv2.imshow('tmp', cv2.pyrDown(tmp))
    # cv2.waitKey(0)

    class BlockDetail:
        def __init__(self):
            self.v_line_start = []
            self.v_line_end = []
            self.v_line_mid = []
            self.v_line = []
            self.h_line_start = []
            self.h_line_end = []
            self.h_line_mid = []
            self.h_line = []
            self.v_angle = None
            self.h_angle = None

        def print(self):
            # print('v_line_start', self.v_line_start)
            print('v_line_mid', self.v_line_mid)
            # print('v_line_end', self.v_line_end)
            print('v_line', self.v_line)
            # print('h_line_start', self.h_line_start)
            print('h_line_mid', self.h_line_mid)
            # print('h_line_end', self.h_line_end)
            print('h_line', self.h_line)
            print('v_angle', self.v_angle)
            print('h_angle', self.h_angle)


    block_details = {}
    max_num_blocks = 10

    block_size = int(max(w, h) // max_num_blocks)

    tmp_grid = copy.deepcopy(grid)
    tmp_grid = cv2.cvtColor(tmp_grid, cv2.COLOR_GRAY2BGR)
    grid_clean = np.ones((h, w), np.uint8) * 255

    rho_diff_h = []
    rho_diff_v = []

    step = int(block_size // 2)

    # print('step', step)

    for row in range(int(0), int(w), int(step)):
        # print('row', row)
        nr = int(row / step)
        for col in range(int(0), int(h), int(step)):
            nc = int(col / step)
            # print('col', col)
            # print("rc", row, col)
            # print(nr, nc)

            bd = BlockDetail()

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
                angle_v = []
                angle_h = []

                for rho, theta in lines[:, 0]:
                    # print(rho, theta)
                    angle = theta * 180 / np.pi
                    while angle > 135:
                        angle -= 180
                    while angle < -45:
                        angle += 180
                    # print(rho, angle)
                    if abs(angle) > angle_threshold_v:
                        # print('append v')
                        rho_v.append(rho)
                        angle_v.append(angle)
                        # print('vert')
                        continue
                    elif abs(angle) < angle_threshold_h:
                        # print('append h')
                        rho_h.append(rho)
                        angle_h.append(angle)
                        # print('hori')
                        continue

                rho_v.sort()
                rho_h.sort()

                angle_median_v = 0
                if len(angle_v) > 0:
                    angle_median_v = abs(median(angle_v))
                    bd.v_angle = angle_median_v

                angle_median_h = 90
                if len(angle_h) > 0:
                    angle_median_h = abs(median(angle_h))
                    bd.h_angle = angle_median_h

                # print('angle_v', angle_v)
                # print('median_v', angle_median_v)
                # print()
                # print('angle_h', angle_h)
                # print('median_h', angle_median_h)
                # print()

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
                    while angle > 135:
                        angle -= 180
                    while angle < -45:
                        angle += 180
                    c = np.cos(theta)
                    s = np.sin(theta)
                    x0 = rho * c
                    y0 = rho * s

                    # print('sin', s)
                    # print('cos', c)

                    # print(x1, y1, x2, y2)

                    if rho in separate_rho_h or rho in separate_rho_v:
                        # print(rho, angle)
                        if angle_median_v - angle_deviation_threshold <= abs(
                                angle) <= angle_median_v + angle_deviation_threshold and rho in separate_rho_v:
                            # print('vert_line')
                            x1 = block_size
                            y1 = int(y0 - abs((x0-x1) / s) * c)
                            x2 = 0
                            y2 = int(y0 + abs((x2-x0) / s) * c)
                            # print('v_coord', x1, y1, x2, y2)
                            bd.h_line_start.append(int((nc * step) + y1))
                            bd.h_line_end.append(int((nc * step) + y2))
                            bd.h_line_mid.append(int((nc * step) + (y1 + y2)/2))
                            bd.h_line.append((int((nc * step) + y1), int((nc * step) + y2)))
                            block_rho_v.append(rho)
                            cv2.line(block_grid_bgr, (x1, y1), (x2, y2), (0, 255, 0), 4)
                            cv2.line(grid_clean, (int((nr * step) + x1), int((nc * step) + y1)), (int((nr * step) + x2), int((nc * step) + y2)), 0, 3)
                            # cv2.line(tmp_grid, (col + x1, row + y1), (col + x2, row + y2), 255, 1)
                            continue
                        elif angle_median_h - angle_deviation_threshold <= abs(
                                angle) <= angle_median_h + angle_deviation_threshold and rho in separate_rho_h:
                            # print('hori_line')
                            y1 = 0
                            x1 = int(x0 + ((y0-y1) / c) * s)
                            y2 = block_size
                            x2 = int(x0 - ((y2-y1) / c) * s)
                            # print('h_coord', x1, y1, x2, y2)
                            bd.v_line_start.append(int((nr * step) + x1))
                            bd.v_line_end.append(int((nr * step) + x2))
                            bd.v_line_mid.append(int((nr * step) + (x1 + x2)/2))
                            bd.v_line.append((int((nr * step) + x1), int((nr * step) + x2)))
                            block_rho_h.append(rho)
                            cv2.line(block_grid_bgr, (x1, y1), (x2, y2), (255, 0, 0), 4)
                            cv2.line(grid_clean, (int((nr * step) + x1), int((nc * step) + y1)), (int((nr * step) + x2), int((nc * step) + y2)), 0, 3)
                            # cv2.line(tmp_grid, (col + x1, row + y1), (col + x2, row + y2), 255, 1)
                            continue
                        else:
                            # print('discarded_line')
                            x1 = int(x0 + 1000 * s)
                            y1 = int(y0 - 1000 * c)
                            x2 = int(x0 - 1000 * s)
                            y2 = int(y0 + 1000 * c)
                            cv2.line(block_grid_bgr, (x1, y1), (x2, y2), (0, 0, 255), 4)
                            # cv2.line(tmp_grid, (row + x1, col + y1), (row + x2, col + y2), 127, 1)
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

            bd.v_line.sort()
            bd.v_line_mid.sort()
            bd.h_line.sort()
            bd.h_line_mid.sort()

            print((nr, nc))
            bd.print()
            print()
            block_details[(nr, nc)] = bd

            tmp_grid[col:col + block_size, row:row + block_size] = block_grid_bgr
            # grid_clean[col:col + block_size, row:row + block_size] &= block_grid_clean
            # cv2.imshow('grid', cv2.pyrDown(tmp_grid))
            # cv2.imshow('block_grid', block_grid_bgr)
            # cv2.imshow('grid_clean', cv2.pyrDown(grid_clean))
            # cv2.waitKey(0)

    # for key, val in block_details.items():
    #     print(key)
    #     val.print()
    #     print()

    print('rho_diff_v', rho_diff_v)
    print('rho_diff_h', rho_diff_h)

    grid_w = median(rho_diff_v)
    grid_h = median(rho_diff_h)

    print("grid_size", grid_w, grid_h)

    end_time = time.time()
    process_time = end_time - start_time
    print("processed in", process_time)

    cv2.imshow('grid_clean', cv2.pyrDown(grid_clean))
    # cv2.imshow('tmp', cv2.pyrDown(tmp))
    # cv2.imshow('grid', cv2.pyrDown(tmp_grid))

    # average_color = img.mean(axis=0).mean(axis=0)
    # print('avg color', average_color)
    # avg_mask = np.ones((h, w, 3), np.uint8)
    # avg_mask[:] = average_color
    avg_mask = cv2.filter2D(img, -1, np.ones((101, 101), np.uint8))
    white_mask = cv2.bitwise_and(cv2.cvtColor(~grid_clean, cv2.COLOR_GRAY2BGR), avg_mask)
    cv2.imshow('white_mask', cv2.pyrDown(white_mask))

    masked = cv2.bitwise_or(img, white_mask)
    cv2.imshow('masked', cv2.pyrDown(masked))
    cv2.waitKey(0)

    # cv2.destroyAllWindows()

cv2.destroyAllWindows()
