import cv2 as cv2
import numpy as np
import copy


def clean(block_details, h, w, block_size):
    step = int(block_size // 2)
    grid_h = block_details['grid_h']
    grid_w = block_details['grid_w']

    grid_cleaner = np.ones((h, w), np.uint8) * 255
    block_details_corrected = copy.deepcopy(block_details)

    for nr in range(0, w // step + 1):
        row = nr * step
        for nc in range(0, h // step + 1):
            col = nc * step
            if (nr, nc) in block_details_corrected.keys():
                bd = block_details_corrected[(nr, nc)]
                lh, lv = 0, 0
                # print()
                # print()
                # print('block', nr, nc)
                # bd.print()
                while lv < len(bd.v_line_mid) - 1:
                    # print('v', lv)
                    if bd.v_line_mid[lv + 1] - bd.v_line_mid[lv] <= grid_h / 3:
                        ((a1, b1), (a2, b2)) = (bd.v_line[lv], bd.v_line[lv + 1])
                        bd.v_line[lv] = ((a1 + a2) / 2, (b1 + b2) / 2)
                        # bd.v_line_mid = [(a+b)/2 for a, b in bd.v_line[v]]
                        bd.del_v(lv + 1)
                    else:
                        lv += 1
                while lh < len(bd.h_line_mid) - 1:
                    # print('h', lh)
                    if bd.h_line_mid[lh + 1] - bd.h_line_mid[lh] <= grid_w / 3:
                        ((a1, b1), (a2, b2)) = (bd.h_line[lh], bd.h_line[lh + 1])
                        bd.h_line[lh] = ((a1 + a2) / 2, (b1 + b2) / 2)
                        # bd.h_line_mid = [(a+b)/2 for a, b in bd.h_line[h]]
                        bd.del_h(lh + 1)
                    else:
                        lh += 1
                bd.v_line_mid = [(a + b) / 2 for (a, b) in bd.v_line]
                bd.h_line_mid = [(a + b) / 2 for (a, b) in bd.h_line]

                grid_deviation = 0.10

                # lh, lv = 0, 0
                # h_grid_deviation, v_grid_deviation = (x*grid_deviation for x in (grid_h, grid_w))
                # valid = [False for x in bd.v_line_mid]
                # first_fail = -1

                def validate_lines(line_mid, grid_size):
                    ll = 0
                    # print('grid_size', grid_size)
                    deviation = grid_size * grid_deviation
                    # print('dev', deviation)
                    valid = [False for x in line_mid]
                    if len(valid) > 0:
                        valid[ll] = True
                    first_fail = -1
                    line_mid1 = line_mid[:]
                    to_del1 = []
                    add_mid1 = []
                    while ll < len(line_mid1) - 1:
                        if not valid[ll]:
                            if ll not in to_del1:
                                to_del1.append(ll)
                            ll += 1
                            continue
                        dist = abs(line_mid1[ll + 1] - line_mid1[ll])
                        if grid_size - deviation <= dist <= grid_size + deviation or \
                                2 * (grid_size - deviation) <= dist <= 2 * (grid_size + deviation) or \
                                3 * (grid_size - deviation) <= dist <= 3 * (grid_size + deviation):
                            valid[ll + 1] = True
                            if 2 * (grid_size - deviation) <= dist <= 2 * (grid_size + deviation):
                                add_mid1.append((line_mid1[ll] + line_mid1[ll + 1]) / 2)
                            if 3 * (grid_size - deviation) <= dist <= 3 * (grid_size + deviation):
                                add_mid1.append((2 * line_mid1[ll] + line_mid1[ll + 1]) / 3)
                                add_mid1.append((line_mid1[ll] + 2 * line_mid1[ll + 1]) / 3)
                        else:
                            if first_fail == -1:
                                first_fail = ll + 1
                            to_del1.append(ll + 1)
                            if ll + 2 >= len(line_mid1):
                                ll += 1
                                continue
                            dist = abs(line_mid1[ll + 2] - line_mid1[ll])
                            if grid_size - deviation <= dist <= grid_size + deviation or \
                                    2 * (grid_size - deviation) <= dist <= 2 * (grid_size + deviation) or \
                                    3 * (grid_size - deviation) <= dist <= 3 * (grid_size + deviation):
                                valid[ll + 2] = True
                                if 2 * (grid_size - deviation) <= dist <= 2 * (grid_size + deviation):
                                    add_mid1.append((line_mid1[ll] + line_mid1[ll + 2]) / 2)
                                if 3 * (grid_size - deviation) <= dist <= 3 * (grid_size + deviation):
                                    add_mid1.append((2 * line_mid1[ll] + line_mid1[ll + 2]) / 3)
                                    add_mid1.append((line_mid1[ll] + 2 * line_mid1[ll + 2]) / 3)
                            else:
                                to_del1.append(ll + 2)
                                if ll + 3 >= len(line_mid1):
                                    ll += 1
                                    continue
                                dist = abs(line_mid1[ll + 3] - line_mid1[ll])
                                if grid_size - deviation <= dist <= grid_size + deviation or \
                                        2 * (grid_size - deviation) <= dist <= 2 * (grid_size + deviation) or \
                                        3 * (grid_size - deviation) <= dist <= 3 * (grid_size + deviation):
                                    valid[ll + 3] = True
                                    if 2 * (grid_size - deviation) <= dist <= 2 * (grid_size + deviation):
                                        add_mid1.append((line_mid1[ll] + line_mid1[ll + 3]) / 2)
                                    if 3 * (grid_size - deviation) <= dist <= 3 * (grid_size + deviation):
                                        add_mid1.append((2 * line_mid1[ll] + line_mid1[ll + 3]) / 3)
                                        add_mid1.append((line_mid1[ll] + 2 * line_mid1[ll + 3]) / 3)
                                else:
                                    to_del1.append(ll + 3)
                        ll += 1
                    cnt1 = sum([1 if x == True else 0 for x in valid])

                    if first_fail == -1 or cnt1 >= len(line_mid):
                        return (to_del1, add_mid1)
                    line_mid2 = line_mid[first_fail:]
                    to_del2 = list(range(first_fail))
                    add_mid2 = []
                    valid = [False for x in line_mid2]
                    valid[0] = True
                    ll = 0
                    while ll < len(line_mid2) - 1:
                        # print('ll', ll)
                        if not valid[ll]:
                            if first_fail + ll not in to_del2:
                                to_del2.append(first_fail + ll)
                            ll += 1
                            continue
                        dist = abs(line_mid2[ll + 1] - line_mid2[ll])
                        if grid_size - deviation <= dist <= grid_size + deviation or \
                                2 * (grid_size - deviation) <= dist <= 2 * (grid_size + deviation) or \
                                3 * (grid_size - deviation) <= dist <= 3 * (grid_size + deviation):
                            valid[ll + 1] = True
                            if 2 * (grid_size - deviation) <= dist <= 2 * (grid_size + deviation):
                                add_mid2.append((line_mid2[ll] + line_mid2[ll + 1]) / 2)
                            if 3 * (grid_size - deviation) <= dist <= 3 * (grid_size + deviation):
                                add_mid2.append((2 * line_mid2[ll] + line_mid2[ll + 1]) / 3)
                                add_mid2.append((line_mid2[ll] + 2 * line_mid2[ll + 1]) / 3)
                        else:
                            if first_fail == -1:
                                first_fail = ll + 1
                            to_del2.append(first_fail + ll + 1)
                            if ll + 2 >= len(line_mid2):
                                ll += 1
                                continue
                            dist = abs(line_mid2[ll + 2] - line_mid2[ll])
                            if grid_size - deviation <= dist <= grid_size + deviation or \
                                    2 * (grid_size - deviation) <= dist <= 2 * (grid_size + deviation) or \
                                    3 * (grid_size - deviation) <= dist <= 3 * (grid_size + deviation):
                                valid[ll + 2] = True
                                if 2 * (grid_size - deviation) <= dist <= 2 * (grid_size + deviation):
                                    add_mid2.append((line_mid2[ll] + line_mid2[ll + 2]) / 2)
                                if 3 * (grid_size - deviation) <= dist <= 3 * (grid_size + deviation):
                                    add_mid2.append((2 * line_mid2[ll] + line_mid2[ll + 2]) / 3)
                                    add_mid2.append((line_mid2[ll] + 2 * line_mid2[ll + 2]) / 3)
                            else:
                                to_del2.append(first_fail + ll + 2)
                                if ll + 3 >= len(line_mid2):
                                    ll += 1
                                    continue
                                dist = abs(line_mid2[ll + 3] - line_mid2[ll])
                                if grid_size - deviation <= dist <= grid_size + deviation or \
                                        2 * (grid_size - deviation) <= dist <= 2 * (grid_size + deviation) or \
                                        3 * (grid_size - deviation) <= dist <= 3 * (grid_size + deviation):
                                    valid[ll + 3] = True
                                    if 2 * (grid_size - deviation) <= dist <= 2 * (grid_size + deviation):
                                        add_mid2.append((line_mid2[ll] + line_mid2[ll + 3]) / 2)
                                    if 3 * (grid_size - deviation) <= dist <= 3 * (grid_size + deviation):
                                        add_mid2.append((2 * line_mid2[ll] + line_mid2[ll + 3]) / 3)
                                        add_mid2.append((line_mid2[ll] + 2 * line_mid2[ll + 3]) / 3)
                                else:
                                    to_del2.append(first_fail + ll + 3)
                        ll += 1

                    cnt2 = sum([1 if x == True else 0 for x in valid])
                    return (to_del1, add_mid1) if cnt1 >= cnt2 else (to_del2, add_mid2)

                # if len(bd.v_line_mid) > 0:
                invalid_v, add_v = validate_lines(bd.v_line_mid, grid_h)
                n_del = 0
                # print('invalid_v', invalid_v)
                # print('add_v', add_v)
                for i in invalid_v:
                    bd.del_v(int(i) - n_del)
                    n_del += 1
                for m in add_v:
                    bd.v_line_mid.append(m)
                bd.v_line_mid.sort()

                # if len(bd.h_line_mid) > 0:
                invalid_h, add_h = validate_lines(bd.h_line_mid, grid_w)
                n_del = 0
                # print('invalid_h', invalid_h)
                # print('add_h', add_h)
                for i in invalid_h:
                    bd.del_h(int(i) - n_del)
                    n_del += 1
                for m in add_h:
                    bd.h_line_mid.append(m)
                bd.h_line_mid.sort()

                bd.calc_line_diff()
                # print('processed')
                # bd.print()
                # print()
                # print()
                for [x1, x2] in bd.v_line:
                    y1, y2 = col, col + block_size
                    # print(x1, y1, x2, y2)
                    cv2.line(grid_cleaner, (int(x1), int(y1)), (int(x2), int(y2)), 0, 1)
                for [y1, y2] in bd.h_line:
                    x1, x2 = row + block_size, row
                    # print(x1, y1, x2, y2)
                    cv2.line(grid_cleaner, (int(x1), int(y1)), (int(x2), int(y2)), 0, 1)
                block_details_corrected[(nr, nc)] = bd
            else:
                continue
                # print('block not found', nr, nc)
            # cv2.imshow('grid_cleaner', cv2.pyrDown(grid_cleaner))
            # cv2.waitKey(0)

    # cv2.imshow('grid_cleaner', cv2.pyrDown(grid_cleaner))
    # cv2.waitKey(0)

    return block_details_corrected
