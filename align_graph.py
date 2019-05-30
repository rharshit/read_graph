import cv2 as cv2
import numpy as np
import copy

def align(block_details_corrected, h, w, block_size):
    step = int(block_size // 2)
    grid_h = block_details_corrected['grid_h']
    grid_w = block_details_corrected['grid_w']

    def align_block(bd1, bd2, orientation, index, size):
        if orientation not in ['v', 'h']:
            print('invalid orientation')
            return
        deviation = size // 5
        if orientation is 'v':
            mid = bd1.v_line_mid[:]
            line = bd2.v_line[:]
            diff = bd2.diff_x
            fin_mid = bd1.prob_v_line_mid
            fin_line = bd2.prob_v_line
        else:
            mid = bd1.h_line_mid[:]
            line = bd2.h_line[:]
            diff = bd2.diff_y
            fin_mid = bd1.prob_h_line_mid
            fin_line = bd2.prob_h_line

        # print('size', size)
        # print('dev', deviation)
        # print('mid', mid)
        # print('line', line)
        # print('diff', diff)
        num_lines = block_size / size
        produce_new_line = len(line) > num_lines // 3
        # print('num_lines', num_lines)
        # print('prod_line', produce_new_line)
        nl, nm = 0, 0
        while nm < len(mid):
            added = False
            # print('nm', nm)
            if nl < len(line):
                while nl < len(line):
                    # print('nl', nl)
                    if nl >= len(line):
                        break
                    ln = line[nl]
                    # print('ln', ln)
                    delta = mid[nm] - ln[index]
                    # print('delta', delta)
                    if abs(delta) <= deviation:
                        # print(':1')
                        fin_line.append((ln[0] + delta, ln[1] + delta))
                        added = True
                    else:
                        # print(':2')
                        if delta < 0:
                            # print(':3')
                            break
                    nl += 1
            else:
                # print(':4')
                added = False
            if added:
                # print(':5')
                fin_mid.append(mid[nm])
            else:
                # print(":6")
                # print('do nothing')
                if diff is not None and produce_new_line:
                    # print(':7')
                    ln = None
                    if index == 0:
                        # print(':8')
                        ln = (mid[nm], mid[nm] - diff)
                    else:
                        # print(':9')
                        ln = (mid[nm] + diff, mid[nm])
                    # print(':ln', ln)
                    fin_line.append(ln)
                    fin_mid.append(mid[nm])
            nm += 1
        # print('fin_mid', fin_mid)
        # print('fin_line', fin_line)
        if orientation is 'v':
            # print(":1")
            bd1.prob_v_line_mid = fin_mid
            bd2.prob_v_line = fin_line
        else:
            # print(":2")
            bd1.prob_h_line_mid = fin_mid
            bd2.prob_h_line = fin_line
        return

    grid_align = np.ones((h, w), np.uint8) * 255

    block_details_corrected_tblr = copy.deepcopy(block_details_corrected)  # top-bottom left-right

    for nr in range(0, w // step + 1):
        row = nr * step
        for nc in range(0, h // step + 1):
            col = nc * step
            if (nr, nc) in block_details_corrected_tblr.keys():
                # print(nr, nc)
                bd1 = block_details_corrected[(nr, nc)]
                # print('bd1')
                # bd1.print()
                if (nr + 1, nc) in block_details_corrected_tblr.keys():
                    # print((nr + 1, nc))
                    bd2 = block_details_corrected_tblr[(nr + 1, nc)]
                    # print('bd2')
                    # bd2.print()
                    align_block(bd1, bd2, 'h', 1, grid_w)
                    block_details_corrected_tblr[(nr + 1, nc)] = bd2
                    # print('bd2')
                    # bd2.print_prob()
                    for [x1, x2] in bd2.prob_v_line:
                        y1, y2 = col, col + block_size
                        # print(x1, y1, x2, y2)
                        cv2.line(grid_align, (int(x1), y1), (int(x2), y2), 0, 1)
                    for [y1, y2] in bd2.prob_h_line:
                        x1, x2 = row + block_size + step, row + step
                        # print(x1, y1, x2, y2)
                        cv2.line(grid_align, (x1, int(y1)), (x2, int(y2)), 0, 1)
                if (nr - 1, nc) in block_details_corrected_tblr.keys():
                    # print((nr - 1, nc))
                    bd2 = block_details_corrected_tblr[(nr - 1, nc)]
                    # print('bd2')
                    # bd2.print()
                    align_block(bd1, bd2, 'h', 0, grid_w)
                    block_details_corrected_tblr[(nr - 1, nc)] = bd2
                    # print('bd2')
                    # bd2.print_prob()
                    for [x1, x2] in bd2.prob_v_line:
                        y1, y2 = col, col + block_size
                        # print(x1, y1, x2, y2)
                        cv2.line(grid_align, (int(x1), y1), (int(x2), y2), 0, 1)
                    for [y1, y2] in bd2.prob_h_line:
                        x1, x2 = row + block_size - step, row - step
                        # print(x1, y1, x2, y2)
                        cv2.line(grid_align, (x1, int(y1)), (x2, int(y2)), 0, 1)
                if (nr, nc + 1) in block_details_corrected_tblr.keys():
                    # print((nr, nc + 1))
                    bd2 = block_details_corrected_tblr[(nr, nc + 1)]
                    # print('bd2')
                    # bd2.print()
                    align_block(bd1, bd2, 'v', 0, grid_h)
                    block_details_corrected_tblr[(nr, nc + 1)] = bd2
                    # print('bd2')
                    # bd2.print_prob()
                    for [x1, x2] in bd2.prob_v_line:
                        y1, y2 = col + step, col + block_size + step
                        # print(x1, y1, x2, y2)
                        cv2.line(grid_align, (int(x1), y1), (int(x2), y2), 0, 1)
                    for [y1, y2] in bd2.prob_h_line:
                        x1, x2 = row + block_size, row
                        # print(x1, y1, x2, y2)
                        cv2.line(grid_align, (x1, int(y1)), (x2, int(y2)), 0, 1)
                if (nr, nc - 1) in block_details_corrected_tblr.keys():
                    # print((nr, nc - 1))
                    bd2 = block_details_corrected_tblr[(nr, nc - 1)]
                    # print('bd2')
                    # bd2.print()
                    align_block(bd1, bd2, 'v', 1, grid_h)
                    block_details_corrected_tblr[(nr, nc - 1)] = bd2
                    # print('bd2')
                    # bd2.print_prob()
                    for [x1, x2] in bd2.prob_v_line:
                        y1, y2 = col - step, col + block_size - step
                        # print(x1, y1, x2, y2)
                        cv2.line(grid_align, (int(x1), y1), (int(x2), y2), 0, 1)
                    for [y1, y2] in bd2.prob_h_line:
                        x1, x2 = row + block_size, row
                        # print(x1, y1, x2, y2)
                        cv2.line(grid_align, (x1, int(y1)), (x2, int(y2)), 0, 1)
                block_details_corrected[(nr, nc)] = bd1
                # print('bd1')
                # bd1.print_prob()
            else:
                # print('block not found')
                continue

            # cv2.imshow('align', cv2.pyrDown(grid_align))
            # cv2.waitKey(0)

    cv2.imshow('align', cv2.pyrDown(grid_align))
    # cv2.waitKey(0)

    return block_details_corrected_tblr
