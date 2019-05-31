import cv2 as cv2
from statistics import mean


def is_blur(img):
    num_parts = 4
    block_size_h, block_size_w = img.shape[0] // num_parts, img.shape[1] // num_parts
    print('size', block_size_h, block_size_w)
    blur = []
    for i in range(num_parts):
        row = i * block_size_w
        for j in range(num_parts):
            col = j * block_size_h
            block = img[col:col + block_size_h, row:row + block_size_w]
            blur_val = cv2.Laplacian(block, cv2.CV_64F).var()
            blur.append(blur_val)
    blurriness = mean(blur)
    return blurriness <= 600