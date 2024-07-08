import cv2
import numpy as np
from math import pi, sqrt


def LoG(sigma):
    n = int(sigma * 6 + 1)
    n = n | 1
    c = -1 / (pi * (sigma ** 4))
    center_x = center_y = n // 2
    kernel = np.zeros((n, n), dtype=np.float32)
    for x in range(n):
        for y in range(n):
            dx = x - center_x
            dy = y - center_y
            v = ((dx ** 2) + (dy ** 2)) / 2 * sigma ** 2
            val = c * (1 - v) * np.exp(-v)
            kernel[x, y] = val
    return kernel


def convolution(img, kernel, row, col):
    pad_top = row
    pad_bot = img.shape[0] - 1 - row
    pad_left = col
    pad_right = img.shape[1] - 1 - col
    out = np.zeros(img.shape, dtype=np.float32)
    bor_img = cv2.copyMakeBorder(img, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res = 0
            for x in range(kernel.shape[0]):
                for y in range(kernel.shape[1]):
                    res += kernel[kernel.shape[0] - 1 - x, kernel.shape[1] - 1 - y] * bor_img[i + x, j + y]
            out[i, j] = res
    return out


def zero_cross(img):
    result = np.zeros(img.shape, dtype=np.uint8)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if (img[i - 1, j] > 0 and img[i + 1, j] < 0) or (img[i - 1, j] < 0 and img[i + 1, j] > 0):
                result[i, j] = img[i, j]
            elif (img[i, j - 1] > 0 and img[i, j + 1] < 0) or (img[i, j - 1] < 0 and img[i, j + 1] > 0):
                result[i, j] = img[i, j]
    return result


def variance_threshold(img, i, j, th, kernel):
    pad = kernel.shape[0] // 2
    loc_reg = img[i - pad:i + pad + 1, j - pad:j + pad + 1]
    loc_std = np.std(loc_reg)
    if loc_std > th:
        return 255
    else:
        return 0


img = cv2.imread("lena.jpg", 0)
kernel = LoG(1)
out = convolution(img, kernel, 3, 3)
# out2=np.zeros(img.shape,dtype=np.uint8)
out2 = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
out2 = np.round(out2).astype(np.uint8)
cv2.imshow("input", img)
cv2.imshow("convolved", out2)
zero = zero_cross(out)
cv2.imshow("zero_crossed", zero)
result = np.zeros(img.shape, dtype=np.uint8)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if (zero[i, j] > 0):
            result[i, j] = variance_threshold(img, i, j, 10, kernel)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()