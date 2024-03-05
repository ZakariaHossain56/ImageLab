import math

import cv2
from sobel_filter import *


def merge(horizontal_convoluted, vertical_convoluted):
    height, width = horizontal_convoluted.shape
    output = np.zeros_like(horizontal_convoluted, dtype='float32')

    for x in range(0, height):
        for y in range(0, width):
            dx = horizontal_convoluted[x, y]
            dy = vertical_convoluted[x, y]
            res = math.sqrt(dx ** 2 + dy ** 2)
            output[x, y] = res
    # print("Merged output")
    # print(output)
    return output

def non_maximum_suppression(gradient_magnitude,gradient_angle):
    M, N = gradient_magnitude.shape
    suppressed = np.zeros((M, N), dtype=np.int32)  # resultant image
    angle = gradient_angle * 180. / np.pi  # max -> 180, min -> -180
    angle[angle < 0] += 180  # max -> 180, min -> 0

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 255
            r = 255

            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                r = gradient_magnitude[i, j - 1]
                q = gradient_magnitude[i, j + 1]

            elif (22.5 <= angle[i, j] < 67.5):
                r = gradient_magnitude[i - 1, j + 1]
                q = gradient_magnitude[i + 1, j - 1]

            elif (67.5 <= angle[i, j] < 112.5):
                r = gradient_magnitude[i - 1, j]
                q = gradient_magnitude[i + 1, j]

            elif (112.5 <= angle[i, j] < 157.5):
                r = gradient_magnitude[i + 1, j + 1]
                q = gradient_magnitude[i - 1, j - 1]

            if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                suppressed[i, j] = gradient_magnitude[i, j]
            else:
                suppressed[i, j] = 0
    return suppressed


def double_thresholding(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):

    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)

def hysteresis(img, weak, strong=255):
    M, N = img.shape

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i, j] == weak):
                if (
                    (img[i+1, j-1] == strong) or (img[i+1, j] == strong) or
                    (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or
                    (img[i, j+1] == strong) or (img[i-1, j-1] == strong) or
                    (img[i-1, j] == strong) or (img[i-1, j+1] == strong)
                ):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img


img=cv2.imread("lena.jpg",cv2.IMREAD_GRAYSCALE)
cv2.imshow("Input grayscaled image",img)
cv2.waitKey(0)

image = cv2.GaussianBlur(img, (3, 3), 0)
cv2.imshow("Gaussian Convoluted image", image)
cv2.waitKey(0)

horizontal_filter,vertical_filter = sobel()

horizontal_convolution = cv2.filter2D(image, -1, horizontal_filter)
normalized_horizontal_convolution=cv2.normalize(horizontal_convolution,horizontal_convolution, 0, 255, cv2.NORM_MINMAX)
normalized_horizontal_convolution = np.round(normalized_horizontal_convolution).astype(np.uint8)
cv2.imshow("Horizontally convoluted image",normalized_horizontal_convolution)

vertical_convolution = cv2.filter2D(image, -1, vertical_filter)
normalized_vertical_convolution=cv2.normalize(vertical_convolution,vertical_convolution, 0, 255, cv2.NORM_MINMAX)
normalized_vertical_convolution = np.round(normalized_vertical_convolution).astype(np.uint8)
cv2.imshow("Vertically convoluted image",normalized_vertical_convolution)

gradient_magnitude = merge(horizontal_convolution, vertical_convolution)
normalized_gradient_magnitude=cv2.normalize(gradient_magnitude,gradient_magnitude, 0, 255, cv2.NORM_MINMAX)
normalized_gradient_magnitude = np.round(normalized_gradient_magnitude).astype(np.uint8)
cv2.imshow("Gradient Magnitude image",normalized_gradient_magnitude)
cv2.waitKey(0)

gradient_angle = np.arctan2(vertical_convolution, horizontal_convolution)

suppresssed_image = non_maximum_suppression(gradient_magnitude,gradient_angle)
normalized_suppressed_image=cv2.normalize(suppresssed_image,suppresssed_image, 0, 255, cv2.NORM_MINMAX)
normalized_suppressed_image = np.round(normalized_suppressed_image).astype(np.uint8)
cv2.imshow("Non maximum suppression",normalized_suppressed_image)
cv2.waitKey(0)

double_threshold_result,weak,strong = double_thresholding(suppresssed_image)
normalized_double_threshold_result=cv2.normalize(double_threshold_result,double_threshold_result, 0, 255, cv2.NORM_MINMAX)
normalized_double_threshold_result = np.round(normalized_double_threshold_result).astype(np.uint8)
cv2.imshow("Double thresholding",normalized_double_threshold_result)
cv2.waitKey(0)

hysteresis_output=hysteresis(double_threshold_result,weak,strong)
normalized_hysteresis_output=cv2.normalize(hysteresis_output,hysteresis_output, 0, 255, cv2.NORM_MINMAX)
normalized_hysteresis_output = np.round(normalized_hysteresis_output).astype(np.uint8)
cv2.imshow("Final hysteresis output",normalized_hysteresis_output)

cv2.waitKey(0)
cv2.destroyAllWindows()