import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

from Lab2 import convolution
from Lab2 import gaussian_filter


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


def find_avg(image, t=-1):
    total1 = 0
    total2 = 0
    c1 = 0
    c2 = 0

    h, w = image.shape
    for x in range(h):
        for y in range(w):
            px = image[x][y]
            if px > t:
                total2 += px
                c2 += 1
            else:
                total1 += px
                c1 += 1
    mu1 = total1 / c1
    mu2 = total2 / c2

    return (mu1 + mu2) / 2


def find_threshold(image):
    total = 0
    h, w = image.shape
    for x in range(h):
        for y in range(w):
            px = image[x, y]
            total += px
    t = total / (h * w)

    dif = find_avg(image=image, t=t)
    while (abs(dif - t) > 0.000001):

        t = dif
        dif = find_avg(image=image, t=t)

    return dif


def make_binary(threshold, image, low=0, high=255):
    out = image.copy()
    h, w = image.shape
    for x in range(h):
        for y in range(w):
            v = image[x, y]

            out[x, y] = high if v > threshold else low

    return out


def plot_historgram(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Plot histogram
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(histogram)
    plt.xlim([0, 256])
    plt.show()


def derivative():
    kernel = gaussian_filter.gaussian(sigmax=0.7, sigmay=0.7)
    size = len(kernel)

    x_derivative = np.zeros((size, size))
    y_derivative = np.zeros((size, size))

    min1 = 1e2
    min2 = 1e2

    cx = size // 2
    for x in range(size):
        for y in range(size):
            cal1 = -(x - cx) / (0.7 ** 2)
            cal2 = -(y - cx) / (0.7 ** 2)
            x_derivative[x, y] = cal1 * kernel[x, y]
            y_derivative[x, y] = cal2 * kernel[x, y]

            if x_derivative[x, y] != 0:
                min1 = min(abs(x_derivative[x, y]), min1)

            if y_derivative[x, y] != 0:
                min2 = min(abs(y_derivative[x, y]), min2)

    normalized_x_derivative = (x_derivative / min1).astype(int)
    normalized_y_derivative = (y_derivative / min2).astype(int)

    # print("actual value")
    # print(x_derivative)
    # print(y_derivative)
    #
    # print("normalized value")
    # print(normalized_x_derivative)     #sobel vertical kernel
    # print(normalized_y_derivative)     #sobel horizontal kernel
    return (x_derivative, y_derivative)

# derivative()
# merge()


image = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Input grayscale image",image)
cv2.waitKey(0)


image = cv2.GaussianBlur(image, (3, 3), 0)
cv2.imshow("Gaussian Convoluted image", image)
cv2.waitKey(0)

x_derivative, y_derivative = derivative()
# print("x_derivative kernel")
# print(x_derivative)
# print("y-derivative kernel")
# print(y_derivative)


convolution_x_derivative = convolution.convolution(image,y_derivative)
convolution_y_derivative = convolution.convolution(image,x_derivative)

gaussian_filter = gaussian_filter.gaussian()

convolution_x_derivative = convolution.convolution(convolution_x_derivative,gaussian_filter)
convolution_y_derivative = convolution.convolution(convolution_y_derivative,gaussian_filter)

merged_output = merge(convolution_x_derivative, convolution_y_derivative)


# print(convolution_x_derivative)
# print(convolution_y_derivative)
# print(merged_output)


normalized_convolution_x_derivative=cv2.normalize(convolution_x_derivative,convolution_x_derivative, 0, 255, cv2.NORM_MINMAX)
normalized_convolution_x_derivative = np.round(normalized_convolution_x_derivative).astype(np.uint8)

normalized_convolution_y_derivative=cv2.normalize(convolution_y_derivative,convolution_y_derivative, 0, 255, cv2.NORM_MINMAX)
normalized_convolution_y_derivative = np.round(normalized_convolution_y_derivative).astype(np.uint8)

normalized_merged_output=cv2.normalize(merged_output,merged_output, 0, 255, cv2.NORM_MINMAX)
normalized_merged_output = np.round(normalized_merged_output).astype(np.uint8)

cv2.imshow("X derivative convolution", normalized_convolution_x_derivative)
cv2.imshow("Y derivative convolution", normalized_convolution_y_derivative)
cv2.imshow("Merged output", normalized_merged_output)
cv2.waitKey(0)


threshold = find_threshold(image=normalized_merged_output)

final_output = make_binary(threshold=threshold * 0.8,image=normalized_merged_output)
cv2.imshow("Final detected edge", final_output)
cv2.waitKey(0)

plot_historgram(final_output)

cv2.waitKey(0)
cv2.destroyAllWindows()