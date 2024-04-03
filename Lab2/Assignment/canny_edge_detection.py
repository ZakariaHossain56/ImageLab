import math
import cv2
from Lab2 import convolution
from Lab2  import gaussian_filter
from Lab2.Assignment.derivative import derivative
from sobel_filter import *


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
    image = gradient_magnitude.copy()

    image = image / image.max() * 255

    M, N = image.shape
    suppressed = np.zeros((M, N), dtype=np.uint8)

    angle = gradient_angle * 180. / np.pi  # max -> 180, min -> -180

    c1 = c2 = c3 = c4 = 0
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 0
            r = 0

            ang = angle[i, j]

            if (-22.5 <= ang < 22.5) or (157.5 <= ang <= 180) or (-180 <= ang <= -157.5):
                r = image[i, j - 1]
                q = image[i, j + 1]
                c1 += 1

            elif (-67.5 <= ang <= -22.5) or (112.5 <= ang <= 157.5):
                r = image[i - 1, j + 1]
                q = image[i + 1, j - 1]
                c2 += 1

            elif (67.5 <= ang <= 112.5) or (-112.5 <= ang <= -67.5):
                r = image[i - 1, j]
                q = image[i + 1, j]
                c3 += 1

            elif (22.5 <= ang < 67.5) or (-167.5 <= ang <= -112.5):
                r = image[i + 1, j + 1]
                q = image[i - 1, j - 1]
                c4 += 1

            if (image[i, j] >= q) and (image[i, j] >= r):
                suppressed[i, j] = image[i, j]
            else:
                suppressed[i, j] = 0
    return suppressed


def double_thresholding(image, threshold):
    highThreshold = threshold * 0.5
    lowThreshold = highThreshold * 0.5

    M, N = image.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(75)
    strong = np.int32(255)

    strong_i, strong_j = np.where(image >= highThreshold)
    # zeros_i, zeros_j = np.where(image < lowThreshold)
    weak_i, weak_j = np.where(np.logical_and((image <= highThreshold), (image >= lowThreshold)))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)

def hysteresis(image, weak, strong=255):
    M, N = image.shape
    out = image.copy()

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (image[i, j] == weak):
                if np.any(image[i - 1:i + 2, j - 1:j + 2] == strong):
                    out[i, j] = strong
                else:
                    out[i, j] = 0
    return out


img=cv2.imread("lena.jpg",cv2.IMREAD_GRAYSCALE)
cv2.imshow("Input grayscaled image",img)
cv2.waitKey(0)

image = cv2.GaussianBlur(img, (3, 3), 0)
cv2.imshow("Gaussian Convoluted image", image)
cv2.waitKey(0)

horizontal_filter, vertical_filter = derivative()

# convolution using sobel filter
horizontal_convolution = convolution.convolution(image,vertical_filter)
vertical_convolution = convolution.convolution(image,horizontal_filter)

gaussian_filter = gaussian_filter.gaussian()

# again convolution using gaussian filter
horizontal_convolution = convolution.convolution(horizontal_convolution,gaussian_filter)
vertical_convolution = convolution.convolution(vertical_convolution,gaussian_filter)

merged_output = merge(horizontal_convolution, vertical_convolution)

# normalized results
normalized_horizontal_convolution=cv2.normalize(horizontal_convolution, horizontal_convolution,0, 255, cv2.NORM_MINMAX)
normalized_horizontal_convolution = np.round(normalized_horizontal_convolution).astype(np.uint8)

normalized_vertical_convolution=cv2.normalize(vertical_convolution,vertical_convolution, 0, 255, cv2.NORM_MINMAX)
normalized_vertical_convolution = np.round(normalized_vertical_convolution).astype(np.uint8)

normalized_merged_output=cv2.normalize(merged_output,merged_output, 0, 255, cv2.NORM_MINMAX)
normalized_merged_output = np.round(normalized_merged_output).astype(np.uint8)

cv2.imshow("Horizontally convoluted image", normalized_horizontal_convolution)
cv2.imshow("Vertically convoluted image", normalized_vertical_convolution)
cv2.imshow("Gradient Magnitude image", normalized_merged_output)
cv2.waitKey(0)


gradient_angle = np.arctan2(vertical_convolution.copy(), horizontal_convolution.copy())
normalized_gradient_angle=cv2.normalize(gradient_angle,gradient_angle, 0, 255, cv2.NORM_MINMAX)
normalized_gradient_angle = np.round(normalized_gradient_angle).astype(np.uint8)
cv2.imshow("Gradient angle",normalized_gradient_angle)
cv2.waitKey(0)

# merged_convolution = convolution.convolution(merged_output,gaussian_filter)

suppresssed_image = non_maximum_suppression(merged_output,gradient_angle)
normalized_suppressed_image=cv2.normalize(suppresssed_image,suppresssed_image, 0, 255, cv2.NORM_MINMAX)
normalized_suppressed_image = np.round(normalized_suppressed_image).astype(np.uint8)
cv2.imshow("Non maximum suppression",normalized_suppressed_image)
cv2.waitKey(0)

threshold = find_threshold(image=suppresssed_image)

double_threshold_result,weak,strong = double_thresholding(suppresssed_image,threshold)
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