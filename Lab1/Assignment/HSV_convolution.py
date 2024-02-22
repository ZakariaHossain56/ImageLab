import math

import cv2
import numpy as np

import grayscale_convolution
def sobelColvolution(img,size,hor,ver):
    height, width = size
    output = np.zeros_like(img, dtype='float32')

    for x in range(0, height):
        for y in range(0, width):
            dx = hor[x, y]
            dy = ver[x, y]

            result = math.sqrt(dx ** 2 + dy ** 2)
            output[x, y] = result
    return output
def HSV_convolution(kernel_with_type,center):
    img = cv2.imread('lena.jpg',cv2.IMREAD_COLOR)
    cv2.imshow('Input RGB image', img)

    cv2.waitKey(0)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_channel, saturation_channel, value_channel = cv2.split(hsv_image)
    print("HSV channel")
    # cv2.imshow('Original Image', img)
    # cv2.imshow('Blue Channel', blue_channel)
    # cv2.imshow('Green Channel', green_channel)
    # cv2.imshow('Red Channel', red_channel)
    kernel, type = kernel_with_type
    if type == "sobel":
        horizontal_kernel, vertical_kernel = kernel
        # print("Horizontal sobel filter")
        # print(horizontal_kernel)
        # print("Vertical sobel filter")
        # print(vertical_kernel)

        # Convolution of hue channel
        hue_horizontal_convolution = grayscale_convolution.convolution(hue_channel, horizontal_kernel, center)
        # cv2.imshow("Horizontal convolution", red_horizontal_convolution)
        # cv2.waitKey(0)
        hue_vertical_convolution = grayscale_convolution.convolution(hue_channel, vertical_kernel, center)
        # cv2.imshow("Vertical convolution", red_vertical_convolution)
        output_hue = sobelColvolution(hue_channel,hue_channel.shape,hue_horizontal_convolution,hue_vertical_convolution)
        cv2.normalize(output_hue, output_hue, 0, 255, cv2.NORM_MINMAX)
        output_hue = np.round(output_hue).astype(np.uint8)
        cv2.imshow("Hue convoluted image",output_hue)

        # Convolution of saturation channel
        saturation_horizontal_convolution = grayscale_convolution.convolution(saturation_channel, horizontal_kernel, center)
        # cv2.imshow("Horizontal convolution", red_horizontal_convolution)
        # cv2.waitKey(0)
        saturation_vertical_convolution = grayscale_convolution.convolution(saturation_channel, vertical_kernel, center)
        # cv2.imshow("Vertical convolution", red_vertical_convolution)
        output_saturation = sobelColvolution(saturation_channel, saturation_channel.shape, saturation_horizontal_convolution, saturation_vertical_convolution)
        cv2.normalize(output_saturation, output_saturation, 0, 255, cv2.NORM_MINMAX)
        output_saturation = np.round(output_saturation).astype(np.uint8)
        cv2.imshow("Saturation convoluted image",output_saturation)

        # Convolution of value channel
        value_horizontal_convolution = grayscale_convolution.convolution(value_channel, horizontal_kernel, center)
        # cv2.imshow("Horizontal convolution", red_horizontal_convolution)
        value_vertical_convolution = grayscale_convolution.convolution(value_channel, vertical_kernel, center)
        # cv2.imshow("Vertical convolution", red_vertical_convolution)
        output_value = sobelColvolution(value_channel, value_channel.shape, value_horizontal_convolution, value_horizontal_convolution)
        cv2.normalize(output_value, output_value, 0, 255, cv2.NORM_MINMAX)
        output_value = np.round(output_value).astype(np.uint8)
        cv2.imshow("Value convoluted image",output_value)

        cv2.waitKey(0)

        output_HSV_image = cv2.merge([hue_channel, saturation_channel, value_channel])

        cv2.imshow("HSV Convoluted image", output_HSV_image)
        cv2.waitKey(0)
        rgb_image = cv2.cvtColor(output_HSV_image, cv2.COLOR_HSV2BGR)
        cv2.imshow("HSV to RGB image",rgb_image)
        return rgb_image
        # cv2.destroyAllWindows()
    else:
        # Hue channel convolution
        hue_convolution = grayscale_convolution.convolution(hue_channel, kernel, center)
        cv2.normalize(hue_convolution, hue_convolution, 0, 255, cv2.NORM_MINMAX)
        output_hue = np.round(hue_convolution).astype(np.uint8)
        cv2.imshow("Hue convoluted image", output_hue)

        # Saturation channel convolution
        saturation_convolution = grayscale_convolution.convolution(saturation_channel, kernel, center)
        cv2.normalize(saturation_convolution, saturation_convolution, 0, 255, cv2.NORM_MINMAX)
        output_saturation = np.round(saturation_convolution).astype(np.uint8)
        cv2.imshow("Saturation convoluted image", output_saturation)

        # Value channel convolution
        value_convolution = grayscale_convolution.convolution(value_channel, kernel, center)
        cv2.normalize(value_convolution, value_convolution, 0, 255, cv2.NORM_MINMAX)
        output_value = np.round(value_convolution).astype(np.uint8)
        cv2.imshow("Value convoluted image", output_value)

        cv2.waitKey(0)

        output_HSV_image = cv2.merge([hue_channel, saturation_channel, value_channel])

        cv2.imshow("HSV Convoluted image", output_HSV_image)
        cv2.waitKey(0)
        rgb_image = cv2.cvtColor(output_HSV_image, cv2.COLOR_HSV2BGR)
        cv2.imshow("HSV to RGB image", rgb_image)
        return rgb_image

