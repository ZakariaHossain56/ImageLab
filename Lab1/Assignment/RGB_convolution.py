import math

import cv2
import numpy as np

import grayscale_convolution
import HSV_convolution
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
def RGB_convolution(kernel_with_type,center):
    img = cv2.imread('lena.jpg',cv2.IMREAD_COLOR)
    cv2.imshow('Input RGB image', img)
    cv2.waitKey(0)
    blue_channel, green_channel, red_channel = cv2.split(img)
    print("RGB channel")
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

        # Convolution of red channel
        red_horizontal_convolution = grayscale_convolution.convolution(red_channel, horizontal_kernel, center)
        # cv2.imshow("Horizontal convolution", red_horizontal_convolution)
        # cv2.waitKey(0)
        red_vertical_convolution = grayscale_convolution.convolution(red_channel, vertical_kernel, center)
        # cv2.imshow("Vertical convolution", red_vertical_convolution)
        output_red = sobelColvolution(red_channel,red_channel.shape,red_horizontal_convolution,red_vertical_convolution)
        cv2.normalize(output_red, output_red, 0, 255, cv2.NORM_MINMAX)
        output_red = np.round(output_red).astype(np.uint8)
        cv2.imshow("Red convoluted image",output_red)

        # Convolution of green channel
        green_horizontal_convolution = grayscale_convolution.convolution(green_channel, horizontal_kernel, center)
        # cv2.imshow("Horizontal convolution", red_horizontal_convolution)
        # cv2.waitKey(0)
        green_vertical_convolution = grayscale_convolution.convolution(green_channel, vertical_kernel, center)
        # cv2.imshow("Vertical convolution", red_vertical_convolution)
        output_green = sobelColvolution(green_channel, green_channel.shape, green_horizontal_convolution, green_vertical_convolution)
        cv2.normalize(output_green, output_green, 0, 255, cv2.NORM_MINMAX)
        output_green = np.round(output_green).astype(np.uint8)
        cv2.imshow("Green convoluted image",output_green)

        # Convolution of blue channel
        blue_horizontal_convolution = grayscale_convolution.convolution(blue_channel, horizontal_kernel, center)
        # cv2.imshow("Horizontal convolution", red_horizontal_convolution)
        blue_vertical_convolution = grayscale_convolution.convolution(blue_channel, vertical_kernel, center)
        # cv2.imshow("Vertical convolution", red_vertical_convolution)
        output_blue = sobelColvolution(blue_channel, blue_channel.shape, blue_horizontal_convolution, blue_vertical_convolution)
        cv2.normalize(output_blue, output_blue, 0, 255, cv2.NORM_MINMAX)
        output_blue = np.round(output_blue).astype(np.uint8)
        cv2.imshow("Blue convoluted image",output_blue)

        cv2.waitKey(0)

        output_RGB_image = cv2.merge([output_blue, output_green, output_red])
        cv2.imshow("RGB Convoluted image", output_RGB_image)
        cv2.waitKey(0)
        hsv_to_rgb_image = HSV_convolution.HSV_convolution(kernel_with_type, center)
        difference = cv2.absdiff(output_RGB_image, hsv_to_rgb_image)
        cv2.waitKey(0)
        cv2.imshow("Difference", difference)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # Red channel convolution
        red_convolution = grayscale_convolution.convolution(red_channel, kernel, center)
        cv2.normalize(red_convolution, red_convolution, 0, 255, cv2.NORM_MINMAX)
        output_red = np.round(red_convolution).astype(np.uint8)
        cv2.imshow("Red convoluted image", output_red)

        # Green channel convolution
        green_convolution = grayscale_convolution.convolution(green_channel, kernel, center)
        cv2.normalize(green_convolution, green_convolution, 0, 255, cv2.NORM_MINMAX)
        output_green = np.round(green_convolution).astype(np.uint8)
        cv2.imshow("Green convoluted image", output_green)

        # blue channel convolution
        blue_convolution = grayscale_convolution.convolution(blue_channel, kernel, center)
        cv2.normalize(blue_convolution, blue_convolution, 0, 255, cv2.NORM_MINMAX)
        output_blue = np.round(blue_convolution).astype(np.uint8)
        cv2.imshow("Blue convoluted image", output_blue)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        output_RGB_image = cv2.merge([output_blue, output_green, output_red])
        cv2.imshow("RGB Convoluted image", output_RGB_image)
        cv2.waitKey(0)
        hsv_to_rgb_image = HSV_convolution.HSV_convolution(kernel_with_type, center)
        print(output_RGB_image.shape)
        print(hsv_to_rgb_image.shape)
        cv2.waitKey(0)
        output_RGB_image = cv2.resize(output_RGB_image, (hsv_to_rgb_image.shape[1], hsv_to_rgb_image.shape[0]))

        print(output_RGB_image.shape)
        print(hsv_to_rgb_image.shape)
        difference = cv2.absdiff(output_RGB_image, hsv_to_rgb_image)

        cv2.imshow("Difference",difference)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



