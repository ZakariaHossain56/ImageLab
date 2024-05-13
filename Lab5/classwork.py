import math

import cv2
import numpy as np



def calculate_descriptors(image,i):
    image = image.copy()
    #_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # area
    area = np.count_nonzero(image)

    # perimeter
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=1)
    border = image - eroded
    perimeter = np.count_nonzero(border)

    cv2.imshow('Border'+str(i), border)
    cv2.imshow('Input image'+str(i), image)

    # max diameter
    height, width = image.shape
    min_x=width
    min_y=width
    max_x=0
    max_y=0
    for i in range(height):
        for j in range(width):
            if(image[i,j] > 0):
                if(i<min_x):
                    min_x = i
                    min_y = j
                if(i>max_x):
                    max_y = i
                    max_y = j

    max_diameter = math.sqrt((max_x-min_x)**2 + (max_y-min_y)**2)

    compactness = perimeter**2/area
    # print(area)
    form_factor = (4*math.pi*area)/(perimeter**2)
    roundness = (4*area)/(math.pi*(max_diameter**2))
    l = []
    l.append(compactness)
    l.append(form_factor)
    l.append(roundness)
    # print(l)
    descriptor_list = []
    descriptor_list.append(l)
    # l.clear()
    # print(descriptor_list)
    return descriptor_list


def start():

    image_name = ['c1.jpg','t1.jpg','p1.png','c2.jpg','t2.jpg','p2.png']

    descr_list = []
    for i in range(len(image_name)):
        img = cv2.imread(image_name[i], 0)
        descriptor = calculate_descriptors(img,i)
        descr_list.append(descriptor)
    print(descr_list)
    return descr_list

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

start()