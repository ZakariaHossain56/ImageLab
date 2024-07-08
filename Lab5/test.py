import cv2
import numpy as np
import json


def get_descriptor(border_image, binary_image):
    area = np.count_nonzero(binary_image)
    perimeter = np.count_nonzero(border_image)

    pixels = []

    # Get the dimensions of the image
    height, width = border_image.shape

    # Iterate over every pixel in the image
    for y in range(height):
        for x in range(width):
            if binary_image[y, x] > 0:  # Check if the pixel is part of the foreground
                pixels.append((y, x))  # Append the coordinates to the list

    x_min = y_min = float('inf')
    x_max = y_max = float('-inf')

    for y, x in pixels:
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y

    # Calculate horizontal and vertical extents
    horizontal_extent = x_max - x_min
    vertical_extent = y_max - y_min

    # Determine the maximum diameter
    max_diameter = max(horizontal_extent, vertical_extent)
    print(max_diameter)

    compactness = (perimeter * perimeter) / area
    print(compactness)

    form_factor = 4 * np.pi * area / (perimeter * perimeter)
    print(form_factor)

    if max_diameter > 0:
        roundness = (4 * area) / (np.pi * max_diameter * max_diameter)
    else:
        roundness = None

    print(roundness)

    if area == 0:
        return None, None, None, None, None, None

    return area, perimeter, max_diameter, compactness, form_factor, roundness


def customized_thresold(image, thresold):
    binary_image = np.zeros_like(image, dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > thresold:
                binary_image[i, j] = 255
            else:
                binary_image[i, j] = 0

    return binary_image


def image_process(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', image)

    binary_image = customized_thresold(image, 128)
    cv2.imshow('binary image', binary_image)

    kernel = np.ones((3, 3), np.uint8)
    eroted_image = cv2.erode(binary_image, kernel)

    cv2.imshow('erotion', eroted_image)

    border_image = cv2.subtract(binary_image, eroted_image)
    cv2.imshow('border', border_image)

    area, perimeter, max_diameter, compactness, form_factor, roundness = get_descriptor(border_image, binary_image)

    return {
        'area': area,
        'perimeter': perimeter,
        'max_diameter': max_diameter,
        'compactness': compactness,
        'form factor': form_factor,
        'roundness': roundness
    }


def match_descriptors(test_image, train_descriptors):
    # Simple Euclidean distance based matching
    min_distance = float('inf')
    best_match = None

    for shape, descriptors in train_descriptors.items():
        distance = 0
        for key in test_image:
            distance += (test_image[key] - descriptors[key]) ** 2
        distance = np.sqrt(distance)

        if distance < min_distance:
            min_distance = distance
            best_match = shape

    return best_match


train_descriptors = {
    'circle': image_process('c1.jpg'),
    'square': image_process('p3.jpg'),
    'trainagle': image_process('t1.jpg')
}

test_image = image_process('c2.jpg')

# Match test descriptors with train descriptors
best_match = match_descriptors(test_image, train_descriptors)

print(f"The test image matches with: {best_match}")

cv2.waitKey(0)
cv2.destroyAllWindows()