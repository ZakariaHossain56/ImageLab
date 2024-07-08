import cv2
import gaussian_filter
import numpy as np

# GRAYSCALE

# img = cv2.imread("lena.jpg",cv2.IMREAD_GRAYSCALE)
# cv2.imshow("Input image",img)
# cv2.waitKey(0)
#
# kernel = gaussian_filter.gaussian(1,1)
# kernel_height,kernel_width = kernel.shape
#
# centerx = kernel_width//2
# centery = kernel_height//2
#
# pad_top = int(centerx)
# pad_bottom = int(kernel_height-centerx-1)
# pad_left = int(centery)
# pad_right = int(kernel_width-centery-1)
#
#
# bordered_image = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)
# cv2.imshow("Bordered image", bordered_image)
# cv2.waitKey(0)
# output = np.zeros_like(bordered_image)
# padded_height, padded_width = bordered_image.shape
#
# for x in range(centerx,padded_height-(kernel_height-(centerx+1))):
#     for y in range(centery,padded_width-(kernel_width-(centery+1))):
#         image_startx = x-centerx
#         image_starty= y-centery
#         sum=0
#         n=kernel_width//2
#         for i in range(-n,n+1):
#             for j in range(-n,n+1):
#                 relative_kernelx=i+1
#                 relative_kernely = j+1
#                 relative_imagex = n-i
#                 relative_imagey = n-j
#                 actual_imagex = image_startx+relative_imagex
#                 actual_imagey = image_starty+relative_imagey
#                 image_value = bordered_image[actual_imagex][actual_imagey]
#                 kernel_value = kernel[relative_kernelx][relative_kernely]
#                 sum+=(image_value*kernel_value)
#             output[x,y]=sum
#
# cv2.normalize(output,output,0,255,cv2.NORM_MINMAX)
# output = np.round(output).astype(np.uint8)
# cv2.imshow("Final output", output)
#
#
# cv2.waitKey()
# cv2.destroyAllWindows()



# RGB
# image = cv2.imread("lena.jpg",cv2.IMREAD_COLOR)
# cv2.imshow("Color image", image)
# cv2.waitKey(0)
#
# blue,green,red = cv2.split(image)
# cv2.imshow("Blue channel", blue)
# cv2.imshow("Green channel",green)
# cv2.imshow("Red channel", red)
# merged = cv2.merge([blue,green,red])
# cv2.imshow("Merged", merged)
#
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# HSV

image = cv2.imread("lena.jpg",cv2.IMREAD_COLOR)
cv2.imshow("Input image", image)
cv2.waitKey(0)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV image", hsv)
color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imshow("Color image", color)



cv2.waitKey(0)
cv2.destroyAllWindows()
