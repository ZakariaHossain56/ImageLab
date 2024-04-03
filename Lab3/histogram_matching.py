import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def generateGaussianDistribution(miu,sigma):
    gaussian=np.zeros(256)
    constant=1/(math.sqrt(2*math.pi*sigma**2))
    for x in range(256):
        calc=-((x-miu)**2)/(2*sigma**2)
        gaussian[x]=constant*math.exp(calc)
    return gaussian

def draw_plot(plot,n,title):
    plt.figure(n)
    plt.plot(plot)
    plt.title(title)
    plt.show()

print("Enter the value of miu and sigma of first gaussian distribution : ")
values = input().split()
miu1, sigma1 = values
miu1 = float(miu1)
sigma1 = float(sigma1)
# print(miu1)
# print(sigma1)
print("Enter the value of miu and sigma of second gaussian distribution : ")
values = input().split()
miu2, sigma2 = values
miu2 = float(miu2)
sigma2 = float(sigma2)
# print(miu2)
# print(sigma2)
gaussian1=generateGaussianDistribution(miu1,sigma1)
# print(gaussian1)
gaussian2=generateGaussianDistribution(miu2,sigma2)
# print(gaussian2)
double_gaussian=np.zeros(256)
for i in range(len(gaussian1)):
    double_gaussian[i] = gaussian1[i] + gaussian2[i]
# draw_plot(double_gaussian,1,"Target histogram")

#target pdf
target_pdf=double_gaussian.copy()
for x in range(256):
    target_pdf[x]=double_gaussian[x]/sum(double_gaussian)
# draw_plot(target_pdf,2,"Target pdf")

#target cdf
target_cdf=target_pdf.copy()
for x in range(1,256):
    target_cdf[x]=target_cdf[x-1]+target_pdf[x]
# draw_plot(target_cdf,3,"Target cdf")



img = cv2.imread("histogram.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Input image",img)

#cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
input_histr = cv2.calcHist([img],[0],None,[256],[0,256])
draw_plot(input_histr,1,"Input image histogram")


histogram=np.zeros(256)
for i in range(256):
    histogram[i]=input_histr[i][0]

#input pdf
pdf=histogram.copy()
for x in range(256):
    pdf[x]=histogram[x]/sum(histogram)
draw_plot(pdf,2,"Input image pdf")


#input cdf
cdf=pdf.copy()
for x in range(1,256):
    cdf[x]=cdf[x-1]+pdf[x]
draw_plot(cdf,3,"Input image cdf")


mapping=cdf.copy()
for x in range(256):
    input_cdf=cdf[x]
    minimum_diff = 256
    pixel=0
    for y in range(256):
        if(abs(target_cdf[y]-input_cdf) < minimum_diff):
            minimum_diff=abs(target_cdf[y]-input_cdf)
            pixel=y
    mapping[x]=pixel

output=img.copy()
height, width = img.shape
# print(height)
# print(width)
for x in range(height):
    for y in range(width):
        img_intensity=img[x,y]
        output[x,y]=mapping[img_intensity]

cv2.imshow("Output image",output)


output_histr = cv2.calcHist([output],[0],None,[256],[0,256])
draw_plot(output_histr,4,"Output image histogram")


histogram=np.zeros(256)
for i in range(256):
    histogram[i]=output_histr[i][0]

#output pdf
output_pdf=histogram.copy()
for x in range(256):
    output_pdf[x]=histogram[x]/sum(histogram)
draw_plot(output_pdf,5,"Output image pdf")


#output cdf
output_cdf=output_pdf.copy()
for x in range(1,256):
    output_cdf[x]=output_cdf[x-1]+output_pdf[x]
draw_plot(output_cdf,6,"Output image cdf")


cv2.waitKey(0)
cv2.destroyAllWindows()