from matplotlib import pyplot as plt
import cv2
import numpy as np
def show_plot(data,title):
    plt.figure()
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("No of pixels")
    plt.plot(data)
    plt.xlim([0, 256])
    plt.show()

image = cv2.imread("histogram.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Input image", image)
cv2.waitKey(0)

row,column = image.shape


hist = cv2.calcHist([image], [0], None, [256], [0,256])
show_plot(hist,title="Input image histogram")

histogram = []
for i in range(256):
    histogram.append(hist[i][0])
show_plot(histogram, "1D")

    
pdf = hist.copy()
for i in range(len(hist)):
    pdf[i]=hist[i]/(row*column)

cdf = pdf.copy()
for i in range(1,len(pdf)):
    cdf[i] = pdf[i]+cdf[i-1]
show_plot(cdf, "CDF")

mapp = cdf.copy()
for i in range(len(cdf)):
    mapp[i] = np.round(cdf[i]*255).astype(np.uint8)
    
output = image.copy()
for i in range(row):
    for j in range(column):
        value = image[i,j]
        output[i,j] = mapp[value]
cv2.imshow("Output", output)
        




cv2.waitKey(0)
cv2.destroyAllWindows()