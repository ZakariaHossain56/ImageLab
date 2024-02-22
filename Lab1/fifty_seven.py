import numpy as np
import cv2

#img = cv2.imread('box.jpg',0)
img = cv2.imread('box.jpg',cv2.IMREAD_GRAYSCALE)

img = cv2.copyMakeBorder(src=img, top=2, bottom=2, left=2, right=2,borderType= cv2.BORDER_CONSTANT)
cv2.imshow('grayscaled image',img)

kernel =(1/273) * np.array([[1, 4, 7, 4,1],
                            [4,16,26,16,4],
                            [7,26,41,26,7],
                            [4,16,26,16,4],
                            [1, 4, 7, 4,1]])

print( img.shape[0] )
print( img.shape[1] )
out=img.copy()

n = int( kernel.shape[0]/2 )

for x in range( n, img.shape[0]-n ):
    for y in range( n, img.shape[1]-n ):

        res = 0
        for j in range( -n, n+1 ):
            for i in range( -n, n+1 ):
                kernel_value = kernel.item(i,j)
                image_value = img.item(x-i,y-j)
                
                res += (kernel_value * image_value)
        
        out[x,y] = res
        #out.itemset((i,j),a+122) #255-a)

print(out)
cv2.normalize(out,out, 0, 255, cv2.NORM_MINMAX)
out = np.round(out).astype(np.uint8)

print(out)
cv2.imshow('normalised output image',out)

cv2.waitKey(0)
cv2.destroyAllWindows()

