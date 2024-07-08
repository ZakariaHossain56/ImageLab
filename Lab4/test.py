import cv2
import numpy as np
import math
def notch_reject_filter(img):
    indices = ((272, 256), (262, 261))
    radius = (5, 5)
    h,w = img.shape
    notch = np.ones(img.shape)

    print(notch.shape)
    count = 0
    for u in range(h):
        for v in range(w):

            for i in range(len(indices)):
                uk1 = indices[i][1]
                vk1 = indices[i][0]

                uk2 = h // 2 - (uk1 - h // 2)
                vk2 = w // 2 - (vk1 - w // 2)

                dist1 = math.sqrt((u - uk1) ** 2 + (v - vk1) ** 2)
                dist2 = math.sqrt((u - uk2) ** 2 + (v - vk2) ** 2)

                # uk1 -= h//2
                # vk1 -= w//2
                # dist1 = math.sqrt( (u - h/2 - uk1) ** 2 + (v -w/2 - vk1)**2 )
                # dist2 = math.sqrt( (u - h/2 + uk1) ** 2 + (v -w/2 + vk1)**2 )

                if (dist1 <= radius[i] or dist2 <= radius[i]):
                    count += 1
                    notch[u, v] = 0
                    break

    print(count)
    # plt.imshow(notch)
    return notch



img = cv2.imread("two_noise.jpeg",cv2.IMREAD_GRAYSCALE)
notch = notch_reject_filter(img)
cv2.imshow("Notch", notch)
cv2.waitKey(0)


cv2.waitKey(0)
cv2.destroyAllWindows()

