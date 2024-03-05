import numpy as np
import math
def log(sigma=1.4):
    row = math.floor(7 * sigma)
    column = math.floor(7 * sigma)
    if (row % 2 == 0):
        row += 1
    if (column % 2 == 0):
        column += 1
    # print(row)
    # print(column)
    centerx = row // 2
    centery = column // 2
    # print(centerx)
    # print(centery)
    constant=-1/(math.pi*sigma**4)
    kernel=np.zeros((row,column))
    for x in range(row):
        for y in range(column):
            calc1=(((x-centerx)**2)+((y-centery)**2))/(2*sigma**2)
            kernel[x][y]=constant*(1-calc1)*math.exp(-calc1)

    normalized_kernel = kernel / np.min(np.abs(kernel))
    normalized_kernel=np.round(normalized_kernel).astype(int)

    print(kernel)
    print(normalized_kernel)
    return kernel


# log(1.4)