import numpy as np
def mean(size):
    row=size
    column=size
    centerx=row//2
    centery=column//2
    kernel=np.ones((size, size), dtype=float) / (size**2)
    normalized_kernel = kernel * (size**2)
    print("Mean kernel")
    # print(kernel)
    print(normalized_kernel)
    return kernel

# mean(3)