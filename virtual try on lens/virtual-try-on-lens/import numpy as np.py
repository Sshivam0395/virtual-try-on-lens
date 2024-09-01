import numpy as np
import sys
import time

SIZE = 1000000

a1 = np.arange(SIZE)
a2 = np.arange(SIZE)
start = time.time()

result = a1 + a2

print("numpy took" , (time.time()-start)*1000)