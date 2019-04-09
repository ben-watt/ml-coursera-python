import numpy as np

al = np.array([1,2,3,4,5])
sm = np.array([1,2,0,0,0])

zeros = sm == 0

print(al[zeros])