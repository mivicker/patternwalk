import numpy as np

pows = 3**np.arange(10)

print(pows + pows.reshape(-1,1))
