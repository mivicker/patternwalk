from itertools import combinations
import numpy as np
from scipy.spatial.distance import cosine as cos_sim
from sklearn.preprocessing import normalize
from walk import calc_best_heading, partitions

def normalized(a, axis=1, order=2):
    return a / a.sum()


new = normalized(np.array([23, 45, 3, 23, 56, 90]).reshape(1, -1))


print(np.floor(new * 8))
