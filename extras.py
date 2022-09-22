import numpy as np

def MAD(data):
    M = np.median(data)
    diff_vector = []
    for x in data:
        diff_vector.append(np.abs(x-M))
    return np.median(np.array(diff_vector))

def robust_z_score_norm(data):
    norm_data = []
    MAD_data = MAD(data)
    for x in data:
        num_x = 0.6745*(x-np.median(data))
        norm_x = num_x/MAD_data
        norm_data.append(norm_x)
    return np.array(norm_data)