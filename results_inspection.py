import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    path = './results_new'
    name= 'decomp_study_SA047_lr1e-05bs8_delta_PANSS_posit_bin'

    a = pd.read_csv('{}/train_log_f0_{}.csv'.format(path, name), index_col=0)
    accs = []
    recalls = []
    f1scores=[]
