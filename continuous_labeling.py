import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import min_max_simple_norm

def PANSSp_from_SAPSg(saps_global):
    return 9.3264 +(1.1072*saps_global)

def SAPSg_from_PANSSp(panss_posit):
    return -3.222 + (0.567*panss_posit)

def PANSSp_from_SAPSg_array(saps_global_array):
    panss_posit_array = []
    for saps_global in saps_global_array:
        panss_posit_array.append(PANSSp_from_SAPSg(saps_global))
    return panss_posit_array

def SAPSg_from_PANSSp_array(panss_posit_array):
    saps_global_array = []
    for panss_posit in panss_posit_array:
        saps_global_array.append(SAPSg_from_PANSSp(panss_posit))
    return saps_global_array

# Use this when you dont have the labels binarized
# BINARIZATION METHOD: SAPS FROM PANSS AND AVERAGE WITH REAL MEASURED SAPS
if __name__ == '__main__':
    labels_path = '../BENDR_datasets/labels/SA010_labels.csv'
    #################################

    labels_info = pd.read_csv(labels_path, index_col=0, decimal=',')
    rec_names = labels_info.index.values
    fig, ax = plt.subplots(figsize=(6,5))

    real_SAPS = labels_info['SAPS_global'].values
    PANSS_posit = labels_info['PANSS_posit'].values
    ax.scatter(PANSS_posit, real_SAPS)

    pred_SAPS = SAPSg_from_PANSSp_array(PANSS_posit)
    ax.scatter(PANSS_posit, pred_SAPS, marker='*', c='chocolate')

    x_line = list(range(min(labels_info['PANSS_posit'].values)-1, max(labels_info['PANSS_posit'].values)+2))
    y_line = SAPSg_from_PANSSp_array(x_line)
    ax.plot(x_line, y_line, linestyle='dashed', c='grey')

    mean_2_SAPS = (real_SAPS + pred_SAPS)/2
    ax.scatter(PANSS_posit, mean_2_SAPS, marker='x', c='r', s=40)
    for i, txt in enumerate(rec_names):
        ax.annotate(txt[6:], (PANSS_posit[i], mean_2_SAPS[i]), fontsize=8)
    plt.xlabel('PANSS positive')
    plt.ylabel('SAPS global')
    plt.show(block=False)
    new_col = min_max_simple_norm(mean_2_SAPS)
    labels_info['real_and_pred_SAPS_norm'] = new_col

    ax.set_title('Labeling of {}'.format(labels_path[-16:-11]), fontsize=10)
    labels_info.to_csv(labels_path[:-4]+'.csv')
    plt.show(block=False)
    a = 0