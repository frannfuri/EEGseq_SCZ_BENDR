import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
    labels_path = '../BENDR_datasets/labels/SA025_labels.csv'
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
    median_threshold = np.median(mean_2_SAPS)
    ax.axhline(median_threshold, linestyle='dashed', linewidth=0.7, c='tomato')
    for i, txt in enumerate(rec_names):
        ax.annotate(txt[6:], (PANSS_posit[i], mean_2_SAPS[i]), fontsize=8)
    plt.xlabel('PANSS positive')
    plt.ylabel('SAPS global')
    plt.show(block=False)
    new_col = []
    for v in mean_2_SAPS:
        if v > median_threshold:
            new_col.append(1)
        else:
            new_col.append(0)
    labels_info['real_and_pred_SAPS'] = new_col

    ax.set_title('Labeling of {}'.format(labels_path[-16:-11]), fontsize=10)
    labels_info.to_csv(labels_path[:-4]+'2.csv')
    plt.show(block=False)
    a = 0