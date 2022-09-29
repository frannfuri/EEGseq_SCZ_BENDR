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
    labels_path = './datasets/labels/SA047_labels.csv'
    #################################

    labels_info = pd.read_csv(labels_path, index_col=0, decimal=',')
    real_SAPS = labels_info['SAPS_global'].values
    pred_SAPS = SAPSg_from_PANSSp_array(labels_info['PANSS_posit'].values)
    mean_2_SAPS = (real_SAPS + pred_SAPS)/2
    median_threshold = np.median(mean_2_SAPS)
    new_col = []
    for v in mean_2_SAPS:
        if v > median_threshold:
            new_col.append(1)
        else:
            new_col.append(0)
    labels_info['real_and_pred_SAPS'] = new_col

    labels_info.to_csv(labels_path)

    a = 0