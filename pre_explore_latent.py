# Code to execute in the cluster to obtain de .pt files
import torch
import yaml
import mne
from architectures import ConvEncoderBENDR_from_scratch
from datasets import charge_dataset, recInfoDataset
from collections import OrderedDict

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def flat_latent_representation(model, dataset):
    list_sorted_rec_names = []
    targets_sorted = []
    outputs_sorted = []
    first_outputs = []
    model.eval()

    for x, y, rec_name in dataset:
        output_ = model(x.unsqueeze(0))   # size [1, 512, 107]
        new_dim = torch.zeros(output_.shape[2])
        for chn in range(output_.shape[1]):
            id = torch.argmax(output_[0][chn,:])
            new_dim[id] += 1
        output = torch.mean(output_[0], dim=0) # size [107]
        output = torch.cat((output, new_dim)) # size [214]

        list_sorted_rec_names.append(rec_name)
        targets_sorted.append(y)
        outputs_sorted.append(output)
        first_outputs.append(output_)
    return outputs_sorted, targets_sorted, list_sorted_rec_names, first_outputs

if __name__ == '__main__':
    dataset_directory = '../BENDR_datasets/decomp_study_SA047'
    ws_path = './linear-rslts_avp_pAug_bw_dp0307_f1f_th04_len40ov30_/best_model_f0_h_scz_study_lr0.0001bs8.pt'
    ################################

    dataset_name = dataset_directory.split('/')[-1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('---Using ' + str(device) + 'device---')

    model = ConvEncoderBENDR_from_scratch(20, 512, projection_head=False, dropout=0)
    ws = torch.load(ws_path, map_location=device)
    ws2 = OrderedDict()

    for k, v in ws.items():
        if k[:8] == 'encoder.':
            ws2[k[8:]] = v

    model.load_state_dict(ws2)
    model.eval()

    with open(dataset_directory + '/info.yml') as infile:
        data_settings = yaml.load(infile, Loader=yaml.FullLoader)
    w_len = data_settings['tlen']

    array_epochs_all_records, sorted_record_names = charge_dataset(directory=dataset_directory,
                                                                   tlen=data_settings['tlen'],
                                                                   overlap=data_settings['overlap_len'],
                                                                   data_max=data_settings['data_max'],
                                                                   data_min=data_settings['data_min'],
                                                                   chns_consider=data_settings['chns_to_consider'],
                                                                   labels_path=data_settings['labels_path'],
                                                                   target_f=data_settings['target_feature'],
                                                                   apply_winsor=data_settings['apply_winsorising'],
                                                                   new_sfreq=256)

    # Reorder the Xs and Ys data
    is_first_rec = True
    for rec in array_epochs_all_records:
        if is_first_rec:
            all_X = rec[0]
            all_y = rec[1]
            is_first_rec = False
        else:
            all_X = torch.cat((all_X, rec[0]), dim=0)
            all_y = torch.cat((all_y, rec[1]), dim=0)

    dataset = recInfoDataset(all_X, all_y, sorted_record_names)

    os, ts, rs, first_os = flat_latent_representation(model, dataset)
    torch.save(os, './{}_{}_flat_output.pt'.format(dataset_name, w_len))
    torch.save(rs, './results/{}_{}_rec.pt'.format(dataset_name, w_len))
    torch.save(ts, './results/{}_{}_target.pt'.format(dataset_name, w_len))
    torch.save(first_os, './results/{}_{}_output.pt'.format(dataset_name, w_len))