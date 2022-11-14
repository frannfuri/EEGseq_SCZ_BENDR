import matplotlib.pyplot as plt
import torch
import csv
import yaml
from datasets import charge_dataset, recInfoDataset
from architectures import LinearHeadBENDR_from_scratch
from utils import all_same
import numpy as np

if __name__ == '__main__':
    # PARAMETERS
    valid_sets_path = '../datasets/valid_sets/val_sets_SA000.csv'
    data_path = '../datasets/h_scz_study'

    #################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('---Using ' + str(device) + 'device---')

    with open(data_path + '/info.yml') as infile:
        data_settings = yaml.load(infile, Loader=yaml.FullLoader)
    with open('./{}'.format(valid_sets_path), newline='') as f:
        reader = csv.reader(f)
        valid_sets = list(reader)

    array_epochs_all_records, sorted_record_names = charge_dataset(directory=data_path,
                                                                   tlen=data_settings['tlen'],
                                                                   overlap=data_settings['overlap_len'],
                                                                   data_max=data_settings['data_max'],
                                                                   data_min=data_settings['data_min'],
                                                                   chns_consider=data_settings['chns_to_consider'],
                                                                   labels_path='../' + data_settings['labels_path'],
                                                                   target_f=data_settings['target_feature'],
                                                                   apply_winsor=data_settings['apply_winsorising'])

    # Reorder the Xs and Ys data
    is_first_rec = True

    #from utils import plot_cm_valid_per_record
    #cm = plot_cm_valid_per_record(array_epochs_all_records, sorted_record_names, data_settings['tlen'],valid_sets_path, fold, m_path, 0.6)


    ####### CUSTOM PARAMETERS #######
    f___ = 0
    model_path = '../rslts_pAug_avpf_vpr_th06_dp0307_bw_len40ov30_/best_model_f{}_h_scz_study_lr0.0001bs16.pt'.format(f___)
    #################################

    fold = f___  ### IT MUST COINCIDE WITH THE FOLD IN THE model_path !!!!!!
    for rec in array_epochs_all_records:
        if is_first_rec:
            all_X = rec[0]
            all_y = rec[1]
            is_first_rec = False
        else:
            all_X = torch.cat((all_X, rec[0]), dim=0)
            all_y = torch.cat((all_y, rec[1]), dim=0)
    print('-------------------------------------------')
    num_cls = data_settings['num_cls']
    samples_tlen = data_settings['tlen']

    valid_set = valid_sets[fold]
    valid_ids = [i for i in range(len(sorted_record_names)) if (sorted_record_names[i][:11] in valid_sets[fold])]
    valid_dataset = recInfoDataset(all_X[valid_ids], all_y[valid_ids], [sorted_record_names[i] for i in valid_ids])

    valid_record_names = np.unique(valid_dataset[:][2])
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)

    model = LinearHeadBENDR_from_scratch(1, samples_len=samples_tlen * 256, n_chn=20,
                                         encoder_h=512, projection_head=False,
                                         # DROPOUTS
                                         enc_do=0.3, feat_do=0.7,  # enc_do=0.1, feat_do=0.4,
                                         pool_length=4,
                                         # MASKS LENGHTS
                                         mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.05, mask_c_span=0.1,
                                         classifier_layers=1, return_features=False,
                                         # IF USE MASK OR NOT
                                         not_use_mask_train=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Predictions
    all_recs_predictions = []
    all_recs_targets = []
    for rec_i in valid_record_names:
        rec_predictions = []
        rec_targets = []
        for x, y, rec_name in validloader:
            if rec_name[0] == rec_i:
                output = model(x)
                prepred = torch.sigmoid(output)
                pred = (prepred >= 0.5).long().squeeze()
                rec_predictions.append(pred.item())
                rec_targets.append(y.item())
        all_recs_predictions.append(rec_predictions)
        assert all_same(rec_targets)
        all_recs_targets.append(rec_targets)

    fig, axs = plt.subplots(len(valid_record_names), 1, figsize=(10,6))
    for i in range(len(valid_record_names)):
        marker_color = 'blue' if all_recs_targets[i][0] == 0 else 'red'
        assert all_recs_targets[i][0] == 1 or all_recs_targets[i][0] == 0
        axs[i].scatter(list(range(len(all_recs_predictions[i]))), all_recs_predictions[i],
                       label='{} (target={})'.format(valid_record_names[i], all_recs_targets[i][0]),
                       c=marker_color, s=20)
        axs[i].set_title('Sample predictions of best model CV it. n°{}'.format(fold + 1), fontsize=8)
        axs[i].legend(fontsize=8)
        axs[i].set_ylim((-0.5, 1.5))
    plt.tight_layout()
    plt.show(block=False)
    a = 0