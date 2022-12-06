import matplotlib.pyplot as plt
import torch
import csv
import yaml
import sys
sys.path.append('..')
from datasets import charge_dataset, recInfoDataset
from architectures import LinearHeadBENDR_from_scratch
from utils import all_same
import numpy as np
from mlxtend.plotting import plot_confusion_matrix
from utils import plot_cm_valid_per_record

if __name__ == '__main__':
    # PARAMETERS
    data_path = '../../BENDR_datasets/h_scz_study'
    n_folds = 6
    model_path1 = '../linear-rslts_avp_pAug_bw_vpr_dp0307_f1f_len40ov30_/best_model_f'
    model_path2 = '_h_scz_study_lr0.0003bs16.pt'
    th = 0.4
    n_divisions_segments = 4

    #################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('---Using ' + str(device) + 'device---')

    with open(data_path + '/info.yml') as infile:
        data_settings = yaml.load(infile, Loader=yaml.FullLoader)
    valid_sets_path = data_settings['valid_sets_path']

    with open('../'+valid_sets_path, newline='') as f:
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

    all_folds_cms = torch.zeros(num_cls, num_cls)
    for f in range(n_folds):
        valid_set = valid_sets[f]
        valid_ids = [i for i in range(len(sorted_record_names)) if (sorted_record_names[i][:11] in valid_sets[f])]
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
        model_path = '{}{}{}'.format(model_path1, f, model_path2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        confusion_matrix = torch.zeros(num_cls, num_cls)
        model.to(device)

        valid_name_analyze = '0'
        valids_preds_ = []
        valid_targets_ = []
        valid_names_ = []
        with torch.no_grad():
            for i, (inputs, classes, rec_info) in enumerate(validloader):
                inputs = inputs.to(device)
                classes = classes.to(device)
                outputs = model(inputs)
                prepreds = torch.sigmoid(outputs)
                preds = (prepreds >= th).long().squeeze()

                if valid_name_analyze != rec_info[0]:
                    if valid_name_analyze != '0':
                        valids_preds_.append(one_record_valid_predictions)
                        valid_targets_.append(one_record_valid_targets)
                    one_record_valid_predictions = []
                    one_record_valid_targets = []
                    valid_names_.append(rec_info[0])
                    valid_name_analyze = rec_info[0]
                assert (preds.item() == 1 or preds.item() == 0) and (  # sanity check
                        classes.item() == 1 or classes.item() == 0)
                one_record_valid_predictions.append(preds.item())
                one_record_valid_targets.append(classes.item())
        valids_preds_.append(one_record_valid_predictions)
        valid_targets_.append(one_record_valid_targets)

        valid_preds_subseg_conclusion = []
        valid_targets_subseg_conclusion = []
        for i in range(len(valid_targets_)):
            assert all_same(valid_targets_[i])
            assert len(valid_targets_[i]) == len(valids_preds_[i])
            index_count = 0
            len_subsegment = len(valid_targets_[i])//n_divisions_segments
            for j in range(n_divisions_segments):
                subseg_preds_ = valids_preds_[i][index_count:(index_count+len_subsegment)]
                subseg_target_ = valid_targets_[i][0]
                assert np.mean(subseg_preds_) >= 0 and np.mean(subseg_preds_) <= 1
                if np.mean(subseg_preds_) >= 0.5:
                    valid_preds_subseg_conclusion.append(1)
                else:
                    valid_preds_subseg_conclusion.append(0)
                valid_targets_subseg_conclusion.append(int(subseg_target_))
                index_count += len_subsegment

        for t, p in zip(torch.tensor(valid_targets_subseg_conclusion).view(-1), torch.tensor(valid_preds_subseg_conclusion).view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        all_folds_cms += confusion_matrix
    figure, ax = plot_confusion_matrix(conf_mat=all_folds_cms.detach().cpu().numpy(),
                                       class_names=['low + sympts', 'high + sympts'],
                                       show_absolute=True, show_normed=False, colorbar=True)
    plt.title('Sum of confusion matrices in validation set for each "best model" in CV')
    plt.ylim([1.5, -0.5])
    plt.tight_layout()
    plt.show()



