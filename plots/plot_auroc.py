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
from sklearn.metrics import roc_curve, roc_auc_score

def plot_roc_curves(path, n_folds):
    fig, axs = plt.subplots(2,3, figsize=(15,8))
    for f in range(n_folds):
        y_prob_preds = np.load('./Y_PROB_PRED_f{}_{}.npy'.format(f, path))
        y_targets = np.load('./Y_TARGET_f{}_{}.npy'.format(f, path))
        fpr, tpr, thrs = roc_curve(y_targets, y_prob_preds)
        auc = roc_auc_score(y_targets, y_prob_preds)
        # plot ROC curve
        axs[f//3, f%3].plot(fpr, tpr, color='darkorange', label="AUC={:.4f}".format(auc))
        axs[f//3, f%3].set_ylabel('True Positive Rate')
        axs[f//3, f%3].set_xlabel('False Positive Rate')
        axs[f//3, f%3].set_title('ROC curve it. n° {}'.format(f+1), fontsize=10)
        diff = []
        for x in range(len(thrs)):
            diff.append(tpr[x]-fpr[x])
        max_id = np.argmax(diff)
        axs[f//3, f%3].plot(fpr[max_id], tpr[max_id], 'r*', markersize=8, label='max(TPR-FPR) --> threshold {:.2f}'.format(thrs[max_id]))
        # marcar threshold 0.4, 0.5 y 0.6
        mark_ths = [0.4, 0.5, 0.6]
        cs = ['b', 'g', 'fuchsia']
        mark_ths_ids = []
        for i in range(3):
            diff_th = []
            for t in thrs:
                diff_th.append(np.abs(t-mark_ths[i]))
            mark_ths_ids.append(np.argmin(diff_th))
        for i in range(3):
            axs[f//3, f%3].plot(fpr[mark_ths_ids[i]], tpr[mark_ths_ids[i]], '.', color= cs[i], markersize=8,
                     label='threshold {:.2f}'.format(thrs[mark_ths_ids[i]]))
        axs[f//3, f%3].legend(fontsize=8, loc=4)
    plt.tight_layout()
    plt.show()
    return thrs


if __name__ == '__main__':
    # PARAMETERS
    # PARAMETERS
    data_path = '../../BENDR_datasets/decomp_study_SA039'  ### CHECK .yml IS OKAY!!
    n_folds = 5
    model_path1 = '../linear-classifier-rslts_avp_pAug_pretOwn_vpr_dp0307_f1f_th04_bcePw02_stepLR01_len40ov30_/best_model_f'
    model_path2 = '_decomp_study_SA039_lr0.0001bs8.pt'



    #################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('---Using ' + str(device) + 'device---')

    with open(data_path + '/info.yml') as infile:
        data_settings = yaml.load(infile, Loader=yaml.FullLoader)
    valid_sets_path = data_settings['valid_sets_path']
    with open('../{}'.format(valid_sets_path), newline='') as f:
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
                                                                   apply_winsor=data_settings['apply_winsorising'], new_sfreq=256)

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

        # Predictions
        ys_prob_pred = []
        ys_target = []
        for x, y, _ in validloader:
            output = model(x)
            prepred = torch.sigmoid(output)
            ys_prob_pred.append(prepred.squeeze().detach().numpy())
            ys_target.append(y.squeeze().detach().numpy())
        np.save('../Y_PROB_PRED_f{}_{}.npy'.format(f, model_path1[3:-14]+model_path2[:-3]), ys_prob_pred)
        np.save('../Y_TARGET_f{}_{}.npy'.format(f, model_path1[3:-14]+model_path2[:-3]), ys_target)
