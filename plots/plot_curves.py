import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import re

if __name__ == '__main__':
    path = '../linear-classifier-rslts_avp_pAug_pretOwn_vpr_dp0507_f1f_th04_clipL05_len40ov30_'
    #path = '../linear-classifier-rslts_avp_pAug_bw_dp0307_f1f_th04_clipL_len40ov30_'
    name= 'decomp_study_SA039_lr5e-05bs8'
    #name = 'h_scz_study_lr0.0001bs8'  #5e-05bs8'
    n_folds = 5
    task_type = 'classifier'
    use_lims = True
    use_val = True
    #use_val = False
    per_record_val = True
    if use_lims:
        loss_lims = (0, 1.5)
        acc_lims = (0.3, 0.95)

    ###############################################
    w_len = int((re.search('len(.+?)ov', path)).group(1))
    overlap = int((re.search('ov(.+?)_',path)).group(1))  # must be a even (par) number?
    train_Dataframes_per_fold = []
    valid_Dataframes_per_fold = []
    for f in range(n_folds):
        a = pd.read_csv('{}/train_Df_f{}_{}.csv'.format(path, f, name), index_col=0)
        b = pd.read_csv('{}/valid_Df_f{}_{}.csv'.format(path, f, name), index_col=0)
        if f == 0:
            n_epochs = len(a['epoch'].unique())
        train_Dataframes_per_fold.append(a)
        valid_Dataframes_per_fold.append(b)

    train_loss_curves_per_fold = []
    train_acc_curves_per_fold = []
    valid_loss_curves_per_fold = []
    valid_acc_curves_per_fold = []

    for f in range(n_folds):
        fold_tr_loss_curve = []
        fold_tr_acc_curve = []
        fold_val_loss_curve = []
        fold_val_acc_curve = []
        for e in range(n_epochs):
            fold_tr_loss_curve.append(train_Dataframes_per_fold[f][train_Dataframes_per_fold[f]['epoch'] == e].mean()['loss'])
            if task_type == 'classifier':
                fold_tr_acc_curve.append(train_Dataframes_per_fold[f][train_Dataframes_per_fold[f]['epoch'] == e].mean()['accuracy'])
            if use_val:
                fold_val_loss_curve.append(valid_Dataframes_per_fold[f][valid_Dataframes_per_fold[f]['epoch'] == e].mean()['loss'])
                if task_type == 'classifier':
                    if not per_record_val:
                        fold_val_acc_curve.append(valid_Dataframes_per_fold[f][valid_Dataframes_per_fold[f]['epoch'] == e].mean()['accuracy'])
        train_loss_curves_per_fold.append(fold_tr_loss_curve)
        train_acc_curves_per_fold.append(fold_tr_acc_curve)
        valid_loss_curves_per_fold.append(fold_val_loss_curve)
        if not per_record_val:
            valid_acc_curves_per_fold.append(fold_val_acc_curve)

    if use_val and per_record_val:
        if task_type == 'classifier':
            for f in range(n_folds):
                with open('{}/mean_acc_curves_f{}_{}.pkl'.format(path, f, name), "rb") as input_file:
                    acc_curve_ = pickle.load(input_file)
                valid_acc_curves_per_fold.append(acc_curve_[1])

    plt.figure()
    for i in range(n_folds):
        plt.plot(train_loss_curves_per_fold[i], label='Train CV it. {}'.format(i+1), linestyle='dashed')
        if use_val:
            plt.plot(valid_loss_curves_per_fold[i], label='Valid CV it. {}'.format(i+1))
    plt.title('Training loss, MODEL: {}\n(samples {}s, overlap {}s)\n[{} ({})]'.format(path[2:8], w_len, overlap, path[15:-1], name), fontsize=8)
    plt.legend(loc='best', ncol=2, fontsize=8)
    plt.xlabel('epoch')
    if use_lims:
        plt.ylim(loss_lims)
    plt.ylabel('cross entropy')

    plt.figure()
    means = np.mean(np.array(train_loss_curves_per_fold), 0)
    stds = np.std(np.array(train_loss_curves_per_fold), 0)
    plt.fill_between(list(range(n_epochs)), means - stds, means + stds, alpha=0.1, color="r")
    plt.plot(list(range(n_epochs)), means, "o-", color="r", label="Train loss")
    if use_val:
        means = np.mean(np.array(valid_loss_curves_per_fold),0)
        stds = np.std(np.array(valid_loss_curves_per_fold),0)
        plt.fill_between(list(range(n_epochs)), means - stds, means + stds, alpha=0.1, color="b")
        plt.plot(list(range(n_epochs)), means, "o-", color="b", label="Valid loss")
    plt.title('Mean training loss, MODEL: {}\n(samples {}s, overlap {}s)\n[{} ({})]'.format(path[2:8], w_len, overlap, path[15:-1], name), fontsize=8)
    plt.legend(loc='best', fontsize=8)
    plt.xlabel('epoch')
    if use_lims:
        plt.ylim(loss_lims)
    plt.ylabel('cross entropy')

    if task_type == 'classifier':
        plt.figure()
        for i in range(n_folds):
            plt.plot(train_acc_curves_per_fold[i], label='Train CV it. {}'.format(i+1), linestyle='dashed')
            if use_val:
                plt.plot(valid_acc_curves_per_fold[i], label='Valid CV it. {}'.format(i+1))
        plt.title('Training accuracy, MODEL: {}\n(samples {}s, overlap {}s)\n[{} ({})]'.format(path[2:8], w_len, overlap, path[15:-1], name), fontsize=8)
        plt.legend(loc='best', ncol=2, fontsize=8)
        plt.xlabel('epoch')
        if use_lims:
            plt.ylim(acc_lims)
        plt.ylabel('accuracy')

        plt.figure()
        means = np.mean(train_acc_curves_per_fold, 0)
        stds = np.std(train_acc_curves_per_fold, 0)
        plt.fill_between(list(range(n_epochs)), means - stds, means + stds, alpha=0.1, color="r")
        plt.plot(list(range(n_epochs)), means, "o-", color="r", label="Train acc")
        if use_val:
            means = np.mean(valid_acc_curves_per_fold, 0)
            stds = np.std(valid_acc_curves_per_fold, 0)
            plt.fill_between(list(range(n_epochs)), means - stds, means + stds, alpha=0.1, color="b")
            plt.plot(list(range(n_epochs)), means, "o-", color="b", label="Valid acc")
        plt.title('Mean training accuracy, MODEL: {}\n(samples {}s, overlap {}s)\n[{} ({})]'.format(path[2:8], w_len, overlap, path[15:-1], name), fontsize=8)
        plt.legend(loc='best', fontsize=8)
        plt.xlabel('epoch')
        if use_lims:
            plt.ylim(acc_lims)
        plt.ylabel('accuracy')

    plt.show()
    a = 0

