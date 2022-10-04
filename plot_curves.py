import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import re

if __name__ == '__main__':
    path = './results2/results2_tests04-10_len40ov30_'
        #'./results/results_SA047_lin_len40ov30_ndo'
    name= 'decomp_study_SA047_scratch_lr0.0003bs16'
        #'decomp_study_SA047_AD1_lr1e-05bs16_real_and_pred_SAPS'
    n_folds = 1
    use_lims = False#True
    if use_lims:
        loss_lims = (0, 3)
        acc_lims = (0.4, 1.0)
    use_val = False


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
            fold_tr_acc_curve.append(train_Dataframes_per_fold[f][train_Dataframes_per_fold[f]['epoch'] == e].mean()['accuracy'])
            if use_val:
                fold_val_loss_curve.append(valid_Dataframes_per_fold[f][valid_Dataframes_per_fold[f]['epoch'] == e].mean()['loss'])
                fold_val_acc_curve.append(valid_Dataframes_per_fold[f][valid_Dataframes_per_fold[f]['epoch'] == e].mean()['accuracy'])
        train_loss_curves_per_fold.append(fold_tr_loss_curve)
        train_acc_curves_per_fold.append(fold_tr_acc_curve)
        valid_loss_curves_per_fold.append(fold_val_loss_curve)
        valid_acc_curves_per_fold.append(fold_val_acc_curve)

    plt.figure()
    for i in range(n_folds):
        plt.plot(train_loss_curves_per_fold[i], label='Train CV it. {}'.format(i+1), linestyle='dashed')
        if use_val:
            plt.plot(valid_loss_curves_per_fold[i], label='Valid CV it. {}'.format(i+1))
    plt.title('Training loss\n(samples {}s, overlap {}s)'.format(w_len, overlap), fontsize=10)
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
    plt.title('Mean training loss\n(samples {}s, overlap {}s)'.format(w_len, overlap), fontsize=10)
    plt.legend(loc='best', fontsize=8)
    plt.xlabel('epoch')
    if use_lims:
        plt.ylim(loss_lims)
    plt.ylabel('cross entropy')

    plt.figure()
    for i in range(n_folds):
        plt.plot(train_acc_curves_per_fold[i], label='Train CV it. {}'.format(i+1), linestyle='dashed')
        if use_val:
            plt.plot(valid_acc_curves_per_fold[i], label='Valid CV it. {}'.format(i+1))
    plt.title('Training accuracy\n(samples {}s, overlap {}s)'.format(w_len, overlap), fontsize=10)
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
    plt.title('Mean training accuracy\n(samples {}s, overlap {}s)'.format(w_len, overlap), fontsize=10)
    plt.legend(loc='best', fontsize=8)
    plt.xlabel('epoch')
    if use_lims:
        plt.ylim(acc_lims)
    plt.ylabel('accuracy')

    plt.show(block=False)
    a = 0

