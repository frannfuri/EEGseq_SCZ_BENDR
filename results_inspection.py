import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

if __name__ == '__main__':
    path = './results/results_new_linear'
    name= 'decomp_study_SA047_AD1_lr1e-05bs16_conclusion'
    n_folds = 4

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
            fold_val_loss_curve.append(valid_Dataframes_per_fold[f][valid_Dataframes_per_fold[f]['epoch'] == e].mean()['loss'])
            fold_val_acc_curve.append(valid_Dataframes_per_fold[f][valid_Dataframes_per_fold[f]['epoch'] == e].mean()['accuracy'])
        train_loss_curves_per_fold.append(fold_tr_loss_curve)
        train_acc_curves_per_fold.append(fold_tr_acc_curve)
        valid_loss_curves_per_fold.append(fold_val_loss_curve)
        valid_acc_curves_per_fold.append(fold_val_acc_curve)


    c = 0

    for i in range(4):
        plt.plot(train_loss_curves_per_fold[i], label='Train CV it. {}'.format(i), linestyle='dashed')
        plt.plot(valid_loss_curves_per_fold[i], label='Valid CV it. {}'.format(i))
    plt.title('Training loss')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('cross entropy')

    for i in range(4):
        plt.plot(train_acc_curves_per_fold[i], label='Train CV it. {}'.format(i), linestyle='dashed')
        plt.plot(valid_acc_curves_per_fold[i], label='Valid CV it. {}'.format(i))
    plt.title('Training accuracy')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

