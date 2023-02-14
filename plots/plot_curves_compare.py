import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import re

if __name__ == '__main__':
    path = '../linear-classifier-rslts_avp_pAug_pretOwn_vpr_dp0507_f1f_th04_sepCE_o2_4segm_len40ov30_'
    name= 'decomp_study_SA047_lr5e-05bs8'
    path2 = '../linear-classifier-rslts_avp_pAug_pretOwn_vpr_dp0507_f1f_th04_sepCE_o2_4segm_len40ov30_'
    name2 = 'decomp_study_SA047_lr5e-05bs8'
    n_folds = 4
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

    train_Dataframes_per_fold2 = []
    valid_Dataframes_per_fold2 = []
    for f in range(n_folds):
        a = pd.read_csv('{}/train_Df_f{}_{}.csv'.format(path2, f, name2), index_col=0)
        b = pd.read_csv('{}/valid_Df_f{}_{}.csv'.format(path2, f, name2), index_col=0)
        if f == 0:
            n_epochs = len(a['epoch'].unique())
        train_Dataframes_per_fold2.append(a)
        valid_Dataframes_per_fold2.append(b)

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

    train_loss_curves_per_fold2 = []
    train_acc_curves_per_fold2 = []
    valid_loss_curves_per_fold2 = []
    valid_acc_curves_per_fold2 = []
    for f in range(n_folds):
        fold_tr_loss_curve2 = []
        fold_tr_acc_curve2 = []
        fold_val_loss_curve2 = []
        fold_val_acc_curve2 = []
        for e in range(n_epochs):
            fold_tr_loss_curve2.append(train_Dataframes_per_fold2[f][train_Dataframes_per_fold2[f]['epoch'] == e].mean()['loss'])
            if task_type == 'classifier':
                fold_tr_acc_curve2.append(train_Dataframes_per_fold2[f][train_Dataframes_per_fold2[f]['epoch'] == e].mean()['accuracy'])
            if use_val:
                fold_val_loss_curve2.append(valid_Dataframes_per_fold2[f][valid_Dataframes_per_fold2[f]['epoch'] == e].mean()['loss'])
                if task_type == 'classifier':
                    if not per_record_val:
                        fold_val_acc_curve2.append(valid_Dataframes_per_fold2[f][valid_Dataframes_per_fold2[f]['epoch'] == e].mean()['accuracy'])
        train_loss_curves_per_fold2.append(fold_tr_loss_curve2)
        train_acc_curves_per_fold2.append(fold_tr_acc_curve2)
        valid_loss_curves_per_fold2.append(fold_val_loss_curve2)
        if not per_record_val:
            valid_acc_curves_per_fold2.append(fold_val_acc_curve2)

    if use_val and per_record_val:
        if task_type == 'classifier':
            for f in range(n_folds):
                with open('{}/mean_acc_curves_f{}_{}.pkl'.format(path2, f, name2), "rb") as input_file:
                    acc_curve_ = pickle.load(input_file)
                valid_acc_curves_per_fold2.append(acc_curve_[1])

# LOSSSS
    #model1
    plt.figure()
    for i in range(n_folds):
        plt.plot(train_loss_curves_per_fold[i], label='Train CV it. {} (m1)'.format(i+1), linestyle='dashed', c='r')
        if use_val:
            plt.plot(valid_loss_curves_per_fold[i], label='Valid CV it. {} (m1)'.format(i+1), c='r')
    #model2
    for i in range(n_folds):
        plt.plot(train_loss_curves_per_fold2[i], label='Train CV it. {} (m2)'.format(i+1), linestyle='dashed', c='b')
        if use_val:
            plt.plot(valid_loss_curves_per_fold2[i], label='Valid CV it. {} (m2)'.format(i+1), c='b')
    plt.title('Training loss, MODELS:\n{} & {}\n(samples {}s, overlap {}s)\n[{} ({}) & {} ({})]'.format(path[2:8], path2[2:8], w_len, overlap, path[15:-1], name, path2[15:-1], name2), fontsize=5)
    if use_lims:
        plt.ylim(loss_lims)
    plt.xlabel('epoch')
    plt.ylabel('cross entropy')
    plt.legend(loc='best', ncol=2, fontsize=8)


    plt.figure()
    #model1
    means = np.mean(np.array(train_loss_curves_per_fold), 0)
    stds = np.std(np.array(train_loss_curves_per_fold), 0)
    plt.fill_between(list(range(n_epochs)), means - stds, means + stds, alpha=0.1, color="coral")
    plt.plot(list(range(n_epochs)), means, "o-", color="coral", label="Train loss (m1)")
    if use_val:
        means = np.mean(np.array(valid_loss_curves_per_fold),0)
        stds = np.std(np.array(valid_loss_curves_per_fold),0)
        plt.fill_between(list(range(n_epochs)), means - stds, means + stds, alpha=0.1, color="darkred")
        plt.plot(list(range(n_epochs)), means, "o-", color="darkred", label="Valid loss (m1)")
    #model2
    means2 = np.mean(np.array(train_loss_curves_per_fold2), 0)
    stds2 = np.std(np.array(train_loss_curves_per_fold2), 0)
    plt.fill_between(list(range(n_epochs)), means2 - stds2, means2 + stds2, alpha=0.1, color="violet")
    plt.plot(list(range(n_epochs)), means2, "o-", color="violet", label="Train loss (m2)")
    if use_val:
        means2 = np.mean(np.array(valid_loss_curves_per_fold2), 0)
        stds2 = np.std(np.array(valid_loss_curves_per_fold2), 0)
        plt.fill_between(list(range(n_epochs)), means2 - stds2, means2 + stds2, alpha=0.1, color="indigo")
        plt.plot(list(range(n_epochs)), means2, "o-", color="indigo", label="Valid loss (m2)")
    plt.title('Mean training loss, MODELS:\n{} & {}\n(samples {}s, overlap {}s)\n[{} ({}) & {} ({})]'.format(path[2:8], path2[2:8], w_len, overlap,
                                                                                            path[15:-1], name, path2[15:-1], name2),
              fontsize=5)
    plt.legend(loc='best', fontsize=8)
    plt.xlabel('epoch')
    if use_lims:
        plt.ylim(loss_lims)
    plt.ylabel('cross entropy')

# ACCURACY
    if task_type == 'classifier':
        plt.figure()
        #model1
        for i in range(n_folds):
            plt.plot(train_acc_curves_per_fold[i], label='Train CV it. {} (m1)'.format(i+1), linestyle='dashed', c='r')
            if use_val:
                plt.plot(valid_acc_curves_per_fold[i], label='Valid CV it. {} (m1)'.format(i+1), c='r')
        #model2
        for i in range(n_folds):
            plt.plot(train_acc_curves_per_fold2[i], label='Train CV it. {} (m2)'.format(i+1), linestyle='dashed', c='b')
            if use_val:
                plt.plot(valid_acc_curves_per_fold2[i], label='Valid CV it. {} (m2)'.format(i+1), c='b')
        plt.title('Training accuracy, MODEL:\n{} & {}\n(samples {}s, overlap {}s)\n[{} ({}) & {} ({})]'.format(path[2:8], path2[2:8], w_len, overlap, path[15:-1], name,
                                                                                                                path2[15:-1], name2), fontsize=5)
        plt.legend(loc='best', ncol=2, fontsize=8)
        plt.xlabel('epoch')
        if use_lims:
            plt.ylim(acc_lims)
        plt.ylabel('accuracy')

        plt.figure()
        #model1
        means = np.mean(train_acc_curves_per_fold, 0)
        stds = np.std(train_acc_curves_per_fold, 0)
        plt.fill_between(list(range(n_epochs)), means - stds, means + stds, alpha=0.1, color="coral")
        plt.plot(list(range(n_epochs)), means, "o-", color="coral", label="Train acc")
        if use_val:
            means = np.mean(valid_acc_curves_per_fold, 0)
            stds = np.std(valid_acc_curves_per_fold, 0)
            plt.fill_between(list(range(n_epochs)), means - stds, means + stds, alpha=0.1, color="darkred")
            plt.plot(list(range(n_epochs)), means, "o-", color="darkred", label="Valid acc")
        #model2
        means2 = np.mean(train_acc_curves_per_fold2, 0)
        stds2 = np.std(train_acc_curves_per_fold2, 0)
        plt.fill_between(list(range(n_epochs)), means2 - stds2, means2 + stds2, alpha=0.1, color="violet")
        plt.plot(list(range(n_epochs)), means2, "o-", color="violet", label="Train acc")
        if use_val:
            means2 = np.mean(valid_acc_curves_per_fold2, 0)
            stds2 = np.std(valid_acc_curves_per_fold2, 0)
            plt.fill_between(list(range(n_epochs)), means2 - stds2, means2 + stds2, alpha=0.1, color="indigo")
            plt.plot(list(range(n_epochs)), means2, "o-", color="indigo", label="Valid acc")
        plt.title('Mean training accuracy, MODEL:\n{} & {}\n(samples {}s, overlap {}s)\n[{} ({}) & {} ({})]'.format(path[2:8], path2[2:8], w_len, overlap, path[15:-1], name,
                                                                                                                     path2[15:-1], name2), fontsize=5)
        plt.legend(loc='best', fontsize=8)
        plt.xlabel('epoch')
        if use_lims:
            plt.ylim(acc_lims)
        plt.ylabel('accuracy')

    plt.show()
    a = 0

