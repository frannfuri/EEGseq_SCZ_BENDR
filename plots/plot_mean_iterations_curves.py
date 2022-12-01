import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import re

if __name__ == '__main__':
    ### HARDCODED FOR 1 FOLD
    general_path1 = './BENDR-rslts_pAug_avpf_dp0307_bw_vpr'
    general_path2 = '_len40ov30_'
    name= 'trivial_set_lr0.0003bs32'
    #n_folds = 1
    n_tries = 5
    use_lims = True
    use_val = False
    per_record_val = False
    if use_lims:
        loss_lims = (0, 1.5)
        acc_lims = (0.3, 0.95)

    ###############################################
    w_len = int((re.search('len(.+?)ov', general_path2)).group(1))
    overlap = int((re.search('ov(.+?)_',general_path2)).group(1))  # must be a even (par) number?
    train_Dataframes_per_try = []
    valid_Dataframes_per_try = []

    for t in range(n_tries):
        a = pd.read_csv('{}/train_Df_f0_{}.csv'.format('{}{}{}'.format(general_path1, t, general_path2), name), index_col=0)
        b = pd.read_csv('{}/valid_Df_f0_{}.csv'.format('{}{}{}'.format(general_path1, t, general_path2), name), index_col=0)
        if t == 0:
            n_epochs = len(a['epoch'].unique())
        train_Dataframes_per_try.append(a)
        valid_Dataframes_per_try.append(b)

    train_loss_curves_per_try = []
    train_acc_curves_per_try = []
    valid_loss_curves_per_try = []
    valid_acc_curves_per_try = []

    for t in range(n_tries):
        fold_tr_loss_curve = []
        fold_tr_acc_curve = []
        fold_val_loss_curve = []
        fold_val_acc_curve = []
        for e in range(n_epochs):
            fold_tr_loss_curve.append(train_Dataframes_per_try[t][train_Dataframes_per_try[t]['epoch'] == e].mean()['loss'])
            fold_tr_acc_curve.append(train_Dataframes_per_try[t][train_Dataframes_per_try[t]['epoch'] == e].mean()['accuracy'])
            if use_val:
                fold_val_loss_curve.append(valid_Dataframes_per_try[t][valid_Dataframes_per_try[t]['epoch'] == e].mean()['loss'])
                if not per_record_val:
                    fold_val_acc_curve.append(valid_Dataframes_per_try[t][valid_Dataframes_per_try[t]['epoch'] == e].mean()['accuracy'])
        train_loss_curves_per_try.append(fold_tr_loss_curve)
        train_acc_curves_per_try.append(fold_tr_acc_curve)
        valid_loss_curves_per_try.append(fold_val_loss_curve)
        if not per_record_val:
            valid_acc_curves_per_try.append(fold_val_acc_curve)

    if use_val and per_record_val:
        for t in range(n_tries):
            with open('{}/mean_acc_curves_f0_{}.pkl'.format('{}{}{}'.format(general_path1, t, general_path2), name), "rb") as input_file:
                acc_curve_ = pickle.load(input_file)
            valid_acc_curves_per_try.append(acc_curve_[1])

    path = general_path1+general_path2
    plt.figure()
    for i in range(n_tries):
        plt.plot(train_loss_curves_per_try[i], label='Train it. {}'.format(i+1), linestyle='dashed')
        if use_val:
            plt.plot(valid_loss_curves_per_try[i], label='Valid it. {}'.format(i+1))
    plt.title('Training loss, MODEL: {}\n(samples {}s, overlap {}s)\n[{} ({})]'.format(path[2:8], w_len, overlap, path[15:-1], name), fontsize=8)
    plt.legend(loc='best', ncol=2, fontsize=8)
    plt.xlabel('epoch')
    if use_lims:
        plt.ylim(loss_lims)
    plt.ylabel('cross entropy')

    plt.figure()
    means = np.mean(np.array(train_loss_curves_per_try), 0)
    stds = np.std(np.array(train_loss_curves_per_try), 0)
    plt.fill_between(list(range(n_epochs)), means - stds, means + stds, alpha=0.1, color="r")
    plt.plot(list(range(n_epochs)), means, "o-", color="r", label="Train loss")
    if use_val:
        means = np.mean(np.array(valid_loss_curves_per_try),0)
        stds = np.std(np.array(valid_loss_curves_per_try),0)
        plt.fill_between(list(range(n_epochs)), means - stds, means + stds, alpha=0.1, color="b")
        plt.plot(list(range(n_epochs)), means, "o-", color="b", label="Valid loss")
    plt.title('Mean training loss, MODEL: {}\n(samples {}s, overlap {}s)\n[{} ({})]'.format(path[2:8], w_len, overlap, path[15:-1], name), fontsize=8)
    plt.legend(loc='best', fontsize=8)
    plt.xlabel('epoch')
    if use_lims:
        plt.ylim(loss_lims)
    plt.ylabel('cross entropy')

    plt.figure()
    for i in range(n_tries):
        plt.plot(train_acc_curves_per_try[i], label='Train CV it. {}'.format(i+1), linestyle='dashed')
        if use_val:
            plt.plot(valid_acc_curves_per_try[i], label='Valid CV it. {}'.format(i+1))
    plt.title('Training accuracy, MODEL: {}\n(samples {}s, overlap {}s)\n[{} ({})]'.format(path[2:8], w_len, overlap, path[15:-1], name), fontsize=8)
    plt.legend(loc='best', ncol=2, fontsize=8)
    plt.xlabel('epoch')
    if use_lims:
        plt.ylim(acc_lims)
    plt.ylabel('accuracy')

    plt.figure()
    means = np.mean(train_acc_curves_per_try, 0)
    stds = np.std(train_acc_curves_per_try, 0)
    plt.fill_between(list(range(n_epochs)), means - stds, means + stds, alpha=0.1, color="r")
    plt.plot(list(range(n_epochs)), means, "o-", color="r", label="Train acc")
    if use_val:
        means = np.mean(valid_acc_curves_per_try, 0)
        stds = np.std(valid_acc_curves_per_try, 0)
        plt.fill_between(list(range(n_epochs)), means - stds, means + stds, alpha=0.1, color="b")
        plt.plot(list(range(n_epochs)), means, "o-", color="b", label="Valid acc")
    plt.title('Mean training accuracy, MODEL: {}\n(samples {}s, overlap {}s)\n[{} ({})]'.format(path[2:8], w_len, overlap, path[15:-1], name), fontsize=8)
    plt.legend(loc='best', fontsize=8)
    plt.xlabel('epoch')
    if use_lims:
        plt.ylim(acc_lims)
    plt.ylabel('accuracy')

    plt.show(block=False)
    a = 0

