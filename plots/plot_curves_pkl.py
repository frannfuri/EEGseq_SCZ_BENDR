import pickle
import matplotlib.pyplot as plt
import re
import numpy as np

if __name__ == '__main__':
    path = '../linear-classifier-rslts_avp_pAug_pretOwn_vpr_dp0507_f1f_th04_newL_o2_len40ov30_'
    name= 'decomp_study_SA047_lr5e-05bs8'
    n_folds = 4
    task_type = 'classifier'
    use_lims = True
    use_val = True
    #use_val = False
    per_record_val = True
    if use_lims:
        loss_lims = (0.3, 2)
        acc_lims = (0.3, 0.95)
#############################################3

    w_len = int((re.search('len(.+?)ov', path)).group(1))
    overlap = int((re.search('ov(.+?)_', path)).group(1))

    train_accs_per_fold = []
    valid_accs_per_fold = []
    train_accs_per_sample_per_fold = []
    valid_accs_per_sample_per_fold = []
    train_losses_per_fold = []
    valid_losses_per_fold = []
    for f in range(n_folds):
        # Accuracy
        file = open('{}/mean_acc_curves_f{}_{}.pkl'.format(path, f, name), 'rb')
        mean_acc = pickle.load(file)
        train_accs_per_fold.append(mean_acc[0])
        valid_accs_per_fold.append(mean_acc[1])
        file.close()
        # Per sample Accuracy
        file = open('{}/mean_acc_per_sample_curves_f{}_{}.pkl'.format(path, f, name), 'rb')
        mean_acc_per_sample = pickle.load(file)
        train_accs_per_sample_per_fold.append(mean_acc_per_sample[0])
        valid_accs_per_sample_per_fold.append(mean_acc_per_sample[1])
        file.close()
        # Loss
        file = open('{}/mean_loss_curves_f{}_{}.pkl'.format(path, f, name), 'rb')
        mean_loss = pickle.load(file)
        train_losses_per_fold.append(mean_loss[0])
        valid_losses_per_fold.append(mean_loss[1])
        file.close()

    n_epochs = len(train_accs_per_fold[0])

    # PLOT ACCURACY
    plt.figure()
    for i in range(n_folds):
        plt.plot(train_accs_per_fold[i], label='Train CV it. {}'.format(i+1), linestyle='dashed')
        plt.plot(valid_accs_per_fold[i], label='Valid CV it. {}'.format(i + 1))
        plt.title('Training accuracy, MODEL: {}\n(samples {}s, overlap {}s)\n[{} ({})]'.format(path[2:8], w_len, overlap,
                                                                                           path[15:-1], name),
                  fontsize=8)
        plt.legend(loc='best', ncol=2, fontsize=8)
        plt.xlabel('epoch')
        if use_lims:
            plt.ylim(acc_lims)
        plt.ylabel('accuracy')

    plt.figure()
    means = np.mean(np.array(train_accs_per_fold), 0)
    stds = np.std(np.array(train_accs_per_fold), 0)
    plt.fill_between(list(range(n_epochs)), means - stds, means + stds, alpha=0.1, color="r")
    plt.plot(list(range(n_epochs)), means, "o-", color="r", label="Train accuracy")
    means = np.mean(np.array(valid_accs_per_fold), 0)
    stds = np.std(np.array(valid_accs_per_fold), 0)
    plt.fill_between(list(range(n_epochs)), means - stds, means + stds, alpha=0.1, color="b")
    plt.plot(list(range(n_epochs)), means, "o-", color="b", label="Valid accs")
    plt.title('Mean training accs, MODEL: {}\n(samples {}s, overlap {}s)\n[{} ({})]'.format(path[2:8], w_len, overlap,
                                                                                            path[15:-1], name),
              fontsize=8)
    plt.legend(loc='best', fontsize=8)
    plt.xlabel('epoch')
    if use_lims:
        plt.ylim(acc_lims)
    plt.ylabel('accuracy')

    # PLOT ACCURACY PER SAMPLE
    plt.figure()
    for i in range(n_folds):
        plt.plot(train_accs_per_sample_per_fold[i], label='Train CV it. {}'.format(i + 1), linestyle='dashed')
        plt.plot(valid_accs_per_sample_per_fold[i], label='Valid CV it. {}'.format(i + 1))
        plt.title(
            'Training accuracy per window, MODEL: {}\n(samples {}s, overlap {}s)\n[{} ({})]'.format(path[2:8], w_len, overlap,
                                                                                         path[15:-1], name),
            fontsize=8)
        plt.legend(loc='best', ncol=2, fontsize=8)
        plt.xlabel('epoch')
        if use_lims:
            plt.ylim(acc_lims)
        plt.ylabel('accuracy per window')

    plt.figure()
    means = np.mean(np.array(train_accs_per_sample_per_fold), 0)
    stds = np.std(np.array(train_accs_per_sample_per_fold), 0)
    plt.fill_between(list(range(n_epochs)), means - stds, means + stds, alpha=0.1, color="r")
    plt.plot(list(range(n_epochs)), means, "o-", color="r", label="Train accuracy")
    means = np.mean(np.array(valid_accs_per_sample_per_fold), 0)
    stds = np.std(np.array(valid_accs_per_sample_per_fold), 0)
    plt.fill_between(list(range(n_epochs)), means - stds, means + stds, alpha=0.1, color="b")
    plt.plot(list(range(n_epochs)), means, "o-", color="b", label="Valid accs")
    plt.title('Mean training accs per window, MODEL: {}\n(samples {}s, overlap {}s)\n[{} ({})]'.format(path[2:8], w_len, overlap,
                                                                                            path[15:-1], name),
              fontsize=8)
    plt.legend(loc='best', fontsize=8)
    plt.xlabel('epoch')
    if use_lims:
        plt.ylim(acc_lims)
    plt.ylabel('accuracy per window')

    # PLOT LOSS
    plt.figure()
    for i in range(n_folds):
        plt.plot(train_losses_per_fold[i], label='Train CV it. {}'.format(i+1), linestyle='dashed')
        plt.plot(valid_losses_per_fold[i], label='Valid CV it. {}'.format(i + 1))
        plt.title('Training loss, MODEL: {}\n(samples {}s, overlap {}s)\n[{} ({})]'.format(path[2:8], w_len, overlap,
                                                                                           path[15:-1], name),
                  fontsize=8)
        plt.legend(loc='best', ncol=2, fontsize=8)
        plt.xlabel('epoch')
        if use_lims:
            plt.ylim(loss_lims)
        plt.ylabel('loss')

    plt.figure()
    means = np.mean(np.array(train_losses_per_fold), 0)
    stds = np.std(np.array(train_losses_per_fold), 0)
    plt.fill_between(list(range(n_epochs)), means - stds, means + stds, alpha=0.1, color="r")
    plt.plot(list(range(n_epochs)), means, "o-", color="r", label="Train loss")
    means = np.mean(np.array(valid_losses_per_fold), 0)
    stds = np.std(np.array(valid_losses_per_fold), 0)
    plt.fill_between(list(range(n_epochs)), means - stds, means + stds, alpha=0.1, color="b")
    plt.plot(list(range(n_epochs)), means, "o-", color="b", label="Valid loss")
    plt.title('Mean training loss, MODEL: {}\n(samples {}s, overlap {}s)\n[{} ({})]'.format(path[2:8], w_len, overlap,
                                                                                            path[15:-1], name),
              fontsize=8)
    plt.legend(loc='best', fontsize=8)
    plt.xlabel('epoch')
    if use_lims:
        plt.ylim(loss_lims)
    plt.ylabel('loss')

    plt.show()
    a = 0
