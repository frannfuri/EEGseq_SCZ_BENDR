import argparse
import pickle
import torch
import yaml
import os
import csv
from datasets import charge_dataset, standardDataset, recInfoDataset
from architectures import Net
from trainables import train_scratch_model, train_scratch_model_no_valid
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from utils import comp_confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

if __name__ == '__main__':
    # Arguments and preliminaries
    parser = argparse.ArgumentParser(description="Train models from simpler to more complex.")
    parser.add_argument('--multi-gpu', action='store_true', help='Distribute model over multiple GPUs')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of dataloader workers.')
    parser.add_argument('--results-filename', default=None, help='What to name the spreadsheet produced with all '
                                                                 'final results.')
    parser.add_argument('--dataset-directory', default=None,
                        help='Where is the ubication of the data samples and the information '
                             'associated to them.')
    parser.add_argument('--random-seed', default=298,
                        help='Set fixed random seed.')
    parser.add_argument('--save-models', action='store_true',
                        help='Wether to save or not the best models per CV iteration.')
    parser.add_argument('--use-valid', action='store_true',
                       help='Wether to use or not a valid subset of the data.')
    parser.add_argument('--plot-cm', action='store_true',
                       help='Wether to plot or not a confusion matrix for each fold best model.')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('---Using ' + str(device) + 'device---')

    with open(args.dataset_directory + '/info.yml') as infile:
        data_settings = yaml.load(infile, Loader=yaml.FullLoader)

    if args.use_valid:
        with open('./{}'.format(data_settings['valid_sets_path']), newline='') as f:
            reader = csv.reader(f)
            valid_sets = list(reader)
    os.makedirs('./results2_' + args.results_filename + '_len{}ov{}_'.format(
                                    data_settings['tlen'], data_settings['overlap_len']), exist_ok=True)

    # Load dataset
    # list of len: n_records
    # each list[n] is of dim [n_segments, 20 , len_segments (256*tlen)]
    array_epochs_all_records, sorted_record_names = charge_dataset(directory=args.dataset_directory,
                                                  tlen=data_settings['tlen'], overlap=data_settings['overlap_len'],
                                                  data_max=data_settings['data_max'], data_min=data_settings['data_min'],
                                                  chns_consider=data_settings['chns_to_consider'],
                                                  labels_path=data_settings['labels_path'], target_f=data_settings['target_feature'],
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

    # Set fixed random number seed
    torch.manual_seed(args.random_seed)
    print('-------------------------------------------')

    # Train parameters
    bs = data_settings['batch_size']
    num_cls = data_settings['num_cls']
    samples_tlen = data_settings['tlen']
    lr = data_settings['lr']
    num_epochs = data_settings['epochs']

    # TODO: WHATS IS THIS? Receptive field: 143 samples | Downsampled by 96 | Overlap of 47 samples | 106 encoded samples/trial
    # K-fold Cross Validation
    best_epoch_fold = []
    print('DATASET: {}'.format(data_settings['name']))
    n_folds = len(valid_sets) if args.use_valid else 1

    for fold in range(n_folds):
        # Dataset and dataloaders

        if args.use_valid:
            valid_ids = [i for i in range(len(sorted_record_names)) if (sorted_record_names[i][:11] in valid_sets[fold])]
            train_ids = [i for i in range(len(sorted_record_names)) if (sorted_record_names[i][:11] not in valid_sets[fold])]
            print('FOLD n°{}'.format(fold))
            print('-----------------------')

            # Split data
            train_dataset = recInfoDataset(all_X[train_ids], all_y[train_ids], [sorted_record_names[i] for i in train_ids])
            valid_dataset = recInfoDataset(all_X[valid_ids], all_y[valid_ids], [sorted_record_names[i] for i in valid_ids])
            #train_dataset = standardDataset(all_X[train_ids], all_y[train_ids])
            #valid_dataset = standardDataset(all_X[valid_ids], all_y[valid_ids])


            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
            validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=True)
            dataloaders = {'train': trainloader, 'valid': validloader}
        else:
            dataset = standardDataset(all_X, all_y)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
            dataloaders = {'train': dataloader}

        # Model
        model = Net()

        if args.multi_gpu:
            model = nn.DataParallel(model)
        model = model.to(device)

        # Loss and Optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Train
        if args.use_valid:
            best_model, curves_accs, curves_losses, train_df, valid_df, best_epoch = train_scratch_model(
                model, criterion, optimizer, dataloaders, device, num_epochs, valid_sets[fold], len(valid_dataset))
        else:
            best_model, curves_accs, curves_losses, train_df, valid_df, best_epoch = train_scratch_model_no_valid(
                                            model, criterion, optimizer, dataloaders, device, num_epochs)

        best_epoch_fold.append(best_epoch)
        train_df.to_csv("./results2_{}_len{}ov{}_/train_Df_f{}_{}_lr{}bs{}.csv".format(args.results_filename, data_settings['tlen'], data_settings['overlap_len'],
                                                                                       fold, args.dataset_directory.split('/')[-1], lr, bs))
        valid_df.to_csv("./results2_{}_len{}ov{}_/valid_Df_f{}_{}_lr{}bs{}.csv".format(args.results_filename, data_settings['tlen'], data_settings['overlap_len'],
                                                                                       fold, args.dataset_directory.split('/')[-1], lr, bs))
        with open('./results2_{}_len{}ov{}_/mean_loss_curves_f{}_{}_lr{}bs{}.pkl'.format(args.results_filename, data_settings['tlen'], data_settings['overlap_len'],
                                                                                       fold, args.dataset_directory.split('/')[-1], lr, bs), 'wb') as f:
            pickle.dump(curves_losses, f)
        with open('./results2_{}_len{}ov{}_/mean_acc_curves_f{}_{}_lr{}bs{}.pkl'.format(args.results_filename, data_settings['tlen'], data_settings['overlap_len'],
                                                                                        fold, args.dataset_directory.split('/')[-1], lr, bs), 'wb') as f:
            pickle.dump(curves_accs, f)

        if args.save_models:
            torch.save(best_model.state_dict(), './{}_results2_{}_len{}ov{}_/best_model_f{}_{}_lr{}bs{}.pt'.format(args.results_filename, data_settings['tlen'],
                                                                                        data_settings['overlap_len'], fold, args.dataset_directory.split('/')[-1], lr, bs))

        # Plot Confusion Matrix
        if args.plot_cm:
            if args.use_valid:
                cm = comp_confusion_matrix(best_model, dataloaders['valid'], data_settings['num_cls'], device)
            else:
                cm = comp_confusion_matrix(best_model, dataloaders['train'], data_settings['num_cls'], device)

            figure, ax = plot_confusion_matrix(conf_mat=cm.detach().cpu().numpy(),
                                               class_names=['low + sympts', 'high + sympts'],
                                               show_absolute=True, show_normed=False, colorbar=True)
            plt.title('Confusion matrix in {} set'.format('validation' if args.use_valid else 'train'))
            plt.ylim([1.5, -0.5])
            plt.tight_layout()
    print('Best epoch for each of the cross-validations iterations:\n{}'.format(best_epoch_fold))
    plt.show()
    a = 0


