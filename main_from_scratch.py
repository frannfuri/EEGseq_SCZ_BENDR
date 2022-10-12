import argparse
import pickle
import torch
import yaml
import os
import csv
from datasets import charge_dataset, standardDataset, recInfoDataset
from architectures import Net, LinearHeadBENDR_from_scratch
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
                        help='Whether to save or not the best models per CV iteration.')
    parser.add_argument('--use-valid', action='store_true',
                       help='Whether to use or not a valid subset of the data.')
    parser.add_argument('--valid-per-record', action='store_true',
                        help="Whether to validate considerating the whole record. "
                             "Will only be done if use-valid is true.")
    parser.add_argument('--plot-cm', action='store_true',
                       help='Whether to plot or not a confusion matrix for each fold best model.')
    parser.add_argument('--load-bendr-weigths', action='store_true', help= "Load BENDR pretrained weigths, it can be encoder or encoder+context.")
    parser.add_argument('--freeze-bendr_encoder', action = 'store_true', help = "Whether to keep the encoder stage frozen. "
                       "Will only be done if bendr weigths are loaded and when using bendr encoder arch.")
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
    # TODO: Creo q lo anterior solo sale cuando se crea un encoder, hay que confirmarlo
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


            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
            validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=True)
            dataloaders = {'train': trainloader, 'valid': validloader}
        else:
            dataset = standardDataset(all_X, all_y)
            train_dataset = dataset
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
            dataloaders = {'train': dataloader}

        # Model
        #model = Net()
        '''
        model = LinearHeadBENDR_from_scratch(n_targets=data_settings['num_cls'], samples_len=samples_tlen * 256, n_chn=20,
                                    encoder_h=512, projection_head=False,
                                             # DROPOUTS
                                    enc_do=0.0, feat_do=0.0, #enc_do=0.1, feat_do=0.4,
                                    pool_length=4,
                                             # MASKS LENGHTS
                                    mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.05, mask_c_span=0.1,
                                    classifier_layers=1, return_features=True,
                                             # IF USE MASK OR NOT
                                    not_use_mask_train=True) #not_use_mask_train=False)
        # REMOVE NORMS OR INITIALIZATION CLASSIFICATION LAYER EVENTUALLY
        '''

        if args.load_bendr_weigths:
            model.load_pretrained_modules('./datasets/encoder.pt', './datasets/contextualizer.pt',
                                          freeze_encoder=args.freeze_bendr_encoder)
        if args.multi_gpu:
            model = nn.DataParallel(model)
        model = model.to(device)

        # Optimizer and Loss
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # loss weigths
        c1_train_instances = 0
        for x, y in train_dataset:
            c1_train_instances += y.item()
        c0_train_instances = len(train_dataset) - c1_train_instances
        c1_w = 1 / (c1_train_instances) * 1000  # Weigth to class 1
        c0_w = 1 / (c0_train_instances) * 1000  # Weight to class 0
        class_weights = torch.tensor([c0_w, c1_w], dtype=torch.float)

        criterion = nn.BCEWithLogitsLoss(weigths=class_weights)

        # Train
        if args.use_valid:
            best_model, curves_accs, curves_losses, train_df, valid_df, best_epoch = train_scratch_model(
                model, criterion, optimizer, dataloaders, device, num_epochs, valid_sets[fold], len(valid_dataset), args.valid_per_record)
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


