import argparse
import pickle
import torch
import yaml
import os
import csv
from datasets import charge_dataset, standardDataset, recInfoDataset
from architectures import Net, LongLinearHeadBENDR_from_scratch, LinearHeadBENDR_from_scratch, BENDRClassification_from_scratch
from trainables import train_scratch_model, train_scratch_model_no_valid, train_scratch_model_per_epoch
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch import nn
from utils import comp_confusion_matrix, MODEL_CHOICES, TASK_CHOICES, ClipLogistCELoss, SCELoss
from mlxtend.plotting import plot_confusion_matrix
from torch.optim import lr_scheduler

if __name__ == '__main__':
    # Arguments and preliminaries
    parser = argparse.ArgumentParser(description="Train models from simpler to more complex.")
    parser.add_argument('model', choices=MODEL_CHOICES)
    parser.add_argument('task', choices=TASK_CHOICES)
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
    parser.add_argument('--load-bendr-weigths', action='store_true', help= "Load BENDR pretrained weigths, it can be encoder or encoder+context.")
    parser.add_argument('--freeze-bendr-encoder', action = 'store_true', help = "Whether to keep the encoder stage frozen. "
                       "Will only be done if bendr weigths are loaded and when using bendr encoder arch.")
    parser.add_argument('--ponderate-loss', action='store_true', help='Add weigths to the loss function.')
    parser.add_argument('--extra-aug', action='store_true', help='Ponderate the input signal with a certain probability'
                                                                 'to avoid overfitting.')
    parser.add_argument('--freeze-first-layers', action='store_true', help = "Whether to keep the 3 first layers of the encoder stage frozen. "
                            "Will only be done if bendr weigths are loaded and when using bendr encoder arch.")
    parser.add_argument('--input-sfreq', default=256,
                        help='Sampling frequency to transform the EEG to input to the network.', type=int)
    parser.add_argument('--own-init', default=None,
                        help="Load my own pretrained weigths, for the complete model.")
    parser.add_argument('--loss-per-epoch', action='store_true')

    parser.add_argument('--n-outputs', default=1, help='Output dimension of the network, it also defines if you have tu use softmax or sigmoid', type=int)
    parser.add_argument('--split-criterion', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('---Using ' + str(device) + 'device---')

    with open(args.dataset_directory + '/info.yml') as infile:
        data_settings = yaml.load(infile, Loader=yaml.FullLoader)

    if args.use_valid:
        with open('./{}'.format(data_settings['valid_sets_path']), newline='') as f:
            reader = csv.reader(f)
            valid_sets = list(reader)
    os.makedirs('./{}-{}-rslts_'.format(args.model, args.task) + args.results_filename + '_len{}ov{}_'.format(
                                    data_settings['tlen'], data_settings['overlap_len']), exist_ok=True)

    if args.task == 'regressor':
        target_feature = data_settings['target_feature'] + '_norm'
    else:
        target_feature = data_settings['target_feature']

    # Load dataset
    # list of len: n_records
    # each list[n] is of dim [n_segments, 20 , len_segments (256*tlen)]
    array_epochs_all_records, sorted_record_names = charge_dataset(directory=args.dataset_directory,
                                                  tlen=data_settings['tlen'], overlap=data_settings['overlap_len'],
                                                  data_max=data_settings['data_max'], data_min=data_settings['data_min'],
                                                  chns_consider=data_settings['chns_to_consider'],
                                                  labels_path=data_settings['labels_path'], target_f=target_feature,
                                                  apply_winsor=data_settings['apply_winsorising'], new_sfreq=args.input_sfreq)

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

            if args.valid_per_record and args.loss_per_epoch:
                trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=False, drop_last=True)
                print('Train set were not shuffled!')
            else:
                trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
            if args.valid_per_record:
                validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)
                print('Validation set were not shuffled!')
            else:
                validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=True)
            dataloaders = {'train': trainloader, 'valid': validloader}
            if args.ponderate_loss:
                # loss weigths
                c1_train_instances = 0
                for x, y, _ in train_dataset:
                    c1_train_instances += y.item()
                c0_train_instances = len(train_dataset) - c1_train_instances
                class_weigth = c0_train_instances/c1_train_instances
        else:
            dataset = standardDataset(all_X, all_y)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
            dataloaders = {'train': dataloader}
            if args.ponderate_loss:
                # loss weigths
                c1_train_instances = 0
                for x, y in dataset:
                    c1_train_instances += y.item()
                c0_train_instances = len(train_dataset) - c1_train_instances
                class_weigth = c0_train_instances / c1_train_instances

        # Model
        if args.model == 'linear':
            model = LinearHeadBENDR_from_scratch(args.n_outputs, samples_len=samples_tlen * args.input_sfreq, n_chn=20,
                                        encoder_h=512, projection_head=False,
                                                 # DROPOUTS
                                        enc_do=0.3, feat_do=0.7, #enc_do=0.1, feat_do=0.4,
                                        pool_length=4,
                                                 # MASKS LENGHTS
                                        mask_p_t= 0.01,
                                        mask_p_c=0.005,
                                        mask_t_span=0.05,
                                        mask_c_span=0.1,
                                        classifier_layers=1, return_features=False,
                                                 # IF USE MASK OR NOT
                                        not_use_mask_train=False)
        elif args.model == 'longlinear':
            model = LongLinearHeadBENDR_from_scratch(1, samples_len=samples_tlen * args.input_sfreq, n_chn=20,
                                                 encoder_h=512, projection_head=False,
                                                 # DROPOUTS
                                                 enc_do=0.5, feat_do=0.7,  # enc_do=0.1, feat_do=0.4,
                                                 pool_length=4,
                                                 # MASKS LENGHTS
                                                 mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.05, mask_c_span=0.1,
                                                 classifier_layers=1, return_features=False,
                                                 # IF USE MASK OR NOT
                                                 not_use_mask_train=False)
        elif args.model == 'BENDR':
            model = BENDRClassification_from_scratch(1, samples_len=samples_tlen * args.input_sfreq, n_chn=20,
                                        encoder_h=512,
                                        contextualizer_hidden=3076, projection_head=False,
                                        new_projection_layers=0, dropout=0., trial_embeddings=None, layer_drop=0,
                                        keep_layers=None,
                                        mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.1, mask_c_span=0.1,
                                        multi_gpu=False, return_features=False)
        else:
            assert 1 == 0

        assert not ((args.load_bendr_weigths == True) and (args.own_init is not None))
        if args.load_bendr_weigths:
            model.load_pretrained_modules('../BENDR_datasets/encoder.pt', '../BENDR_datasets/contextualizer.pt',
                                          freeze_encoder=args.freeze_bendr_encoder, device=device)
            if not args.freeze_bendr_encoder:
                if args.freeze_first_layers:
                    model.freeze_first_layers(layers_to_freeze='first')
        elif args.own_init is not None:
            #model.load_state_dict(torch.load(args.own_init, map_location=device))

            if not args.model == 'longlinear':
                if args.model == 'BENDR':
                    states_dicts_path = args.own_init.split('///')
                    assert len(states_dicts_path) == 2
                    orig_ws = torch.load(states_dicts_path[0], map_location=device)
                    new_ws = OrderedDict()
                    for k, v in orig_ws.items():
                        if not ('classifier.1' in k):
                            new_ws[k] = v
                    model.load_state_dict(new_ws, strict=False)
                    model.load_state_dict(torch.load(states_dicts_path[1], map_location=device), strict=False)
                else:
                    model.load_state_dict(torch.load(args.own_init, map_location=device))
            else:
                orig_ws = torch.load(args.own_init, map_location=device)
                new_ws = OrderedDict()
                for k, v in orig_ws.items():
                    if k[:10] != 'classifier':
                        new_ws[k] = v
                model.load_state_dict(new_ws, strict=False)

            for param in model.parameters():
                param.requires_grad = True
            if not args.model == 'BENDR':
                model.enc_augment.freeze_enc_aug(freeze=False)
            else:
                model.contextualizer.mask_replacement.requires_grad = True
            if args.freeze_first_layers:
                model.freeze_first_layers(layers_to_freeze='first')


        if args.multi_gpu:
            model = nn.DataParallel(model)
        model = model.to(device)

        # Optimizer and Loss
        if args.freeze_first_layers:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        if args.task == 'regressor':
            criterion = nn.L1Loss()  #MSELoss()
        else:
            if args.ponderate_loss:
                criterion = nn.BCEWithLogitsLoss(weigth=class_weigth)
            else:
                if args.n_outputs > 1:
                    loss_dict = dict()
                    #loss_dict['alpha'] = 10
                    #loss_dict['beta'] = 0.5
                    #loss_dict['labels'] = 2
                    loss_dict['tau'] = 0.1
                    #criterion = SCELoss(loss_dict)
                    #criterion = ClipLogistCELoss(loss_dict) #nn.BCEWithLogitsLoss()
                    criterion = ClipLogistCELoss(loss_dict) #nn.BCEWithLogitsLoss() #
                    #criterion0 = nn.CrossEntropyLoss()
                    #criterion1 = nn.CrossEntropyLoss() #label_smoothing=0.9) # label_smoothing=0.7)
                    criterion0 = None
                    criterion1 = None
                    assert (criterion is None and ((criterion0 is not None) and (criterion1 is not None))) or (criterion is not None and ((criterion0 is None) and (criterion1 is None)))
                else:
                    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.5))
                    criterion0 = None
                    criterion1 = None

        # Train
        if args.task == 'classifier':
            if args.use_valid:
                if args.loss_per_epoch:
                    best_model, curves_accs, curves_losses, best_epoch, curves_accs_per_sample = train_scratch_model_per_epoch(
                        model, criterion0, criterion1, optimizer, dataloaders, device, num_epochs,
                        valid_sets[fold], len(valid_dataset), args.valid_per_record, args.extra_aug,
                        use_clip_grad=False, n_outputs=args.n_outputs)
                else:
                    best_model, curves_accs, curves_losses, train_df, valid_df, best_epoch = train_scratch_model(
                                                    model, criterion, optimizer, dataloaders, device, num_epochs,
                                                    valid_sets[fold], len(valid_dataset), args.valid_per_record, args.extra_aug, type_task=args.task, use_clip_grad=False, n_outputs=args.n_outputs,
                    criterion0=criterion0, criterion1=criterion1, split_criterion=args.split_criterion, scheduler=exp_lr_scheduler)
            else:
                best_model, curves_accs, curves_losses, train_df, valid_df, best_epoch = train_scratch_model_no_valid(
                                                model, criterion, optimizer, dataloaders, device, num_epochs, type_task=args.task, use_clip_grad=False, n_outputs = args.n_outputs)
        else:
            if args.use_valid:
                best_model, curves_losses, train_df, valid_df, best_epoch = train_scratch_model(
                                                model, criterion, optimizer, dataloaders, device, num_epochs,
                                                valid_sets[fold], len(valid_dataset), args.valid_per_record, args.extra_aug, type_task=args.task, use_clip_grad=False, n_outputs=args.n_outputs)
            else:
                best_model, curves_losses, train_df, valid_df, best_epoch = train_scratch_model_no_valid(
                                                model, criterion, optimizer, dataloaders, device, num_epochs, type_task=args.task, use_clip_grad=False)

        best_epoch_fold.append(best_epoch)
        if not args.loss_per_epoch:
            train_df.to_csv("./{}-{}-rslts_{}_len{}ov{}_/train_Df_f{}_{}_lr{}bs{}.csv".format(args.model, args.task, args.results_filename, data_settings['tlen'], data_settings['overlap_len'],
                                                                                           fold, args.dataset_directory.split('/')[-1], lr, bs))
            valid_df.to_csv("./{}-{}-rslts_{}_len{}ov{}_/valid_Df_f{}_{}_lr{}bs{}.csv".format(args.model, args.task, args.results_filename, data_settings['tlen'], data_settings['overlap_len'],
                                                                                           fold, args.dataset_directory.split('/')[-1], lr, bs))
        with open('./{}-{}-rslts_{}_len{}ov{}_/mean_loss_curves_f{}_{}_lr{}bs{}.pkl'.format(args.model, args.task, args.results_filename, data_settings['tlen'], data_settings['overlap_len'],
                                                                                       fold, args.dataset_directory.split('/')[-1], lr, bs), 'wb') as f:
            pickle.dump(curves_losses, f)
        if args.task == 'classifier':
            with open('./{}-{}-rslts_{}_len{}ov{}_/mean_acc_curves_f{}_{}_lr{}bs{}.pkl'.format(args.model, args.task, args.results_filename, data_settings['tlen'], data_settings['overlap_len'],
                                                                                            fold, args.dataset_directory.split('/')[-1], lr, bs), 'wb') as f:
                pickle.dump(curves_accs, f)
            if args.loss_per_epoch:
                with open(
                        './{}-{}-rslts_{}_len{}ov{}_/mean_acc_per_sample_curves_f{}_{}_lr{}bs{}.pkl'.format(args.model, args.task,
                                                                                                 args.results_filename,
                                                                                                 data_settings['tlen'],
                                                                                                 data_settings[
                                                                                                     'overlap_len'],
                                                                                                 fold,
                                                                                                 args.dataset_directory.split(
                                                                                                     '/')[-1], lr, bs),
                        'wb') as f:
                    pickle.dump(curves_accs_per_sample, f)

        if args.save_models:
            torch.save(best_model.state_dict(), './{}-{}-rslts_{}_len{}ov{}_/best_model_f{}_{}_lr{}bs{}.pt'.format(args.model, args.task, args.results_filename, data_settings['tlen'],
                                                                                        data_settings['overlap_len'], fold, args.dataset_directory.split('/')[-1], lr, bs))
    print('Best epoch for each of the cross-validations iterations:\n{}'.format(best_epoch_fold))
    plt.show()
    a = 0
