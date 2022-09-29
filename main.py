import argparse
import pickle
import torch
import yaml
import os
import csv
from datasets import charge_dataset, standardDataset
from architectures import MODEL_CHOICES, LinearHeadBENDR, BENDRClassification
from trainables import train_model
from torch.optim import lr_scheduler
from torch import nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tunes BENDER models.")
    parser.add_argument('model', choices=MODEL_CHOICES)
    parser.add_argument('--freeze-encoder', action='store_true', help="Whether to keep the encoder stage frozen. "
                                                                      "Will only be done if not randomly initialized.")
    parser.add_argument('--random-init', action='store_true', help='Randomly initialized BENDR for comparison.')
    # TODO: ?
    parser.add_argument('--multi-gpu', action='store_true', help='Distribute BENDR over multiple GPUs')
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
    parser.add_argument('--not-use-mask', action='store_true',
                       help='Wheter to use or not mask in the train stage.')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('---Using ' + str(device) + 'device---')

    with open(args.dataset_directory + '/info.yml') as infile:
        data_settings = yaml.load(infile, Loader=yaml.FullLoader)

    os.makedirs('./results_' + args.results_filename + '_len{}ov{}_'.format(
                                    data_settings['tlen'], data_settings['overlap_len']), exist_ok=True)

    # Load dataset
    # list of len = n_records
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

    with open('./{}'.format(data_settings['valid_sets_path']), newline='') as f:
        reader = csv.reader(f)
        valid_sets = list(reader)
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
    for fold in range(len(valid_sets)):
        valid_ids = [i for i in range(len(sorted_record_names)) if (sorted_record_names[i][:11] in valid_sets[fold])]
        train_ids = [i for i in range(len(sorted_record_names)) if (sorted_record_names[i][:11] not in valid_sets[fold])]
        print('FOLD n°{}'.format(fold))
        print('-----------------------')

        # Split data
        train_dataset = standardDataset(all_X[train_ids], all_y[train_ids])
        valid_dataset = standardDataset(all_X[valid_ids], all_y[valid_ids])

        # Dataloaders
        to_drop_last = True if args.model=='linear' else False
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=to_drop_last)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=True, drop_last=to_drop_last)
        dataloaders = {'train': trainloader, 'valid': validloader}


        # MODEL
        if args.model == MODEL_CHOICES[0]:
            model = BENDRClassification(targets=data_settings['num_cls'], samples_len=samples_tlen * 256, n_chn=20,
                                        encoder_h=512, contextualizer_hidden=3076, projection_head=False,
                                        new_projection_layers=0, dropout=0., trial_embeddings=None, layer_drop=0,
                                        keep_layers=None, mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.1, mask_c_span=0.1,
                                        multi_gpu=False, return_features=True, regression_option=False)
        else:
            model = LinearHeadBENDR(n_targets=data_settings['num_cls'], samples_len=samples_tlen * 256, n_chn=20,
                                    encoder_h=512, projection_head=False, enc_do=0.1, feat_do=0.4, pool_length=4,
                                    mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.05, mask_c_span=0.1,
                                    classifier_layers=1, return_features=True, not_use_mask_train=args.not_use_mask)

        if not args.random_init:
            model.load_pretrained_modules('./datasets/encoder.pt', './datasets/contextualizer.pt',
                                          freeze_encoder=args.freeze_encoder)
        if args.multi_gpu:
            model = nn.DataParallel(model)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

        c1_train_instances = 0
        for x,y in train_dataset:
            c1_train_instances += y.item()
        c0_train_instances = len(train_dataset) - c1_train_instances
        c1_w = 1 / (c1_train_instances) * 1000  # Weigth to class 1
        c0_w = 1 / (c0_train_instances) * 1000  # Weight to class 0
        class_weights = torch.tensor([c0_w, c1_w], dtype=torch.float)

        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        sched = lr_scheduler.OneCycleLR(optimizer, lr, epochs=num_epochs, steps_per_epoch=len(dataloaders['train']),
                                        pct_start=0.3, last_epoch=-1)


        ##################################
        best_model, accs_curves, loss_curves, train_log, valid_log, best_epoch = train_model(model=model, criterion=criterion,
                                                                            optimizer=optimizer, scheduler=sched,
                                                                            dataloaders=dataloaders,
                                                                            device=device,
                                                                            num_epochs=num_epochs)
        best_epoch_fold.append(best_epoch)
        train_log.to_csv("./results_{}_len{}ov{}_/train_Df_f{}_{}_lr{}bs{}_{}.csv".format(args.results_filename,
                                                                                data_settings['tlen'], data_settings['overlap_len'], fold,
                                                                                args.dataset_directory.split('/')[-1], lr, bs,
                                                                                data_settings['target_feature']))
        valid_log.to_csv("./results_{}_len{}ov{}_/valid_Df_f{}_{}_lr{}bs{}_{}.csv".format(args.results_filename,
                                                                                data_settings['tlen'], data_settings['overlap_len'], fold,
                                                                                args.dataset_directory.split('/')[-1], lr, bs,
                                                                                data_settings['target_feature']))
        with open('./results_{}_len{}ov{}_/mean_loss_curves_f{}_{}_lr{}bs{}_{}.pkl'.format(args.results_filename,
                                                                                data_settings['tlen'], data_settings['overlap_len'], fold,
                                                                                args.dataset_directory.split('/')[-1],
                                                                                lr, bs, data_settings['target_feature']), 'wb') as f:
            pickle.dump(loss_curves, f)
        with open('./results_{}_len{}ov{}_/mean_acc_curves_f{}_{}_lr{}bs{}_{}.pkl'.format(args.results_filename,
                                                                                data_settings['tlen'], data_settings['overlap_len'], fold,
                                                                                args.dataset_directory.split('/')[-1],
                                                                                lr, bs, data_settings['target_feature']), 'wb') as f:
            pickle.dump(accs_curves, f)

        if args.save_models:
            torch.save(best_model.state_dict(), './results_{}_len{}ov{}_/best_model_f{}_{}_lr{}bs{}_{}.pt'.format(args.results_filename,
                                                                                     data_settings['tlen'], data_settings['overlap_len'], fold,
                                                                                     args.dataset_directory.split('/')[-1], lr, bs,
                                                                                     data_settings['target_feature']))


    print('Best epoch for each of the cross-validations iterations:\n{}'.format(best_epoch_fold))

    a = 0