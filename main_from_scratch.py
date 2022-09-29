import argparse
import pickle
import torch
import yaml
import os
import csv
from datasets import charge_dataset, standardDataset
from architectures import Net
from trainables import train_model, train_scratch_model
from torch.optim import lr_scheduler
from torch import nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train models from simpler to more complex.")
    parser.add_argument('--multi-gpu', action='store_true', help='Distribute model over multiple GPUs')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of dataloader workers.')
    parser.add_argument('--results-filename', default=None, help='What to name the spreadsheet produced with all '
                                                                 'final results.')
    parser.add_argument('--name-train', default=None)
    parser.add_argument('--dataset-directory', default=None,
                        help='Where is the ubication of the data samples and the information '
                             'associated to them.')
    parser.add_argument('--random-seed', default=298,
                        help='Set fixed random seed.')
    parser.add_argument('--save-models', action='store_true',
                        help='Wether to save or not the best models per CV iteration.')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('---Using ' + str(device) + 'device---')

    with open(args.dataset_directory + '/info.yml') as infile:
        data_settings = yaml.load(infile, Loader=yaml.FullLoader)

    os.makedirs('./{}_results_'.format(args.name_train) + args.results_filename + '_len{}ov{}_'.format(
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

    # Train parameters
    bs = data_settings['batch_size']
    num_cls = data_settings['num_cls']
    samples_tlen = data_settings['tlen']
    lr = data_settings['lr']
    num_epochs = data_settings['epochs']

    # Dataset and dataloaders
    dataset = standardDataset(all_X, all_y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
    dataloaders = {'train': dataloader}

    # Model
    model = Net()

    if args.multi_gpu:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # , weight_decay=0.01)

    # Train
    best_model, curves_accs, curves_losses, train_df, valid_df, best_epoch = train_scratch_model(
                                    model, criterion, optimizer, dataloaders, device, num_epochs, use_valid=False)

    train_df.to_csv("./{}_results2_{}_len{}ov{}_/train_Df_{}_lr{}bs{}_{}.csv".format(args.name_train, args.results_filename,
                                                                                      data_settings['tlen'],
                                                                                      data_settings['overlap_len'],
                                                                                      args.dataset_directory.split('/')[
                                                                                          -1], lr, bs,
                                                                                      data_settings['target_feature']))
    valid_df.to_csv("./{}_results2_{}_len{}ov{}_/valid_Df_{}_lr{}bs{}_{}.csv".format(args.name_train, args.results_filename,
                                                                                      data_settings['tlen'],
                                                                                      data_settings['overlap_len'],
                                                                                      args.dataset_directory.split('/')[
                                                                                          -1], lr, bs,
                                                                                      data_settings['target_feature']))
    with open('./{}_results2_{}_len{}ov{}_/mean_loss_curves_{}_lr{}bs{}_{}.pkl'.format(args.name_train, args.results_filename,
                                                                                       data_settings['tlen'],
                                                                                       data_settings['overlap_len'],
                                                                                       args.dataset_directory.split(
                                                                                           '/')[-1],
                                                                                       lr, bs,
                                                                                       data_settings['target_feature']),
              'wb') as f:
        pickle.dump(curves_losses, f)
    with open('./{}_results2_{}_len{}ov{}_/mean_acc_curves_{}_lr{}bs{}_{}.pkl'.format(args.name_train, args.results_filename,
                                                                                      data_settings['tlen'],
                                                                                      data_settings['overlap_len'],
                                                                                      args.dataset_directory.split('/')[
                                                                                          -1],
                                                                                      lr, bs,
                                                                                      data_settings['target_feature']),
              'wb') as f:
        pickle.dump(curves_accs, f)

    if args.save_models:
        torch.save(best_model.state_dict(),
                   './{}_results2_{}_len{}ov{}_/best_model_{}_lr{}bs{}_{}.pt'.format(args.results_filename,
                                                                                     data_settings['tlen'],
                                                                                     data_settings['overlap_len'],
                                                                                     args.dataset_directory.split('/')[
                                                                                         -1], lr, bs,
                                                                                     data_settings['target_feature']))

    print('Best epoch for each of the cross-validations iterations:\n{}'.format(best_epoch))
    a = 0

    '''
    # TODO: WHATS IS THIS? Receptive field: 143 samples | Downsampled by 96 | Overlap of 47 samples | 106 encoded samples/trial
    with open('./{}'.format(data_settings['valid_sets_path']), newline='') as f:
        reader = csv.reader(f)
        valid_sets = list(reader)
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

        dataset_sizes = {x: len(dataloaders[x]) * bs for x in ['train', 'valid']}
    '''