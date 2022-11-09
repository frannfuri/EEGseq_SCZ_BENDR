import torch
import mne
import yaml
import argparse
from datasets import charge_dataset, recInfoDataset
from architectures import LinearHeadBENDR_from_scratch, ConvEncoderBENDR_from_scratch, EncodingAugment_from_scratch
from torch import nn

if __name__ == '__main__':
    # Arguments and preliminaries
    parser = argparse.ArgumentParser(description="Train models from simpler to more complex.")
    parser.add_argument('--multi-gpu', action='store_true', help='Distribute model over multiple GPUs')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of dataloader workers.')
    parser.add_argument('--dataset-directory', default=None,
                        help='Where is the ubication of the data samples and the information '
                             'associated to them.')
    parser.add_argument('--random-seed', default=298,
                        help='Set fixed random seed.')
    parser.add_argument('--load-bendr-weigths', action='store_true',
                        help="Load BENDR pretrained weigths, it can be encoder or encoder+context.")
    parser.add_argument('--weigths-dir', default=None,
                        help="Where is the ubication of the model weigths to load into the model. "
                             "Will only be done if not load_bendr_weigths.")
    parser.add_argument('--extra-aug', action='store_true', help='Ponderate the input signal with a certain probability'
                                                                 'to avoid overfitting.')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('---Using ' + str(device) + 'device---')

    with open(args.dataset_directory + '/info.yml') as infile:
        data_settings = yaml.load(infile, Loader=yaml.FullLoader)

    # Load dataset
    # list of len: n_records
    # each list[n] is of dim ([n_segments, 20 , len_segments (256*tlen)], [n_segments])
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

    the_dataset = recInfoDataset(all_X, all_y, sorted_record_names) #[sorted_record_names[i] for i in train_ids])

    the_loader = torch.utils.data.DataLoader(the_dataset, batch_size=1, shuffle=False)

    print(the_dataset[20])
    # Set fixed random number seed
    torch.manual_seed(args.random_seed)
    print('-------------------------------------------')

    # Train parameters
    num_cls = data_settings['num_cls']
    samples_tlen = data_settings['tlen']


    encoder_dropout = 0.3 #0.1
    #features_dropout = 0.7 #0.4
    # Model
    model_encoder = ConvEncoderBENDR_from_scratch(20, encoder_h=512, projection_head=False, dropout=encoder_dropout)
    model_encoding = EncodingAugment_from_scratch(512, 0.01, 0.005, mask_c_span=0.1, mask_t_span=0.05, use_mask_train=True)
    '''
    model = LinearHeadBENDR_from_scratch(1, samples_len=samples_tlen * 256, n_chn=20,
                                         encoder_h=512, projection_head=False,
                                         # DROPOUTS
                                         enc_do=0.3, feat_do=0.7,  # enc_do=0.1, feat_do=0.4,
                                         pool_length=4,
                                         # MASKS LENGHTS
                                         mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.05, mask_c_span=0.1,
                                         classifier_layers=1, return_features=False,
                                         # IF USE MASK OR NOT
                                         not_use_mask_train=False)
    '''

    if args.load_bendr_weigths:
        model_encoder.load_state_dict(torch.load('./datasets/encoder.pt', map_location=device), strict=True)
        model_encoding.load_state_dict(torch.load('./datasets/contextualizer.pt', map_location=device), strict=False)
        #model.load_pretrained_modules('./datasets/encoder.pt', './datasets/contextualizer.pt',
        #                              freeze_encoder=True, dev=device)
        print('Loaded BENDR weigths.')
    else:
        #if args.weigths_dir is not None:
        #    model.load_state_dict(torch.load( args.weigths_dir, map_location=device))
        #    print('Loaded custom weigths from \'{}\'.'.format(args.weigths_dir.split('/')[-1]))
        #else:
        print('Not weigths loaded!')
    if args.multi_gpu:
        model_encoder = nn.DataParallel(model_encoder)
        model_encoding = nn.DataParallel(model_encoding)
    model_encoder = model_encoder.to(device)
    model_encoding = model_encoding.to(device)

    a = 0


