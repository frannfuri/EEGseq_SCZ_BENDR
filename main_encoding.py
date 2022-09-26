import torch
import yaml
import numpy as np
import os
from datasets import charge_dataset, standardDataset
from architectures import ConvEncoderBENDR, EncodingAugment, Flatten
from torch import nn


if __name__ == '__main__':
    dataset_directory = './datasets/decomp_study_SA047_AD1'
    encoder_weigths = './datasets/encoder.pt'
    freeze_encoder = True
    use_encoding_augment = True
    if use_encoding_augment:
        encoding_aug_weigths = './datasets/contextualizer.pt'
    random_seed = 298


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(dataset_directory + '/info.yml') as infile:
        data_settings = yaml.load(infile, Loader=yaml.FullLoader)

    # Load dataset
    # list of len = n_records
    # each list[n] is of dim [n_segments, 20 , len_segments (256*tlen)]
    array_epochs_all_records, sorted_record_names = charge_dataset(directory=dataset_directory,
                                                                   tlen=data_settings['tlen'],
                                                                   overlap=data_settings['overlap_len'],
                                                                   data_max=data_settings['data_max'],
                                                                   data_min=data_settings['data_min'],
                                                                   chns_consider=data_settings['chns_to_consider'],
                                                                   labels_path=data_settings['labels_path'],
                                                                   target_f=data_settings['target_feature'],
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

    dataset = standardDataset(all_X, all_y)

    # Set fixed random number seed
    torch.manual_seed(random_seed)
    print('-------------------------------------------')


    # Model parameters
    num_cls = data_settings['num_cls']
    samples_tlen = data_settings['tlen']

    # Convolutional Encoder
    model_encoder = ConvEncoderBENDR(20, encoder_h=512, enc_width=(3, 2, 2, 2, 2, 2), dropout=0., #dropout=0.1,
                             projection_head=False, enc_downsample=(3, 2, 2, 2, 2, 2))
    # Load pretrained weights and freeze (or not) the weights
    model_encoder.load(encoder_weigths, strict=False) # TODO: What happend if strict=True?
    model_encoder.freeze_features(not freeze_encoder)

    samples_len = 256*samples_tlen
    encoded_samples = model_encoder.downsampling_factor(samples_len)
    encoder_h = 512
    mask_c_span = 0.1
    mask_t_span = 0.05
    mask_t_span = mask_t_span if mask_t_span > 1 else int(mask_t_span * encoded_samples)
    # Important for short things like P300
    mask_t_span = 0 if encoded_samples < 2 else mask_t_span
    mask_c_span = mask_c_span if mask_c_span > 1 else int(mask_c_span * encoder_h)

    # Encoding Augment (relative position + input conditioning)
    model_encoding_aug = EncodingAugment(512, mask_p_t=0.01, mask_p_c=0.005, mask_c_span=mask_c_span,
                                           mask_t_span=mask_t_span)
    model_encoding_aug.init_from_contextualizer(encoding_aug_weigths)

    summarizer = nn.AdaptiveAvgPool1d(4)
    flatten = Flatten()

    x_encoded = []
    Ys = []
    for x,y in dataset:
        x = x.unsqueeze(0)
        # dim (1, 20, 10240)
        x = model_encoder(x)
        # dim (1, 512, 107)
        x = model_encoding_aug(x)
        # dim (1, 1536, 107)
        x = flatten(summarizer(x)) # dim (1, 1536, 4)
        # dim (1, 6144)
        x_encoded.append(x.squeeze().detach().cpu().numpy())
        Ys.append(y.item())
    x_encoded = np.array(x_encoded)
    Ys = np.array(Ys)

    np.save('./x_encoded.npy', x_encoded)
    np.save('./Ys.npy', Ys)
    a = 0

