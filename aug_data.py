import numpy as np
import mne
import matplotlib.pyplot as plt
import os
import random
from utils import all_same
import eeglabio

if __name__ == '__main__':
    data_to_augment_path = './datasets/h_scz_study/'
    sbj_folder = 'SA000'
    path_of_augmented_data = './datasets/h_scz_study_AD1/'
    fortmat_type = 'set'
    segment_size_sec = 15 # in seconds
    n_artificial_records = 10

    ###############################################3
    all_sf = []    # for sanity check

    for root, _, files in os.walk(data_to_augment_path + sbj_folder):
        for file in sorted(files):

            new_raw = mne.io.read_raw_eeglab(os.path.join(root, file), preload=True)
            sf = new_raw.info['sfreq']
            all_sf.append(sf)
            segment_size_samples = int(sf * segment_size_sec)
            raw_len = new_raw._data.shape[1]
            n_segments = int(np.ceil(raw_len/segment_size_samples))

            ids_of_segments = []
            c = 0
            for i in range(n_segments-1):
                ids_of_segments.append((c,c+segment_size_samples))
                c += segment_size_samples
            ids_of_segments.append((c,raw_len))

            mne.export.export_raw('{}/{}_0{}'.format(path_of_augmented_data+sbj_folder, file[:11], file[11:]), new_raw, fmt='eeglab')
            seg_combinations = []
            for gen_data in range(n_artificial_records):
                ids_of_segments_shuffled = random.sample(ids_of_segments, len(ids_of_segments))
                while ids_of_segments_shuffled == ids_of_segments:
                    ids_of_segments_shuffled = random.sample(ids_of_segments, len(ids_of_segments))
                gen_raw = np.zeros(new_raw._data.shape)
                c = 0
                for id_seg_shuff in ids_of_segments_shuffled:
                    gen_raw[:, c:(c+ (id_seg_shuff[1] - id_seg_shuff[0]))] = new_raw._data[:, id_seg_shuff[0]:id_seg_shuff[1]]
                    c += (id_seg_shuff[1] - id_seg_shuff[0])
                assert c == raw_len
                seg_combinations.append(ids_of_segments_shuffled)
                # Save raw file
                gen_raw = mne.io.RawArray(gen_raw, mne.create_info(new_raw.info['ch_names'], new_raw.info['sfreq']))
                mne.export.export_raw('{}/{}_{}{}'.format(path_of_augmented_data+sbj_folder, file[:11], gen_data+1, file[11:]),
                    gen_raw, fmt='eeglab')

    assert all_same(all_sf) # Sanity check

    a = 0