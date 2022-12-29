import os
import mne
import torch
import numpy as np
import pandas as pd

from channels import stringify_channel_mapping
from collections import OrderedDict
from torch.utils.data import Dataset as TorchDataset
from utils import InstanceTransform, MappingDeep1010, TemporalInterpolation, To1020
from extras import *

def eeglab_set_to_array_epochs(path, label, tlen, overlap, data_max, data_min,
                        chns_consider, apply_winsor, new_sfreq):
    new_raw = mne.io.read_raw_eeglab(path, preload=True)

    # Consider only some channels
    ch_to_remove = list(set(new_raw.ch_names) - set(chns_consider))
    new_raw.drop_channels(ch_to_remove)

    print('Procesing the epochs...')
    epochs_one_rec_instance = EEGrecord_instance_divide_in_epochs(raw=new_raw, data_max=data_max, data_min=data_min, tlen=tlen,
                                                         overlap=overlap, apply_winsor=apply_winsor, label=label, new_sfreq=new_sfreq)
    is_first_epoch = True
    for ep_id in range(0, len(epochs_one_rec_instance)):
        if is_first_epoch:
            # Consider only 3 last channels (Fz Cz Pz)
            all_eps_x = torch.unsqueeze(epochs_one_rec_instance.__getitem__(ep_id)[0], dim=0)
            all_eps_y = torch.unsqueeze(epochs_one_rec_instance.__getitem__(ep_id)[1], dim=0)
            is_first_epoch = False
        else:
            all_eps_x = torch.cat([all_eps_x, torch.unsqueeze(epochs_one_rec_instance.__getitem__(ep_id)[0], dim=0)], 0)
            all_eps_y = torch.cat([all_eps_y, torch.unsqueeze(epochs_one_rec_instance.__getitem__(ep_id)[1], dim=0)], 0)
    assert len(all_eps_x) == len(all_eps_y)
    return all_eps_x, all_eps_y, len(all_eps_y)*[new_raw.filenames[0].split('/')[-1][:11]]




def charge_dataset(directory, tlen, overlap, data_max, data_min, chns_consider, labels_path, target_f, apply_winsor, new_sfreq):
    sorted_record_names = []
    array_epochs_all_records = []
    for root, folders, _ in os.walk(directory):
        for fold_day in sorted(folders):
            for root_day, _, files in os.walk(os.path.join(root, fold_day)):
                for file in sorted(files):
                    if file.endswith('set'): # Only read records in .set format
                        print('================ Processing record {} ================'.format(file))
                        target_info = pd.read_csv('{}/{}_labels.csv'.format(labels_path, file[:5]),
                                                  index_col=0, decimal=',')
                        target_info = target_info.to_dict()
                        target_info = target_info[target_f]
                        label = target_info[file[:11]]
                        if isinstance(label, str):
                            label = float(label)
                        array_epochs_X_one_rec, array_epochs_Y_one_rec, epochs_names = eeglab_set_to_array_epochs(path=os.path.join(root_day, file),
                                                                        label=label,
                                                                        tlen=tlen, overlap=overlap, data_max=data_max,
                                                                        data_min=data_min, chns_consider=chns_consider,
                                                                        apply_winsor=apply_winsor, new_sfreq=new_sfreq)
                        sorted_record_names = [*sorted_record_names, *epochs_names]
                        array_epochs_all_records.append((array_epochs_X_one_rec, array_epochs_Y_one_rec))
    return array_epochs_all_records, sorted_record_names

class standardDataset(TorchDataset):
    def __init__(self, X, y):
        if X.shape[0] != y.shape[0]:
            print('First dimesion of X and y must be the same')
            return

        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        Xi = self.X[idx]
        yi = self.y[idx]
        # TODO: Is necesary to unsqueeze Xi (in dim 0) ??
        return Xi.float(), yi.float()

class recInfoDataset(TorchDataset):
    def __init__(self, X, y, rec_info):
        if X.shape[0] != y.shape[0] or X.shape[0] != len(rec_info):
            print('First dimesion of X and y must be the same')
            return

        self.X = X
        self.y = y
        self.rec_info = rec_info

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        Xi = self.X[idx]
        yi = self.y[idx]
        rec_info_i = self.rec_info[idx]
        # TODO: Is necesary to unsqueeze Xi (in dim 0) ??
        return Xi.float(), yi.float(), rec_info_i


class EEGrecord_instance_divide_in_epochs(TorchDataset):
    def __init__(self, raw: mne.io.Raw, data_max, data_min, tlen, overlap, apply_winsor,
                 tmin=0, new_sfreq=256, label=None):
        self.filename = raw.filenames[0].split('/')[-1]
        self.orig_sfreq = raw.info['sfreq']
        self.new_sfreq = new_sfreq
        self.ch_names = raw.ch_names
        self.apply_winsor = apply_winsor
        self.label = label

        ch_list = []
        for i in raw.ch_names:
            ch_list.append([i, '2'])
        self.ch_list = np.array(ch_list, dtype='<U21')

        # Segment the recording
        self.epochs = mne.make_fixed_length_epochs(raw, duration=tlen, overlap=overlap)
        self.epochs.drop_bad()

        self.tlen = tlen
        self.transforms = list()
        self._different_deep1010s = list()
        self.return_mask = False
        self.data_max = data_max
        self.data_min = data_min

        # Maps particular channels to a consistent index for each loaded trial
        xform = MappingDeep1010(channels=self.ch_list, data_max=data_max, data_min=data_min,
                                return_mask=self.return_mask)
        self.add_transform(xform)
        self._add_deep1010(self.ch_names, xform.mapping.numpy(), [])
        self.new_sequence_len = int(tlen * self.new_sfreq)

        # Over- or Under- sampling to match the desired sampling freq
        """
        WARNING: if downsample the signal below a suitable Nyquist freq given the low-pass filtering
        of the original singal, it will cause aliasing artifacts (ruined data)
        """
        self.add_transform(TemporalInterpolation(self.new_sequence_len, new_sfreq=self.new_sfreq))

        print("Constructed {} channel maps".format(len(self._different_deep1010s)))
        for names, deep_mapping, unused, count in self._different_deep1010s:
            print('=' * 20)
            print("Used by {} recordings:".format(count))
            print(stringify_channel_mapping(names, deep_mapping))
            print('-' * 20)
            print("Excluded {}".format(unused))
            print('=' * 20)

        # Only conserve 19 channels of the 10/20 International System + 1 scale channel
        self.add_transform(To1020())
        self._safe_mode = False

    def __getitem__(self, index):
        ep = self.epochs[index]

        x = ep.get_data()
        if len(x.shape) != 3 or 0 in x.shape:
            assert 1 == 0
            print("I don't know why: This  `filename` index{}/{}".format(index, len(self)))
            print(self.epochs.info['description'])
            print("Using trial {} in place for now...".format(index - 1))
            return self.__getitem__(index - 1)


        # 3 MAD threshold Winsorising
        x = torch.from_numpy(x.squeeze(0)).float()
        if self.apply_winsor:
            for i in range(x.shape[-2]):
                assert len(x.shape) == 2
                mad = MAD(x[i, :])
                med = np.median(x[i, :])
                # Winsorising
                x[i, :] = np.clip(x[i, :], med - 3 * mad, med + 3 * mad)

        #y = torch.tensor(ep.events[0, -1]).squeeze().long()
        if isinstance(self.label, int):
            y = torch.tensor(self.label).squeeze().long()
        else:
            y = torch.tensor(self.label).squeeze().float()
        return self._execute_transforms(x, y)

    def __len__(self):
        return len(self.epochs)

    def __str__(self):
        return "{} trials | {} transforms".format(len(self), len(self.transforms))

    def event_mapping(self):
        """
        Maps the labels returned by this to the events as recorded in the original annotations or stim channel.
        Returns
        -------
        mapping : dict
        Keys are the class labels used by this object, values are the original event signifier.
        """
        return self.epoch_codes_to_class_labels

    #def get_targets(self):
    #    return np.apply_along_axis(lambda x: self.epoch_codes_to_class_labels[x[0]], 1,
    #                               self.epochs.events[list(range(len(self.epochs))), -1, np.newaxis]).squeeze()

    def add_transform(self, transform_item):
        """
        Add a transformation that is applied to every fetched item in the dataset
        Parameters
        ----------
        transform : BaseTransform
                    For each item retrieved by __getitem__, transform is called to modify that item.
        """
        if isinstance(transform_item, InstanceTransform):
            self.transforms.append(transform_item)

    def _add_deep1010(self, ch_names: list, deep1010map: np.ndarray, unused_ch):
        for i, (old_names, old_map, unused_ch, count) in enumerate(self._different_deep1010s):
            if np.all(deep1010map == old_map):
                self._different_deep1010s[i] = (old_names, old_map, unused_ch, count + 1)
                return
        self._different_deep1010s.append((ch_names, deep1010map, unused_ch, 1))

    def _execute_transforms(self, *x):
        for transform in self.transforms:
            assert isinstance(transform, InstanceTransform)
            if transform.only_trial_data:
                new_x = transform(x[0])
                if isinstance(new_x, (list, tuple)):
                    x = (*new_x, *x[1:])
                else:
                    x = (new_x, *x[1:])
            else:
                x = transform(*x)

            if self._safe_mode:
                for i in range(len(x)):
                    if torch.any(torch.isnan(x[i])):
                        raise ValueError('error')
        return x

