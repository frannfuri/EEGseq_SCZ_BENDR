import mne
import mne.epochs
import torch
import numpy as np
import os

from tqdm import tqdm
from torch.nn.functional import interpolate
from channels import map_dataset_channels_deep_1010, DEEP_1010_CH_TYPES, SCALE_IND, \
    EEG_INDS, EOG_INDS, REF_INDS, EXTRA_INDS, DEEP_1010_CHS_LISTING

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size


# TODO: Only to SA047 for now
class SubjectOnlySampler(BatchSampler):
    """
    BatchSampler - from a clinical dataset with different subjects with different records,
    samples only the data corresponding to a one subject at a time
    """

    def __init__(self, epochs_all_subjects, t_IDs, train_data=True, batch_size=None):
        # TODO: Does it make sense that there is a batch_size option?
        self.t = 0
        self.indexes = []
        for t_id in t_IDs:
            self.indexes.append(list(range(self.t, self.t + len(epochs_all_subjects[t_id][1]))))
            self.t += len(epochs_all_subjects[t_id][1])

    def __len__(self):
        c = 0
        for b_list in self.indexes:
            c += len(b_list)
        return c


class InstanceTransform(object):

    def __init__(self, only_trial_data=True):
        """
        Trial transforms are, for the most part, simply operations that are performed on the loaded tensors when they are
        fetched via the :meth:`__call__` method. Ideally this is implemented with pytorch operations for ease of execution
        graph integration.
        """
        self.only_trial_data = only_trial_data

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, *x):
        """
        Modifies a batch of tensors.
        Parameters
        ----------
        x : torch.Tensor, tuple
            The trial tensor, not including a batch-dimension. If initialized with `only_trial_data=False`, then this
            is a tuple of all ids, labels, etc. being propagated.
        Returns
        -------
        x : torch.Tensor, tuple
            The modified trial tensor, or tensors if not `only_trial_data`
        """
        raise NotImplementedError()

    def new_channels(self, old_channels):
        """
        This is an optional method that indicates the transformation modifies the representation and/or presence of
        channels.
        Parameters
        ----------
        old_channels : ndarray
                       An array whose last two dimensions are channel names and channel types.
        Returns
        -------
        new_channels : ndarray
                      An array with the channel names and types after this transformation. Supports the addition of
                      dimensions e.g. a list of channels into a rectangular grid, but the *final two dimensions* must
                      remain the channel names, and types respectively.
        """
        return old_channels

    def new_sfreq(self, old_sfreq):
        """
        This is an optional method that indicates the transformation modifies the sampling frequency of the underlying
        time-series.
        Parameters
        ----------
        old_sfreq : float
        Returns
        -------
        new_sfreq : float
        """
        return old_sfreq

    def new_sequence_length(self, old_sequence_length):
        """
        This is an optional method that indicates the transformation modifies the length of the acquired extracts,
        specified in number of samples.
        Parameters
        ----------
        old_sequence_length : int
        Returns
        -------
        new_sequence_length : int
        """
        return old_sequence_length


def min_max_normalize(x: torch.Tensor, low=-1, high=1):
    if len(x.shape) == 2:
        xmin = x.min()
        xmax = x.max()
        if xmax - xmin == 0:
            x = 0
            return x
    elif len(x.shape) == 3:
        xmin = torch.min(torch.min(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        xmax = torch.max(torch.max(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        constant_trials = (xmax - xmin) == 0
        if torch.any(constant_trials):
            # If normalizing multiple trials, stabilize the normalization
            xmax[constant_trials] = xmax[constant_trials] + 1e-6

    x = (x - xmin) / (xmax - xmin)

    # Now all scaled 0 -> 1, remove 0.5 bias
    x -= 0.5
    # Adjust for low/high bias and scale up
    x += (high + low) / 2
    return (high - low) * x


class MappingDeep1010(InstanceTransform):
    """
    Maps various channel sets into the Deep10-10 scheme, and normalizes data between [-1, 1] with an additional scaling
    parameter to describe the relative scale of a trial with respect to the entire dataset.
    TODO - refer to eventual literature on this
    """

    def __init__(self, channels, data_max, data_min, return_mask=False, th_clipping=None):
        """
        Creates a Deep10-10 mapping for the provided dataset.
        Parameters
        ----------
        dataset : Dataset
        add_scale_ind : bool
                        If `True` (default), the scale ind is filled with the relative scale of the trial with respect
                        to the data min and max of the dataset.
        return_mask : bool
                      If `True` (`False` by default), an additional tensor is returned after this transform that
                      says which channels of the mapping are in fact in use.
        """
        super().__init__()
        self.mapping = map_dataset_channels_deep_1010(channels)
        # TODO: improve scaling
        # self.max_scale = 2*float(th_clipping)
        self.return_mask = return_mask
        self.max_scale = data_max - data_min

    def __call__(self, x):
        if self.max_scale is not None:
            scale = 2 * (torch.clamp_max((x.max() - x.min()) / self.max_scale, 1.0) - 0.5)
        else:
            scale = 0

        x = (x.transpose(1, 0) @ self.mapping).transpose(1, 0)

        for ch_type_inds in (EEG_INDS, EOG_INDS, REF_INDS, EXTRA_INDS):
            x[ch_type_inds, :] = min_max_normalize(x[ch_type_inds, :])

        used_channel_mask = self.mapping.sum(dim=0).bool()
        x[~used_channel_mask, :] = 0

        x[SCALE_IND, :] = scale

        if self.return_mask:
            return (x, used_channel_mask)
        else:
            return x

    def new_channels(self, old_channels: np.ndarray):
        channels = list()
        for row in range(self.mapping.shape[1]):
            active = self.mapping[:, row].nonzero().numpy()
            if len(active) > 0:
                channels.append("-".join([old_channels[i.item(), 0] for i in active]))
            else:
                channels.append(None)
        return np.array(list(zip(channels, DEEP_1010_CH_TYPES)))


class TemporalInterpolation(InstanceTransform):

    def __init__(self, desired_sequence_length, mode='nearest', new_sfreq=None):
        """
        This is in essence a DN3 wrapper for the pytorch function
        `interpolate() <https://pytorch.org/docs/stable/nn.functional.html>`_
        Currently only supports single dimensional samples (i.e. channels have not been projected into more dimensions)
        Warnings
        --------
        Using this function to downsample data below a suitable nyquist frequency given the low-pass filtering of the
        original data will cause dangerous aliasing artifacts that will heavily affect data quality to the point of it
        being mostly garbage.
        Parameters
        ----------
        desired_sequence_length: int
                                 The desired new sequence length of incoming data.
        mode: str
              The technique that will be used for upsampling data, by default 'nearest' interpolation. Other options
              are listed under pytorch's interpolate function.
        new_sfreq: float, None
                   If specified, registers the change in sampling frequency
        """
        super().__init__()
        self._new_sequence_length = desired_sequence_length
        self.mode = mode
        self._new_sfreq = new_sfreq

    def __call__(self, x):
        # squeeze and unsqueeze because these are done before batching
        if len(x.shape) == 2:
            return interpolate(x.unsqueeze(0), self._new_sequence_length, mode=self.mode).squeeze(0)
        # Supports batch dimension
        elif len(x.shape) == 3:
            return interpolate(x, self._new_sequence_length, mode=self.mode)
        else:
            raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")

    def new_sequence_length(self, old_sequence_length):
        return self._new_sequence_length

    def new_sfreq(self, old_sfreq):
        if self._new_sfreq is not None:
            return self._new_sfreq
        else:
            return old_sfreq


class To1020(InstanceTransform):
    EEG_20_div = [
        'FP1', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'T5', 'P3', 'PZ', 'P4', 'T6',
        'O1', 'O2'
    ]

    def __init__(self, only_trial_data=True, include_scale_ch=True, include_ref_chs=False):
        """
        Transforms incoming Deep1010 data into exclusively the more limited 1020 channel set.
        """
        super(To1020, self).__init__(only_trial_data=only_trial_data)
        self._inds_20_div = [DEEP_1010_CHS_LISTING.index(ch) for ch in self.EEG_20_div]
        if include_ref_chs:
            self._inds_20_div.append([DEEP_1010_CHS_LISTING.index(ch) for ch in ['A1', 'A2']])
        if include_scale_ch:
            self._inds_20_div.append(SCALE_IND)

    def new_channels(self, old_channels):
        return old_channels[self._inds_20_div]

    def __call__(self, *x):
        x = list(x)
        for i in range(len(x)):
            # Assume every tensor that has deep1010 length should be modified
            if len(x[i].shape) > 0 and x[i].shape[0] == len(DEEP_1010_CHS_LISTING):
                x[i] = x[i][self._inds_20_div, ...]
        return x


def whole_dataset_get_max_and_min(path_to_data, format_type='set', rd=0.9, chns_to_consider=['Cz']):
    raw_records = []
    for root, folders, _ in os.walk(path_to_data):
        for fold_day in sorted(folders):
            for root_day, _, files in os.walk(os.path.join(root, fold_day)):
                for file in sorted(files):
                    if file.endswith(format_type):  # and 'PSG' in file:
                        print('====================Processing record ' + str(file[:-4]) + '======================')
                        if format_type == 'set':
                            new_raw = mne.io.read_raw_eeglab(os.path.join(root_day, file), preload=True)
                            ch_to_remove = list(set(new_raw.ch_names) - set(chns_to_consider))
                            new_raw.drop_channels(ch_to_remove)
                            raw_records.append(new_raw)
                        elif format_type == 'edf':
                            new_raw = mne.io.read_raw_edf(os.path.join(root, file), preload=True)
                            # TODO: HARDCODED
                            #assert new_raw.ch_names == ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal',
                            #                            'EMG submental', 'Temp rectal', 'Event marker']
                            #new_raw.rename_channels({'EEG Fpz-Cz': 'Fpz', 'EEG Pz-Oz': 'Pz'})
                            ch_to_remove = list(set(new_raw.ch_names) - set(chns_to_consider))
                            new_raw.drop_channels(ch_to_remove)
                            raw_records.append(new_raw)
    pbar = tqdm(raw_records)
    print('Numb. of records: {}'.format(len(raw_records)))
    total_dmax = None
    total_dmin = None
    total_running_dev_max = 0
    total_running_dev_min = 0
    for raw_rec in pbar:
        _max, _min = get_record_max_and_min(raw_rec)
        if total_dmax is None:
            total_dmax = _max
        if total_dmin is None:
            total_dmin = _min
        total_dmax = max(total_dmax, _max)
        total_dmin = min(total_dmin, _min)
        total_running_dev_max = rd * total_running_dev_max + (1 - rd) * (total_dmax - _max) ** 2
        total_running_dev_min = rd * total_running_dev_min + (1 - rd) * (total_dmin - _min) ** 2
        pbar.set_postfix(
            dict(dmax=total_dmax, dmin=total_dmin, dev_max=total_running_dev_max, dev_min=total_running_dev_min))
    return total_dmax, total_dmin


def get_record_max_and_min(rawEeg: mne.io.eeglab.eeglab.RawEEGLAB):  # , rd=0.9):
    """
    This utility function is used early on to determine the *data_max* and *data_min* parameters that are added to the
    configuratron to properly create the Deep1010 mapping. Running factor tracks how much the individual max and mins
    deviate from the max and min while searching, useful to track how large the peaks in data appear.
    Parameters
    ----------
    rawEeg: a complete record in mne RawEEGLAB format
    rd: float
    Returns
    -------
    max, min
    """
    dmax = None
    dmin = None
    # running_dev_max = 0
    # running_dev_min = 0
    for chn in range(rawEeg._data.shape[0]):
        data = rawEeg._data[chn, :]
        _max = data.max()
        _min = data.min()
        if dmax is None:
            dmax = _max
        if dmin is None:
            dmin = _min
        dmax = max(dmax, _max)
        dmin = min(dmin, _min)
        # running_dev_max = rd * running_dev_max + (1 - rd) * (dmax - _max) ** 2
        # running_dev_min = rd * running_dev_min + (1 - rd) * (dmin - _min) ** 2
        # pbar.set_postfix(dict(dmax=dmax, dmin=dmin, dev_max=running_dev_max, dev_min=running_dev_min))

    return dmax, dmin


def modified_mae(input, target):
    return torch.mean(target*torch.abs(input-target))