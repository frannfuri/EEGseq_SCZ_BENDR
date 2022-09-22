import torch, tqdm
import copy
import numpy as np
from torch import nn
from copy import deepcopy
from math import ceil

MODEL_CHOICES = ['BENDR', 'linear']


class LinearHeadBENDR(nn.Module):

    @property
    def num_features_for_classification(self):
        return self.encoder_h * self.pool_length

    def features_forward(self, x):
        x = self.encoder(x)
        x = self.enc_augment(x)
        x = self.summarizer(x)
        return self.extended_classifier(x)

    def __init__(self, n_targets, samples_len, n_chn, encoder_h=512, projection_head=False, enc_do=0.1, feat_do=0.4,
                 pool_length=4, mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.05, mask_c_span=0.1,
                 classifier_layers=1, return_features=True):
        if classifier_layers < 1:
            self.pool_length = pool_length
            self.encoder_h = 3 * encoder_h
        else:
            self.pool_length = pool_length // classifier_layers
            self.encoder_h = encoder_h

        super().__init__()
        self.samples_len = samples_len
        self.n_chn = n_chn
        self.return_features = return_features
        self.n_targets = n_targets
        self.make_new_classification_layer()
        self._init_state = self.state_dict()
        ##

        self.encoder = ConvEncoderBENDR(n_chn, encoder_h=encoder_h, projection_head=projection_head, dropout=enc_do)
        encoded_samples = self.encoder.downsampling_factor(samples_len)

        mask_t_span = mask_t_span if mask_t_span > 1 else int(mask_t_span * encoded_samples)
        # Important for short things like P300
        mask_t_span = 0 if encoded_samples < 2 else mask_t_span
        mask_c_span = mask_c_span if mask_c_span > 1 else int(mask_c_span * encoder_h)

        self.enc_augment = EncodingAugment(encoder_h, mask_p_t, mask_p_c, mask_c_span=mask_c_span,
                                           mask_t_span=mask_t_span)
        tqdm.tqdm.write(self.encoder.description(None, samples_len) + " | {} pooled".format(pool_length))  # sfreq ?
        self.summarizer = nn.AdaptiveAvgPool1d(pool_length)

        classifier_layers = [self.encoder_h * self.pool_length for i in range(classifier_layers)] if \
            not isinstance(classifier_layers, (tuple, list)) else classifier_layers
        classifier_layers.insert(0, 3 * encoder_h * pool_length)
        self.extended_classifier = nn.Sequential(Flatten())
        for i in range(1, len(classifier_layers)):
            self.extended_classifier.add_module("ext-classifier-{}".format(i), nn.Sequential(
                nn.Linear(classifier_layers[i - 1], classifier_layers[i]),
                nn.Dropout(feat_do),
                nn.ReLU(),
                nn.BatchNorm1d(classifier_layers[i]),
            ))

    def internal_loss(self, forward_pass_tensors):
        return None

    def clone(self):
        """
        Standard way to copy models, weights and all.
        """
        return deepcopy(self)

    def reset(self):
        self.load_state_dict(self._init_state)

    def forward(self, *x):
        features = self.features_forward(*x)
        if self.return_features:
            return self.classifier_forward(features), features
        else:
            return self.classifier_forward(features)

    def make_new_classification_layer(self):
        """
        Make a distinction between the classification layer(s) and the rest of the network. Using a basic
        formulation of a network being composed of two parts feature_extraction & classifier.
        This method implement the classification side, so that methods like :py:meth:`freeze_features` works
        as intended.
        Anything besides a layer that just flattens anything incoming to a vector and Linearly weights this to
        the target should override this method, and there should be a variable called `self.classifier`
        """
        classifier = nn.Linear(self.num_features_for_classification, self.n_targets)
        nn.init.xavier_normal_(classifier.weight)
        classifier.bias.data.zero_()
        self.classifier = nn.Sequential(Flatten(), classifier)

    def freeze_features(self, unfreeze=False, freeze_classifier=False):
        """
        Sometimes, the features learned by a model in one domain can be applied to another case.

        This method freezes (or un-freezes) all but the `classifier` layer. So that any further training
        doesnt (or does if unfreeze=True) affect these weights.

        Parameters
        ----------
        unfreeze : bool
                   To unfreeze weights after a previous call to this.
        freeze_classifier: bool
                   Commonly, the classifier layer will not be frozen (default). Setting this to `True` will freeze this
                   layer too.
        """
        for param in self.parameters():
            param.requires_grad = unfreeze

        if isinstance(self.classifier, nn.Module) and not freeze_classifier:
            for param in self.classifier.parameters():
                param.requires_grad = True

    # @property
    # def num_features_for_classification(self):
    #    raise NotImplementedError

    def classifier_forward(self, features):
        return self.classifier(features)

    # def features_forward(self, x):
    #    raise NotImplementedError

    def load(self, filename, include_classifier=False, freeze_features=True):
        state_dict = torch.load(filename)
        if not include_classifier:
            for key in [k for k in state_dict.keys() if 'classifier' in k]:
                state_dict.pop(key)
        self.load_state_dict(state_dict, strict=False)
        if freeze_features:
            self.freeze_features()

    def save(self, filename, ignore_classifier=False):
        state_dict = self.state_dict()
        if ignore_classifier:
            for key in [k for k in state_dict.keys() if 'classifier' in k]:
                state_dict.pop(key)
        print("Saving to {} ...".format(filename))
        torch.save(state_dict, filename)

    def load_encoder(self, encoder_file, freeze=False, strict=True):
        self.encoder.load(encoder_file, strict=strict)
        self.encoder.freeze_features(not freeze)
        print("Loaded {}".format(encoder_file))

    def load_pretrained_modules(self, encoder_file='./datasets/encoder.pt',
                                contextualizer_file='./datasets/contextualizer.pt',
                                strict=False, freeze_encoder=True):
        self.load_encoder(encoder_file, strict=strict, freeze=freeze_encoder)
        self.enc_augment.init_from_contextualizer(contextualizer_file)


class ConvEncoderBENDR(nn.Module):
    def __init__(self, in_features, encoder_h=256, enc_width=(3, 2, 2, 2, 2, 2),
                 dropout=0., projection_head=False, enc_downsample=(3, 2, 2, 2, 2, 2)):
        super().__init__()
        self.in_features = in_features
        self.encoder_h = encoder_h
        if not isinstance(enc_width, (list, tuple)):
            enc_width = [enc_width]
        if not isinstance(enc_downsample, (list, tuple)):
            enc_downsample = [enc_downsample]
        assert len(enc_downsample) == len(enc_width)

        # Centerable convolutions make life simpler
        enc_width = [e if e % 2 else e + 1 for e in enc_width]
        self._downsampling = enc_downsample
        self._width = enc_width

        self.encoder = nn.Sequential()
        for i, (width, downsample) in enumerate(zip(enc_width, enc_downsample)):
            self.encoder.add_module("Encoder_{}".format(i), nn.Sequential(
                nn.Conv1d(in_features, encoder_h, width, stride=downsample, padding=width // 2),
                nn.Dropout2d(dropout),
                nn.GroupNorm(encoder_h // 2, encoder_h),
                nn.GELU(),
            ))
            in_features = encoder_h

        if projection_head:
            self.encoder.add_module("projection-1", nn.Sequential(
                nn.Conv1d(in_features, in_features, 1),
                nn.Dropout2d(dropout * 2),
                nn.GroupNorm(in_features // 2, in_features),
                nn.GELU()
            ))

    def load(self, filename, strict=True):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def freeze_features(self, unfreeze=False):
        for param in self.parameters():
            param.requires_grad = unfreeze

    def description(self, sfreq=None, sequence_len=None):
        widths = list(reversed(self._width))[1:]
        strides = list(reversed(self._downsampling))[1:]

        rf = self._width[-1]
        for w, s in zip(widths, strides):
            rf = rf if w == 1 else (rf - 1) * s + 2 * (w // 2)

        desc = "Receptive field: {} samples".format(rf)
        if sfreq is not None:
            desc += ", {:.2f} seconds".format(rf / sfreq)

        ds_factor = np.prod(self._downsampling)
        desc += " | Downsampled by {}".format(ds_factor)
        if sfreq is not None:
            desc += ", new sfreq: {:.2f} Hz".format(sfreq / ds_factor)
        desc += " | Overlap of {} samples".format(rf - ds_factor)
        if sequence_len is not None:
            desc += " | {} encoded samples/trial".format(sequence_len // ds_factor)
        return desc

    def downsampling_factor(self, samples):
        for factor in self._downsampling:
            samples = ceil(samples / factor)
        return samples

    def forward(self, x):
        return self.encoder(x)


# FIXME this is redundant with part of the contextualizer
class EncodingAugment(nn.Module):
    def __init__(self, in_features, mask_p_t=0.1, mask_p_c=0.01, mask_t_span=6, mask_c_span=64, dropout=0.1,
                 position_encoder=25):
        super().__init__()
        self.mask_replacement = torch.nn.Parameter(torch.zeros(in_features), requires_grad=True)
        self.p_t = mask_p_t
        self.p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        transformer_dim = 3 * in_features

        conv = nn.Conv1d(in_features, in_features, position_encoder, padding=position_encoder // 2, groups=16)
        nn.init.normal_(conv.weight, mean=0, std=2 / transformer_dim)
        nn.init.constant_(conv.bias, 0)
        conv = nn.utils.weight_norm(conv, dim=2)
        self.relative_position = nn.Sequential(conv, nn.GELU())

        self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            Permute([0, 2, 1]),
            nn.Conv1d(in_features, transformer_dim, 1),
        )

    def forward(self, x, mask_t=None, mask_c=None):
        bs, feat, seq = x.shape

        if self.training:
            if mask_t is None and self.p_t > 0 and self.mask_t_span > 0:
                mask_t = _make_mask((bs, seq), self.p_t, x.shape[-1], self.mask_t_span)
            if mask_c is None and self.p_c > 0 and self.mask_c_span > 0:
                mask_c = _make_mask((bs, feat), self.p_c, x.shape[1], self.mask_c_span)

        if mask_t is not None:
            x.transpose(2, 1)[mask_t] = self.mask_replacement
        if mask_c is not None:
            x[mask_c] = 0

        x = self.input_conditioning(x + self.relative_position(x))
        return x

    def init_from_contextualizer(self, filename):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=False)
        for param in self.parameters():
            param.requires_grad = False
        print("Initialized mask embedding and position encoder from ", filename)


# Layer type
class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


# Layer type
class Permute(nn.Module):
    def __init__(self, axes):
        super().__init__()
        self.axes = axes

    def forward(self, x):
        return x.permute(self.axes)


def _make_span_from_seeds(seeds, span, total=None):
    inds = list()
    for seed in seeds:
        for i in range(seed, seed + span):
            if total is not None and i >= total:
                break
            elif i not in inds:
                inds.append(int(i))
    return np.array(inds)


def _make_mask(shape, p, total, span, allow_no_inds=False):
    # num_mask_spans = np.sum(np.random.rand(total) < p)
    # num_mask_spans = int(p * total)
    mask = torch.zeros(shape, requires_grad=False, dtype=torch.bool)

    for i in range(shape[0]):
        mask_seeds = list()
        while not allow_no_inds and len(mask_seeds) == 0 and p > 0:
            mask_seeds = np.nonzero(np.random.rand(total) < p)[0]

        mask[i, _make_span_from_seeds(mask_seeds, span, total=total)] = True

    return mask


class BENDRClassification(nn.Module):

    @property
    def num_features_for_classification(self):
        return self.encoder_h

    def features_forward(self, *x):
        encoded = self.encoder(x[0])

        if self.trial_embeddings is not None and len(x) > 1:
            embeddings = self.trial_embeddings(x[-1])
            encoded += embeddings.unsqueeze(-1).expand_as(encoded)

        context = self.contextualizer(encoded)
        # return self.projection_mlp(context[:, :, 0])
        # return nn.functional.adaptive_max_pool1d(context, output_size=1)
        return context[:, :, -1]

    def __init__(self, targets=2, samples_len=6 * 256, n_chn=20, encoder_h=512, contextualizer_hidden=3076,
                 projection_head=False,
                 new_projection_layers=0, dropout=0., trial_embeddings=None, layer_drop=0, keep_layers=None,
                 mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.1, mask_c_span=0.1, multi_gpu=False,
                 return_features=True, regression_option=False):
        self.encoder_h = encoder_h
        self.contextualizer_hidden = contextualizer_hidden
        super(BENDRClassification, self).__init__()
        self.samples_len = samples_len
        self.n_chn = n_chn
        self.return_features = return_features
        self.targets = targets
        self.regression_option = regression_option
        if self.regression_option:
            self.make_new_regression_layer(numb_of_targets=self.targets)
        else:
            self.make_new_classification_layer(numb_of_targets=self.targets)
        self._init_state = self.state_dict()

        encoder = ConvEncoderBENDR(n_chn, encoder_h=encoder_h, dropout=dropout, projection_head=projection_head)
        encoded_samples = encoder.downsampling_factor(samples_len)

        mask_t_span = mask_t_span if mask_t_span > 1 else int(mask_t_span * encoded_samples)
        mask_c_span = mask_c_span if mask_c_span > 1 else int(mask_c_span * encoder_h)
        contextualizer = BENDRContextualizer(encoder_h, hidden_feedforward=contextualizer_hidden, finetuning=True,
                                             mask_p_t=mask_p_t, mask_p_c=mask_p_c, layer_drop=layer_drop,
                                             mask_c_span=mask_c_span, dropout=dropout,
                                             mask_t_span=mask_t_span)

        self.encoder = nn.DataParallel(encoder) if multi_gpu else encoder
        self.contextualizer = nn.DataParallel(contextualizer) if multi_gpu else contextualizer

        tqdm.tqdm.write(encoder.description(sequence_len=samples_len))

        self.projection_mlp = nn.Sequential()
        for p in range(1, new_projection_layers + 1):
            self.projection_mlp.add_module("projection-{}".format(p), nn.Sequential(
                nn.Linear(encoder_h, encoder_h),
                nn.Dropout(dropout),
                nn.BatchNorm1d(encoder_h),
                nn.GELU(),
            ))
        self.trial_embeddings = nn.Embedding(trial_embeddings, encoder_h, scale_grad_by_freq=True) \
            if trial_embeddings is not None else trial_embeddings

    def load_encoder(self, encoder_file, freeze=False, strict=True):
        self.encoder.load(encoder_file, strict=strict)
        self.encoder.freeze_features(unfreeze=not freeze)

    def load_contextualizer(self, contextualizer_file, freeze=False, strict=True):
        self.contextualizer.load(contextualizer_file, strict=strict)
        self.contextualizer.freeze_features(unfreeze=not freeze)

    def load_pretrained_modules(self, encoder_file, contextualizer_file, freeze_encoder=False,
                                freeze_contextualizer=False, freeze_position_conv=False,
                                freeze_mask_replacement=True, strict=False):
        self.load_encoder(encoder_file, freeze=freeze_encoder, strict=strict)
        self.load_contextualizer(contextualizer_file, freeze=freeze_contextualizer, strict=strict)
        self.contextualizer.mask_replacement.requires_grad = freeze_mask_replacement
        if freeze_position_conv:
            for p in self.contextualizer.relative_position.parameters():
                p.requires_grad = False

    # CUSTOM FUNCTION
    def load_whole_pretrained_modules(self, state_dict_file, freeze_encoder, freeze_contextualizer,
                                      freeze_position_conv,
                                      freeze_mask_replacement, device):
        self.load_state_dict(torch.load(state_dict_file, map_location=device))
        self.encoder.freeze_features(unfreeze=not freeze_encoder)
        # TODO: WITH MASK REPLACEMENT.REQUIRES_GRAD ?
        self.contextualizer.freeze_features(unfreeze=not freeze_contextualizer, finetuning=freeze_mask_replacement)
        if freeze_position_conv:
            for p in self.contextualizer.relative_position.parameters():
                p.requires_grad = False

    def reset(self):
        self.load_state_dict(self._init_state)

    def forward(self, *x):
        features = self.features_forward(*x)
        if self.regression_option:
            if self.return_features:
                return self.regressor_forward(features), features
            else:
                return self.regressor_forward(features)
        else:
            if self.return_features:
                return self.classifier_forward(features), features
            else:
                return self.classifier_forward(features)

    def make_new_classification_layer(self, numb_of_targets):
        """
        This allows for a distinction between the classification layer(s) and the rest of the network. Using a basic
        formulation of a network being composed of two parts feature_extractor & classifier.
        This method is for implementing the classification side, so that methods like :py:meth:`freeze_features` works
        as intended.
        Anything besides a layer that just flattens anything incoming to a vector and Linearly weights this to the
        target should override this method, and there should be a variable called `self.classifier`
        """
        classifier = nn.Linear(self.num_features_for_classification, numb_of_targets)
        nn.init.xavier_normal_(classifier.weight)
        classifier.bias.data.zero_()
        self.classifier = nn.Sequential(Flatten(), classifier)

    def make_new_regression_layer(self, numb_of_targets=1):
        regressor = nn.Linear(self.num_features_for_classification, numb_of_targets)
        nn.init.xavier_normal_(regressor.weight)
        regressor.bias.data.zero_()
        self.regressor = nn.Sequential(Flatten(), regressor)

    def freeze_features(self, unfreeze=False, freeze_classifier=False):
        """
        In many cases, the features learned by a model in one domain can be applied to another case.
        This method freezes (or un-freezes) all but the `classifier` layer. So that any further training does not (or
        does if unfreeze=True) affect these weights.
        Parameters
        ----------
        unfreeze : bool
                   To unfreeze weights after a previous call to this.
        freeze_classifier: bool
                   Commonly, the classifier layer will not be frozen (default). Setting this to `True` will freeze this
                   layer too.
        """
        for param in self.parameters():
            param.requires_grad = unfreeze

        if isinstance(self.classifier, nn.Module) and not freeze_classifier:
            for param in self.classifier.parameters():
                param.requires_grad = True

    def classifier_forward(self, features):
        return self.classifier(features)

    def regressor_forward(self, features):
        return self.regressor(features)

    def clone(self):
        """
        This provides a standard way to copy models, weights and all.
        """
        return deepcopy(self)

    def load(self, filename, include_classifier=False, freeze_features=True):
        state_dict = torch.load(filename)
        if not include_classifier:
            for key in [k for k in state_dict.keys() if 'classifier' in k]:
                state_dict.pop(key)
        self.load_state_dict(state_dict, strict=False)
        if freeze_features:
            self.freeze_features()

    def save(self, filename, ignore_classifier=False):
        state_dict = self.state_dict()
        if ignore_classifier:
            for key in [k for k in state_dict.keys() if 'classifier' in k]:
                state_dict.pop(key)
        print("Saving to {} ...".format(filename))
        torch.save(state_dict, filename)


class _Hax(nn.Module):
    """T-fixup assumes self-attention norms are removed"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class BENDRContextualizer(nn.Module):
    def __init__(self, in_features, hidden_feedforward=3076, heads=8, layers=8, dropout=0.15, activation='gelu',
                 position_encoder=25, layer_drop=0.0, mask_p_t=0.1, mask_p_c=0.004, mask_t_span=6, mask_c_span=64,
                 start_token=-5, finetuning=False):
        super().__init__()

        self.dropout = dropout
        self.in_features = in_features
        self._transformer_dim = in_features * 3

        encoder = nn.TransformerEncoderLayer(d_model=in_features * 3, nhead=heads, dim_feedforward=hidden_feedforward,
                                             dropout=dropout, activation=activation)
        encoder.norm1 = _Hax()
        encoder.norm2 = _Hax()

        self.norm = nn.LayerNorm(self._transformer_dim)

        # self.norm_layers = nn.ModuleList([copy.deepcopy(norm) for _ in range(layers)])
        self.transformer_layers = nn.ModuleList([copy.deepcopy(encoder) for _ in range(layers)])
        self.layer_drop = layer_drop
        self.p_t = mask_p_t
        self.p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        self.start_token = start_token
        self.finetuning = finetuning

        # Initialize replacement vector with 0's
        self.mask_replacement = torch.nn.Parameter(torch.normal(0, in_features ** (-0.5), size=(in_features,)),
                                                   requires_grad=True)

        self.position_encoder = position_encoder > 0
        if position_encoder:
            conv = nn.Conv1d(in_features, in_features, position_encoder, padding=position_encoder // 2, groups=16)
            nn.init.normal_(conv.weight, mean=0, std=2 / self._transformer_dim)
            nn.init.constant_(conv.bias, 0)
            conv = nn.utils.weight_norm(conv, dim=2)
            self.relative_position = nn.Sequential(conv, nn.GELU())

        self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            Permute([0, 2, 1]),
            nn.Conv1d(in_features, self._transformer_dim, 1),
            Permute([2, 0, 1]),
        )

        self.output_layer = nn.Conv1d(self._transformer_dim, in_features, 1)
        self.apply(self.init_bert_params)

    def init_bert_params(self, module):
        if isinstance(module, nn.Linear):
            # module.weight.data.normal_(mean=0.0, std=0.02)
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
            # Tfixup
            module.weight.data = 0.67 * len(self.transformer_layers) ** (-0.25) * module.weight.data

        # if isinstance(module, nn.Conv1d):
        #     # std = np.sqrt((4 * (1.0 - self.dropout)) / (self.in_features * self.in_features))
        #     # module.weight.data.normal_(mean=0.0, std=std)
        #     nn.init.xavier_uniform_(module.weight.data)
        #     module.bias.data.zero_()

    def forward(self, x, mask_t=None, mask_c=None):
        bs, feat, seq = x.shape
        if self.training and self.finetuning:
            if mask_t is None and self.p_t > 0:
                mask_t = _make_mask((bs, seq), self.p_t, x.shape[-1], self.mask_t_span)
            if mask_c is None and self.p_c > 0:
                mask_c = _make_mask((bs, feat), self.p_c, x.shape[1], self.mask_c_span)

        if mask_t is not None:
            x.transpose(2, 1)[mask_t] = self.mask_replacement
        if mask_c is not None:
            x[mask_c] = 0

        if self.position_encoder:
            x = x + self.relative_position(x)
        x = self.input_conditioning(x)

        if self.start_token is not None:
            in_token = self.start_token * torch.ones((1, 1, 1), requires_grad=True).to(x.device).expand(
                [-1, *x.shape[1:]])
            # in_token shape 1, batch_size, seq_len   (creo..)
            x = torch.cat([in_token, x], dim=0)

        for layer in self.transformer_layers:
            if not self.training or torch.rand(1) > self.layer_drop:
                x = layer(x)

        return self.output_layer(x.permute([1, 2, 0]))

    def freeze_features(self, unfreeze=False, finetuning=False):
        for param in self.parameters():
            param.requires_grad = unfreeze
        if self.finetuning or finetuning:
            self.mask_replacement.requires_grad = False

    def load(self, filename, strict=True):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)