import torch
import time
import copy
import sklearn.metrics as skmetrics
import pandas as pd
from collections import OrderedDict
import numpy as np
from utils import decision, all_same
import random

def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1, precision, recall

def train_scratch_model_no_valid(model, criterion, optimizer, dataloaders, device, num_epochs, use_clip_grad):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_acc = 0.0

    train_log = list()
    valid_log = list()
    train_accs = []
    valid_accs = []
    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has only training phase
        all_phases = ['train']

        for phase in all_phases:
            model.train()  # Set model to training mode

            running_loss = 0.0
            running_corrects = 0
            it = 0

            train_num_samples = 0
            val_num_samples = 0
            # Batch iterations
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # track history only if train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    labels = labels.to(torch.float64)
                    loss = criterion(outputs.squeeze(1), labels)
                    prepreds = torch.sigmoid(outputs)
                    preds = (prepreds >= 0.4).long().squeeze(1)


                    # backward + optimize only if in train phase
                    if it % 5 == 0:
                        print('iteration nb. {}'.format(it))
                    train_metrics = OrderedDict()
                    train_metrics['epoch'] = epoch
                    train_metrics['iter'] = it
                    it += 1
                    train_metrics['loss'] = loss.item()
                    # TODO: REVISAR accuracy !!
                    train_metrics['accuracy'] = (torch.sum(preds == labels.data) / inputs.size(0)).item()
                    # TODO: REVISAR F1score
                    f1score, preciss, recall = f1_loss(labels, preds)
                    train_metrics['f1score'] = f1score.item()
                    train_metrics['preciss'] = preciss.item()
                    train_metrics['recall'] = recall.item()
                    # TODO: WHY .PARAM_GROUPS[0][lr]?
                    train_metrics['lr'] = optimizer.param_groups[0]['lr']
                    loss.backward()
                    if use_clip_grad:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    optimizer.step()
                    train_log.append(train_metrics)
                    train_num_samples += 1 * inputs.size(0)


                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / train_num_samples
            epoch_acc = running_corrects / train_num_samples
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc.item())

            print('[{}];   Acc.: {:.4};   Loss: {:.4f}.'.format(phase, epoch_acc, epoch_loss))

            # deep copy the model
            if epoch_acc > best_acc:  # / val_num_samples > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best train Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, (train_accs, valid_accs), (train_losses, valid_losses), pd.DataFrame(train_log), pd.DataFrame(
        valid_log), best_epoch

def train_scratch_model(model, criterion, optimizer, dataloaders, device, num_epochs, valid_rec_names, valid_len,
                        valid_per_record, extra_aug, use_clip_grad, n_divisions_segments=4):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_acc = 0.0

    train_log = list()
    valid_log = list()
    train_accs = []
    valid_accs = []
    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        all_phases = ['train', 'valid']

        for phase in all_phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            it = 0

            train_num_samples = 0
            val_num_samples = 0

            valids_preds_ = []
            valid_targets_ = []
            valid_names_ = []
            valid_name_analyze = '0'
            # Batch iterations
            for inputs, labels, rec_info in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Extra augmentation?
                if extra_aug:
                    for batch_sample in range(inputs.shape[0]):
                        if decision(0.2):
                            inputs[batch_sample, :, :] = round(random.uniform(0.8, 1.2), 2) * inputs[batch_sample, :, :]
                        elif decision(0.2):
                            inputs[batch_sample, :, :] = -1 * inputs[batch_sample, :, :]

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # track history only if train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    labels = labels.to(torch.float64)
                    loss = criterion(outputs.squeeze(1), labels)
                    prepreds = torch.sigmoid(outputs)
                    preds = (prepreds >= 0.4).long().squeeze(1)


                    # backward + optimize only if in train phase
                    if phase == 'train':
                        if len(dataloaders[phase]) > 100:
                            if it % 50 == 0:
                                print('iteration nb. {}'.format(it))
                        else:
                            if it % 5 == 0:
                                print('iteration nb. {}'.format(it))
                        train_metrics = OrderedDict()
                        train_metrics['epoch'] = epoch
                        train_metrics['iter'] = it
                        it += 1
                        train_metrics['loss'] = loss.item()
                        # TODO: REVISAR accuracy !!
                        train_metrics['accuracy'] = (torch.sum(preds == labels.data) / inputs.size(0)).item()
                        # TODO: REVISAR F1score
                        f1score, preciss, recall = f1_loss(labels, preds)
                        train_metrics['f1score'] = f1score.item()
                        train_metrics['preciss'] = preciss.item()
                        train_metrics['recall'] = recall.item()
                        # TODO: WHY .PARAM_GROUPS[0][lr]?
                        train_metrics['lr'] = optimizer.param_groups[0]['lr']
                        loss.backward()
                        if use_clip_grad:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                        optimizer.step()
                        train_log.append(train_metrics)
                        train_num_samples += 1 * inputs.size(0)
                    else:
                        valid_metrics = OrderedDict()
                        valid_metrics['epoch'] = epoch
                        valid_metrics['loss'] = loss.item()
                        if valid_per_record:
                            if valid_name_analyze != rec_info[0]:
                                if valid_name_analyze != '0':
                                    valids_preds_.append(one_record_valid_predictions)
                                    valid_targets_.append(one_record_valid_targets)
                                one_record_valid_predictions = []
                                one_record_valid_targets = []
                                valid_names_.append(rec_info[0])
                                valid_name_analyze = rec_info[0]
                            assert (preds.item() == 1 or preds.item() == 0) and (   # sanity check
                                    labels.item() == 1 or labels.item() == 0)
                            one_record_valid_predictions.append(preds.item())
                            one_record_valid_targets.append(labels.item())
                        else:
                            valid_metrics['accuracy'] = (torch.sum(preds == labels.data) / inputs.size(0)).item()
                        f1score, preciss, recall = f1_loss(labels, preds)
                        valid_metrics['f1score'] = f1score.item()
                        valid_metrics['preciss'] = preciss.item()
                        valid_metrics['recall'] = recall.item()
                        valid_log.append(valid_metrics)
                        val_num_samples += 1 * inputs.size(0)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                if phase == 'train' or (phase == 'valid' and not valid_per_record):
                    running_corrects += torch.sum(preds == labels.data)
            if phase == 'valid' and valid_per_record:
                valids_preds_.append(one_record_valid_predictions)
                valid_targets_.append(one_record_valid_targets)


                tot_ = 0
                corr_ = 0
                for i_ in range(len(valid_targets_)):
                    assert all_same(valid_targets_[i_])
                    assert len(valid_targets_[i_]) == len(valids_preds_[i_])
                    index_count_ = 0
                    len_subsegment_ = len(valid_targets_[i_])//n_divisions_segments
                    for j_ in range(n_divisions_segments):
                        subsegment_preds_ = valids_preds_[i_][index_count_:(index_count_+len_subsegment_)]
                        subsegment_corrects_ = 0
                        for pred_ii in subsegment_preds_:
                            if pred_ii == valid_targets_[i_][0]:
                                subsegment_corrects_ += 1
                        if subsegment_corrects_/len_subsegment_ >= 0.5:
                            corr_ += 1
                        tot_ += 1
                        index_count_ += len_subsegment_

            if phase == 'train':
                epoch_loss = running_loss / train_num_samples
                epoch_acc = running_corrects / train_num_samples
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                epoch_loss = running_loss / val_num_samples
                if valid_per_record:
                    epoch_acc = corr_/tot_
                else:
                    epoch_acc = running_corrects / val_num_samples
                valid_losses.append(epoch_loss)
                valid_accs.append(epoch_acc)

            print('[{}];   Acc.: {:.4};   Loss: {:.4f}.'.format(phase, epoch_acc, epoch_loss))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, (train_accs, valid_accs), (train_losses, valid_losses), pd.DataFrame(train_log), pd.DataFrame(
        valid_log), best_epoch