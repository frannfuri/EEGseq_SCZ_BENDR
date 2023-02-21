import torch
import time
import copy
import sklearn.metrics as skmetrics
import pandas as pd
from collections import OrderedDict
import numpy as np
from utils import decision, all_same, accuracy_per_segments, accuracy_per_segments_detection
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

def train_scratch_model_no_valid(model, criterion, optimizer, dataloaders, device, num_epochs, type_task, use_clip_grad, n_outputs):
    since = time.time()
    # DONT IMPLEMENTED YET
    assert type_task != 'regressor'
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
                    if n_outputs == 1:
                        labels = labels.to(torch.float64)
                        loss = criterion(outputs.squeeze(1), labels)
                        prepreds = torch.sigmoid(outputs)
                        preds = (prepreds >= 0.4).long().squeeze(1)
                    else:
                        labels = labels.to(torch.int64)
                        loss = criterion(outputs,labels)
                        _, preds = torch.max(outputs, 1)


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
                        valid_per_record, extra_aug,  type_task, use_clip_grad, n_divisions_segments=4, n_outputs=1, split_criterion=False,
                        criterion0=None, criterion1=None, scheduler=None):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_acc = 0.0
    best_loss = 999.999

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
                    if type_task == 'classifier':
                        if n_outputs == 1:
                            labels = labels.to(torch.float64)
                            loss = criterion(outputs.squeeze(1), labels)
                            prepreds = torch.sigmoid(outputs)
                            preds = (prepreds >= 0.4).long().squeeze(1)
                        else:
                            labels = labels.to(torch.int64)
                            if split_criterion:
                                zero_ids = [i for i in range(len(labels)) if labels[i] == 0]
                                one_ids = [i for i in range(len(labels)) if labels[i] == 1]
                                loss0 = criterion0(outputs[zero_ids], labels[zero_ids])
                                loss1 = criterion1(outputs[one_ids], labels[one_ids])
                                assert criterion is None
                                if len(zero_ids) == 0:
                                    loss = loss1
                                elif len(one_ids) == 0:
                                    loss = loss0
                                else:
                                    loss = loss0 + loss1
                            else:
                                loss = criterion(outputs, labels)
                                assert (criterion0 is None) and (criterion1 is None)
                            _, preds = torch.max(outputs, 1)
                    else:
                        loss = criterion(outputs.squeeze(1).double(), labels.double())
                        if split_criterion:
                            assert 1 == 0
                        preds = outputs

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
                        if type_task == 'classifier':
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
                        if type_task == 'classifier':
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
                if type_task == 'classifier':
                    if phase == 'train' or (phase == 'valid' and not valid_per_record):
                        running_corrects += torch.sum(preds == labels.data)
            if phase == 'valid' and valid_per_record and type_task=='classifier':
                valids_preds_.append(one_record_valid_predictions)
                valid_targets_.append(one_record_valid_targets)


                #corr_, tot_ = accuracy_per_segments(valids_preds_, valid_targets_, n_seg=3, percent=0.5)
                corr_, tot_ = accuracy_per_segments_detection(valids_preds_, valid_targets_, n_seg=3, min_detect=2)

            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                epoch_loss = running_loss / train_num_samples
                train_losses.append(epoch_loss)
                if type_task=='classifier':
                    epoch_acc = running_corrects / train_num_samples
                    train_accs.append(epoch_acc.item())
            else:
                epoch_loss = running_loss / val_num_samples
                if type_task=='classifier':
                    if valid_per_record:
                        epoch_acc = corr_/tot_
                    else:
                        epoch_acc = running_corrects / val_num_samples
                    valid_accs.append(epoch_acc)
                valid_losses.append(epoch_loss)

            if type_task == 'classifier':
                print('[{}];   Acc.: {:.4};   Loss: {:.4f}.'.format(phase, epoch_acc, epoch_loss))
            else:
                print('[{}];   Loss: {:.4f}.'.format(phase, epoch_loss))

            # deep copy the model
            if type_task=='classifier':
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
            else:
                if phase == 'valid' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    if type_task == 'classifier':
        print('Best val Acc: {:4f}'.format(best_acc))
    else:
        print('Lowest val Loss: {:4f}'.format(best_loss))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    if type_task == 'classifier':
        return model, (train_accs, valid_accs), (train_losses, valid_losses), pd.DataFrame(train_log), pd.DataFrame(
            valid_log), best_epoch
    else:
        return model, (train_losses, valid_losses), pd.DataFrame(train_log), pd.DataFrame(
            valid_log), best_epoch


##################################################################

def train_scratch_model_per_epoch(model, criterion0, criterion1, optimizer, dataloaders, device, num_epochs, valid_rec_names, valid_len,
                        valid_per_record, extra_aug, use_clip_grad, n_divisions_segments=4, n_outputs=1):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_acc = 0.0
    best_loss = 999.999

    train_accs_per_sample = []
    valid_accs_per_sample = []
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

            num_samples = 0

            logits_ = []
            preds_ = []
            targets_ = []
            names_ = []
            valid_name_analyze = '0'
            # Batch iterations
            for inputs, labels, rec_info in dataloaders[phase]:
                it += 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Extra augmentation?
                if extra_aug:
                    for batch_sample in range(inputs.shape[0]):
                        if decision(0.2):
                            inputs[batch_sample, :, :] = round(random.uniform(0.8, 1.2), 2) * inputs[batch_sample, :, :]
                        elif decision(0.2):
                            inputs[batch_sample, :, :] = -1 * inputs[batch_sample, :, :]


                # Forward
                # track history only if train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if n_outputs == 1:
                        labels = labels.to(torch.float64)
                        ##loss = criterion(outputs.squeeze(1), labels)
                        prepreds = torch.sigmoid(outputs)
                        preds = (prepreds >= 0.4).long().squeeze(1)
                    else:
                        labels = labels.to(torch.int64)
                        #loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                    targets_.append(labels)
                    logits_.append(outputs)
                    preds_.append(preds.detach().cpu().numpy())
                    names_.append(rec_info)

                    num_samples += 1 * inputs.size(0)

                    # backward + optimize only if in train phase
                    if len(dataloaders[phase]) > 100:
                        if it % 50 == 0:
                            print('iteration nb. {}'.format(it))
                    else:
                        if it % 5 == 0:
                            print('iteration nb. {}'.format(it))

            zero_logits_ = []
            zero_targets_ = []
            zero_preds_ = []
            zero_names_ = []
            one_logits_ = []
            one_targets_ = []
            one_preds_ = []
            one_names_ = []
            for batch_i in range(len(targets_)):
                for elem in range(len(targets_[batch_i])):
                    if targets_[batch_i][elem] == 0:
                        if n_outputs == 1:
                            zero_logits_.append(logits_[batch_i][elem])
                        else:
                            zero_logits_.append(logits_[batch_i][elem].unsqueeze(0))
                        zero_preds_.append(preds_[batch_i][elem])
                        zero_targets_.append(targets_[batch_i][elem].unsqueeze(0))
                        zero_names_.append(names_[batch_i][elem])
                    elif targets_[batch_i][elem] == 1:
                        if n_outputs == 1:
                            one_logits_.append(logits_[batch_i][elem])
                        else:
                            one_logits_.append(logits_[batch_i][elem].unsqueeze(0))
                        one_preds_.append(preds_[batch_i][elem])
                        one_names_.append(names_[batch_i][elem])
                        one_targets_.append(targets_[batch_i][elem].unsqueeze(0))
                    else:
                        assert 1 == 0

            # Zero the parameter gradients
            optimizer.zero_grad()
            assert torch.tensor(one_targets_).float().mean().item() == 1.
            loss0 = criterion0(torch.cat(zero_logits_), torch.cat(zero_targets_))
            #loss1 = criterion1(torch.cat(one_logits_).mean(), torch.cat(one_targets_).float().mean().long())
            loss1 = criterion1(torch.cat(one_logits_), torch.cat(one_targets_))
            loss = loss0 + loss1

            if phase == 'train':
                loss.backward()
                if use_clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

            # Accuracy computation
            if valid_per_record:
                tot_ = 0
                corr_ = 0
                names_ = zero_names_ + one_names_
                each_rec_name = list(set(names_))
                preds_ = zero_preds_ + one_preds_
                targets_ = np.concatenate((torch.tensor(zero_targets_).detach().numpy(), torch.tensor(one_targets_).detach().numpy()))
                for this_rec_name in each_rec_name:
                    this_indexes = [i for i in range(len(names_)) if names_[i] == this_rec_name]
                    this_preds = np.array(preds_)[this_indexes]
                    assert all_same(targets_[this_indexes])
                    this_target = targets_[this_indexes].mean()
                    len_subsegment_ = len(this_indexes)//3
                    index_count_ = 0
                    for j in range(3):
                        subsegment_preds_ = this_preds[index_count_:(index_count_+len_subsegment_)]
                        subsegments_corrs_ = 0
                        for pred_ii in subsegment_preds_:
                            if pred_ii == this_target:
                                subsegments_corrs_ += 1
                        if subsegments_corrs_/len_subsegment_ >= 0.5:
                            corr_ += 1
                        tot_ += 1
                        index_count_ += len_subsegment_

            accc = corr_/tot_
            acc_per_sample = (np.sum(preds_ == targets_) / len(preds_))
            if phase == 'train':
                train_losses.append(loss.item())
                train_accs.append(accc)
                train_accs_per_sample.append(acc_per_sample)
            else:
                valid_losses.append(loss.item())
                valid_accs.append(accc)
                valid_accs_per_sample.append(acc_per_sample)


            print('[{}];   Acc.: {:.4};   Loss: {:.4f}.'.format(phase, accc, loss))
            if phase == 'valid' and accc > best_acc:
                best_acc = accc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, (train_accs, valid_accs), (train_losses, valid_losses), best_epoch, (train_accs_per_sample, valid_accs_per_sample)
