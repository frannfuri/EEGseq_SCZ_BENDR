import torch
import time
import copy
import sklearn.metrics as skmetrics
import pandas as pd
from collections import OrderedDict
from utils import modified_mae

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs):
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
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

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
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs[0], 1)
                    labels = labels.to(torch.int64)
                    loss = criterion(outputs[0], labels)

                    # backward + optimize only if in train phase
                    if phase == 'train':
                        if it % 5 == 0:
                            print('iteration nb. {}'.format(it))
                        train_metrics = OrderedDict()
                        train_metrics['epoch'] = epoch
                        train_metrics['iter'] = it
                        it += 1
                        train_metrics['loss'] = loss.item()
                        # TODO: REVISAR accuracy !!
                        train_metrics['accuracy'] = torch.sum(preds == labels.data) / inputs.size(0)
                        # TODO: REVISAR F1score
                        train_metrics['f1score'], train_metrics['preciss'], train_metrics['recall'] = f1_loss(
                                                                                                    labels,
                                                                                                    preds)
                        # TODO: WHY .PARAM_GROUPS[0][lr]?
                        train_metrics['lr'] = optimizer.param_groups[0]['lr']
                        loss.backward()
                        optimizer.step()
                        train_log.append(train_metrics)
                        train_num_samples += 1 * inputs.size(0)
                    else:
                        valid_metrics = OrderedDict()
                        valid_metrics['epoch'] = epoch
                        valid_metrics['loss'] = loss.item()
                        valid_metrics['accuracy'] = torch.sum(preds == labels.data) / inputs.size(0)
                        valid_metrics['f1score'], valid_metrics['preciss'], valid_metrics['recall'] = f1_loss(
                                                                                                    labels,
                                                                                                    preds)
                        valid_log.append(valid_metrics)
                        val_num_samples += 1 + inputs.size(0)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                # TODO: Probar de la otra manera
                scheduler.step()


            if phase == 'train':
                epoch_loss = running_loss / train_num_samples
                epoch_acc = running_corrects / train_num_samples
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
            else:
                epoch_loss = running_loss / val_num_samples
                epoch_acc = running_corrects / val_num_samples
                valid_losses.append(epoch_loss)
                valid_accs.append(epoch_acc)

            print('[{}];   Acc.: {:.4};   Loss: {:.4f}.'.format(phase, epoch_acc, epoch_loss))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc: #/ val_num_samples > best_acc:
                best_acc = epoch_acc / val_num_samples
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