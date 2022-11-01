import numpy as np
import torch
import time
import random

__all__ = ['AverageMeter', 'time_string', 'convert_secs2time', 'time_file_str',
           'print_log', 'feature_extractor', 'EarlyStop']

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def time_file_str():
    ISOTIMEFORMAT = '%Y-%m-%d'
    string = '{}'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string + '-{}'.format(random.randint(1, 10000))


def print_log(print_string, log):
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()

def feature_extractor(x, model, target_layers):
    target_activations = list()
    for name, module in model._modules.items():
        x = module(x)
        if name in target_layers:
            target_activations += [x]
    return target_activations, x

def denormalization(x):
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    x = (x.transpose(1, 2, 0) * 255.).astype(np.uint8)
    return x


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


class EarlyStop():
    """Used to early stop the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=True, delta=0, save_name="checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            save_name (string): The filename with which the model and the optimizer is saved when improved.
                            Default: "checkpoint.pt"
        """
        self.patience = patience
        self.verbose = verbose
        self.save_name = save_name
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, optimizer, log):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, log)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print_log((f'EarlyStopping counter: {self.counter} out of {self.patience}'), log)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, log)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model, optimizer, log):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print_log((f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'),
                      log)
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, self.save_name)
        self.val_loss_min = val_loss