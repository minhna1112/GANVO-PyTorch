from __future__ import division
import shutil
import numpy as np
import torch
from path import Path
import datetime
from collections import OrderedDict
from matplotlib import cm


def save_path_formatter(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)

    # take __dict__ attribute or args
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    # initial Dictionary object with (key,value)
    keys_with_prefix = OrderedDict()
    keys_with_prefix['epoch_size'] = 'epoch_size'
    keys_with_prefix['sequence_length'] = 'seq'
    keys_with_prefix['rotation_mode'] = 'rot_'
    keys_with_prefix['padding_mode'] = 'padding_'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path(','.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    return save_path / timestamp


def save_checkpoint(save_path,ganvo_state, is_best, filename='checkpoint.pth.tar'):
    state = [ganvo_state]
    torch.save(state, save_path / 'GANVO_{}'.format(filename))

    if is_best:
        shutil.copyfile(save_path / 'GANVO_{}'.format(filename),
                        save_path / 'GANVO_model_best.pth.tar')
