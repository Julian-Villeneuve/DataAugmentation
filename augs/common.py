import torch

def resize_time_series(series, new_length, method='bilinear'):
    '''
    Resize a time series, as if it was a 1 dimensional image

    '''
    image_like = series.unsqueeze(dim = 0)
    return torch.nn.functional.interpolate( image_like, size = [ 1, new_length ], mode = method )[ 0 ]


def pad_to_length(series, target_length):
    return torch.pad(series, [[0, max(target_length - series.shape[0],0) ], [0, 0]])


def cut_time_series(series, cut_start, cut_end, insert=None):

    input_length = series.shape[0]
    to_concat = [series[:cut_start], insert, series[cut_end:]]

    series = torch.concat([x for x in to_concat if not x is None], axis = 0)
    series = pad_to_length(series, input_length)
    return series

def check_proba(proba, shape = None):
    if shape is None:
        shape = [1]
    return torch.rand(shape) <= proba