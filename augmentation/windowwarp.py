from augmentation import Augmentation
from augmentation.common import resize_time_series, cut_time_series, check_proba
import torch
from typing import Union

class WindowWarp(Augmentation):
    def __init__(
            self,
            batch_size: int,
            do_prob: float,
            sequence_shape: Union[list, tuple],
            min_window_size: int,
            max_window_size: int,
            scale_factor: float,
            method: str = 'bilinear',
    ):
        """
        Use bilinear interpolation (as if it was an image) to resize a window and insert the window back.

        Args:
                min_window_size:
                max_window_size:
                batch_size:
                do_prob:
                sequence_shape: in the form of [Length, Channels]
        """

        super().__init__()
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.scale_factor = scale_factor
        self.sequence_shape = sequence_shape
        self.batch_size = batch_size
        self.do_prob = do_prob
        self.method = method

    def call(self, example: dict) -> dict:
        x = example["input"]

        # apply the function across a tensor with shape [batchsize] to return
        # a tensor with shape [batchsize, length, channels]
        x = torch.map_fn(self.singular_call, x, dtype = torch.float32) # TODO : map

        example["input"] = x

        return example

    def singular_call(self, input: torch.Tensor) -> torch.Tensor:
        if check_proba(self.do_prob):
            start, end = self.get_window()
            window_size = end - start

            target_window_size = torch.cast(torch.maximum(int((float(window_size) * self.scale_factor)), 2), torch.int64)

            window = input[start:end]
            window = resize_time_series(window, target_window_size, method = self.method)

            zeros = torch.zeros_like(input)[:torch.maximum(torch.cast(0, torch.int64), window_size - target_window_size)]

            input = torch.concat([input[:start], window, input[end:], zeros], axis = 0)[:self.sequence_shape[0]]

            input = torch.reshape(input, shape = self.sequence_shape)

        return input

    def get_window(self):
        # max val is exclusive
        start = (torch.FloatTensor().uniform_(0, self.sequence_shape[0] - self.max_window_size + 1)).type(torch.int64)
        end = start + (torch.FloatTensor().uniform_(self.min_window_size, self.max_window_size + 1)).type(torch.int64)
        
        return start, end

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError