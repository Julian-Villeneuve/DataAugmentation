from augs.augmentation import Augmentation
from augs.common import check_proba
import torch
from typing import Union


class RandomShifter(Augmentation):
    def __init__(self, shift_backward_max: int, shift_forward_max: int, sequence_shape: Union[list, tuple], do_prob=1.0):
        """
        Padded random shift of a sequence. A sequence is shifted forward or back by n timesteps by choosing n as a
        value between the forward max and backward max. If backward max is provided as a negative number, it will
        be converted to positive.

        Args:
                shift_backward_max:
                shift_forward_max:
                sequence_shape:
        """

        super().__init__()
        self.shift_forward_max = shift_forward_max
        self.shift_backward_max = abs(shift_backward_max)
        self.sequence_shape = sequence_shape
        self.do_prob = do_prob

    def call(self, example: dict) -> dict:
        input = example["input"]
        input = map(self.singular_call, input)
        example["input"] = input
        return example

    def singular_call(self, input: torch.Tensor) -> torch.Tensor:
        if check_proba(self.do_prob):
            start = self.get_start_position()
            input = self.shift(input, start)

        return input

    def shift(self, input: torch.Tensor, start: torch.Tensor) -> torch.Tensor:

        input = torch.nn.functional.pad(input, [[self.shift_backward_max, self.shift_forward_max], [0, 0]])
        input = input[start : start + self.sequence_shape[0]]
        return input

    def get_start_position(self) -> torch.Tensor:
        start = (torch.FloatTensor().uniform_(0, self.shift_backward_max + self.shift_forward_max)).type(torch.int64)

        return start

    @classmethod
    def from_config(cls, config):

        return cls(
            shift_backward_max=config.augment.random_shift.backward,
            shift_forward_max=config.augment.random_shift.forward,
            sequence_shape=config.model.input_shape,
        )
