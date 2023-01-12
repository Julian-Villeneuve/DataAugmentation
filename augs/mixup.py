from augs.augmentation import Augmentation
import torch
from typing import Union


class Mixup(Augmentation):
    def __init__(
        self,
        batch_size: int,
        do_prob: float,
        sequence_shape: Union[list, tuple],
        linear_mix_min: float = 0.1,
        linear_mix_max: float = 0.5,
    ):
        """
        Linear Mix of two random MTS within the batch, for each MTS within the batch, with chance based on do_prob

        Args:
                batch_size:
                do_prob:
                linear_mix_min:
                linear_mix_max:
                sequence_shape: in the form of [Length, Channels]
        """

        super().__init__()
        self.linear_mix_max = linear_mix_max
        self.linear_mix_min = linear_mix_min
        self.sequence_shape = sequence_shape
        self.batch_size = batch_size
        self.do_prob = do_prob

    def call(self, example: dict) -> dict:

        array_do_aug = torch.rand( self.batch_size ) < self.do_prob

        take_from_mixup_addition_percentage = (torch.rand( 1 ) * ( self.linear_mix_max - self.linear_mix_min)) + self.linear_mix_min
        take_from_mixup_addition_percentage = array_do_aug * take_from_mixup_addition_percentage

        x = example["input"]
        original_input = x

        perm = torch.randperm( x.shape[ 0 ] )
        mixup_addition = original_input[ [1, 0], : ]

        original_input = original_input.permute( 1, 2, 3, 4, 0 )
        mixup_addition = mixup_addition.permute( 1, 2, 3, 4, 0 )

        

        x = original_input * (1 - take_from_mixup_addition_percentage) + mixup_addition * (
            take_from_mixup_addition_percentage
        )

        x = x.permute( 4, 0, 1, 2, 3 )

        example["input"] = x

        return example

    def singular_call(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError
