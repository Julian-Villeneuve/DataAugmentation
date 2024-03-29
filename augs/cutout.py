from augs.augmentation import Augmentation
import torch
from typing import Union


class Cutout(Augmentation):
    def __init__(
        self,
        batch_size: int,
        do_prob: float,
        sequence_shape: Union[list, tuple],
        min_cutout_len: int,
        max_cutout_len: int,
        channel_drop_prob: float,
    ):
        """
        Linear Mix of two random MTS within the batch, for each MTS within the batch, with chance based on do_prob

        Args:
                min_cutout_len:
                max_cutout_len:
                channel_drop_prob:
                batch_size:
                do_prob:
                sequence_shape: in the form of [Length, Channels]
        """

        super().__init__()
        self.max_cutout_len = max_cutout_len
        self.min_cutout_len = min_cutout_len
        self.sequence_shape = sequence_shape
        self.batch_size = batch_size
        self.do_prob = do_prob
        self.channel_drop_prob = channel_drop_prob

    def call(self, example: dict) -> dict:

        x = example["input"]

        # apply the function across a tensor with shape [batchsize] to return
        # a tensor with shape [batchsize, length, channels]

        # batch_cutout_masks = torch.zeros((self.batch_size,)).type(torch.float32) 
        # for i in range(batch_cutout_masks.shape[0]):
        #     batch_cutout_masks[i] = self.get_cutout_mask(batch_cutout_masks[i])
        batch_cutout_masks = self.get_cutout_mask( None )
        batch_cutout_masks = torch.stack( (batch_cutout_masks, self.get_cutout_mask( None )), 0 )
        
        x = x.permute( 4, 3, 0, 1, 2 )

        x = x * batch_cutout_masks

        x = x.permute( 2, 3, 4, 1, 0 )

        example["input"] = x

        return example

    def get_length_wise_cut_array(self):

        # generate an array representing timesteps, like [1, 2, 3...]
        time = torch.range(0, self.sequence_shape[0] - 1, dtype=torch.float32)

        # generate the start and end value
        start = torch.distributions.uniform.Uniform( 0, self.sequence_shape[0] - self.max_cutout_len + 1).sample()
        end = start + torch.distributions.uniform.Uniform( self.min_cutout_len, self.max_cutout_len ).sample()

        do = (torch.rand(1).item() < self.do_prob)

        # return 1 for values between start and end and 0 elsewhere
        return (torch.logical_and(time > start, time < end).type(torch.float32)) * do

    def get_channel_wise_cut_array(self):
        random = torch.rand( self.sequence_shape[ 1 ] ).type( torch.float32 )
        return random < self.channel_drop_prob 

    def get_cutout_mask(self, nothing: torch.Tensor) -> torch.Tensor:
        """

        Args:
            nothing: this is just a placeholder

        Returns:
            the cutout mask with shape = self.sequence_shape
        """

        timesteps = torch.reshape(self.get_length_wise_cut_array(), (self.sequence_shape[0], 1))
        channels = torch.reshape(self.get_channel_wise_cut_array(), (1, self.sequence_shape[1]))

        # timesteps * channels returns 1s where we want the cutout to occur
        # the mask is in the inverse, where we want 1s to represent where the cutout does not occur
        cutout_mask = (timesteps * channels < 0.99).type(torch.float32)
        return cutout_mask

    def singular_call(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError
