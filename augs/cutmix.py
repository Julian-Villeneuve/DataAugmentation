from augs.augmentation import Augmentation
import torch
from typing import Union


class Cutmix(Augmentation):
    def __init__(
        self,
        batch_size: int,
        do_prob: float,
        sequence_shape: Union[list, tuple],
        min_cutmix_len: int,
        max_cutmix_len: int,
        channel_replace_prob: float,
    ):
        """
        For each MTS, select a section by location [length, channels] and replace it with another random MTS's same
        section by location.

        Args:
                min_cutmix_len:
                max_cutmix_len:
                channel_replace_prob:
                batch_size:
                do_prob:
                sequence_shape: in the form of [Length, Channels]
        """

        super().__init__()
        self.max_cutmix_len = max_cutmix_len
        self.min_cutmix_len = min_cutmix_len
        self.sequence_shape = sequence_shape
        self.batch_size = batch_size
        self.do_prob = do_prob
        self.channel_replace_prob = channel_replace_prob

    def call(self, example: dict) -> dict:
        x = example["input"]

        # apply the function across a tensor with shape [batchsize] to return
        # a tensor with shape [batchsize, length, channels]
        # batch_cutmix_masks = torch.zeros((self.batch_size,)).type(torch.float32)
        # for i in range(batch_cutmix_masks.shape[0]):
        #     batch_cutmix_masks[i] = self.get_cutout_mask(batch_cutmix_masks[i])
        #batch_cutmix_masks = map(self.get_cutmix_mask, torch.zeros((self.batch_size,))).type(torch.float32)

        batch_cutmix_masks = self.get_cutmix_mask( None )
        batch_cutmix_masks = torch.stack( (batch_cutmix_masks, self.get_cutmix_mask( batch_cutmix_masks )), 0 )
            
        batch_cutmix_masks.type( torch.float32 )

        # get a mixup addition sequence
        original_input = x
        #mixup_addition = tf.random.shuffle(x)
        idx = torch.randperm(x.nelement())

        perm = torch.randperm( original_input.shape[ 0 ] )
        mixup_addition = original_input[ perm, : ]

        

        original_input = original_input.permute( 4, 3, 0, 1, 2 )
        mixup_addition = mixup_addition.permute( 4, 3, 0, 1, 2 )

        # return original sequence where cutmixmask == 1 and mixup sequence otherwise
        aled = torch.where(batch_cutmix_masks == 1, original_input, mixup_addition)
        aled = aled.permute( 2, 3, 4, 1, 0 )

        example["input"] = aled
        return example

    def get_length_wise_cut_array(self):
        # generate an array representing timesteps, like [1, 2, 3...]
        time = torch.range(0, self.sequence_shape[0] - 1, dtype=torch.float32)

        # generate the start and end value
        #start = (torch.FloatTensor().uniform_(0, self.sequence_shape[0] - self.max_cutmix_len)).type(torch.int64)
        #end = start + (torch.FloatTensor().uniform_(self.min_cutmix_len, self.max_cutmix_len)).type(torch.int64)

        start = torch.distributions.uniform.Uniform( 0, self.sequence_shape[0] - self.max_cutmix_len + 1).sample()
        end = start + torch.distributions.uniform.Uniform( self.min_cutmix_len, self.max_cutmix_len ).sample()

        # do = torch.cast(
        #     torch.random.uniform(
        #         (),
        #     )
        #     < self.do_prob,
        #     torch.float32,
        # )
        do = (torch.rand(1).item() < self.do_prob)

        si = self.sequence_shape[ 0 ]
        startVec = torch.Tensor( si )
        startVec.fill_( start.item() )
        endVec = torch.Tensor( si )
        endVec.fill_( end.item() )

        # return 1 for values between start and end and 0 elsewhere
        s =  (torch.logical_and(time > startVec, time < endVec)).type(torch.float32)
        s *= (1 if do else 0)
        return s

    def get_channel_wise_cut_array(self):
        # generate an array representing which channels to cut
        #return (torch.rand((self.sequence_shape[1],)) < self.channel_replace_prob).type(torch.float32) 
        random = torch.rand( self.sequence_shape[ 1 ] ).type( torch.float32 )
        return random < self.channel_replace_prob 
        #return ((torch.FloatTensor(self.sequence_shape[1],)< self.channel_replace_prob).uniform_(0,1)).type(torch.float32) #aled

    def get_cutmix_mask(self, nothing: torch.Tensor) -> torch.Tensor:
        """

        Args:
            nothing: this is just a placeholder

        Returns:
            the cutmix mask with shape = self.sequence_shape
        """

        timesteps = torch.reshape(self.get_length_wise_cut_array(), (self.sequence_shape[0], 1))
        channels = torch.reshape(self.get_channel_wise_cut_array(), (1, self.sequence_shape[1]))

        # timesteps * channels returns 1s where we want the cutmix to occur
        # the mask is in the inverse, where we want 1s to represent where the cutmix does not occur
        cutmix_mask = (timesteps * channels < 0.99).type(torch.float32)
        return cutmix_mask

    def singular_call(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError
