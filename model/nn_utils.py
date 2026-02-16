import torch
import torch.nn as nn

class Conv():
    """ Wrapper for convolution operations. """
    def _conv_nd(dims, *args, **kwargs):
        """ Create a 1D, 2D, or 3D convolution module. """
        if dims == 1:
            return nn.Conv1d(*args, **kwargs)
        elif dims == 2:
            return nn.Conv2d(*args, **kwargs)
        elif dims == 3:
            return nn.Conv3d(*args, **kwargs)
        raise ValueError(f"unsupported dimensions: {dims}")

    def _zero_module(module):
        """ Zero out the parameters of a module and return it. """
        for p in module.parameters():
            p.detach().zero_()
        return module

    def zero_conv(dims, *args, **kwargs):
        """ Create a zero-initialized convolutional layer. """
        return Conv._zero_module(Conv._conv_nd(dims, *args, **kwargs))


class Masker:
    """ Wrapper to build masks. """
    @staticmethod
    def mask1d(size: int, prob: float = 0.0, device: str = "cpu") -> torch.Tensor:
        """ Basic Bernoulli mask. """
        if prob <= 0.0:
            return torch.ones(size, device=device)
        return (torch.rand(size, device=device) >= prob).float()