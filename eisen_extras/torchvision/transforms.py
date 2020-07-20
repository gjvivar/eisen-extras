import numpy as np
import torch
from typing import List

from PIL.Image import Image
from torchvision import transforms as tvt


class PILImageTorchvisionTransforms:
    """# TODO Torchvision transforms for PIL Image input."""
    def __init__(self, fields: List[str], transform):
        self.fields = fields
        self.transform = transform
        self.expected_type = Image

    def __call__(self, data: dict) -> dict:
        for field in self.fields:
            self._check_data_type(data[field])
            data[field] = self.transform(data[field])
        return data

    def _check_data_type(self, instance):
        """# TODO"""
        if not isinstance(instance, self.expected_type):
            raise TypeError('Input must be of type PIL Image. Type %s is the current input.' % type(instance))


class CenterCrop(PILImageTorchvisionTransforms):
    """# TODO"""
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.CenterCrop(*args, **kwargs)
        super(CenterCrop, self).__init__(fields, transform)


class ColorJitter(PILImageTorchvisionTransforms):
    """# TODO"""
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.ColorJitter(*args, **kwargs)
        super(ColorJitter, self).__init__(fields, transform)


class FiveCrop(PILImageTorchvisionTransforms):
    """# TODO"""
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.FiveCrop(*args, **kwargs)
        super(FiveCrop, self).__init__(fields, transform)


class Grayscale(PILImageTorchvisionTransforms):
    """# TODO"""
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.Grayscale(*args, **kwargs)
        super(Grayscale, self).__init__(fields, transform)


class Pad(PILImageTorchvisionTransforms):
    """# TODO"""
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.Pad(*args, **kwargs)
        super(Pad, self).__init__(fields, transform)


class RandomAffine(PILImageTorchvisionTransforms):
    """# TODO"""
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.RandomAffine(*args, **kwargs)
        super(RandomAffine, self).__init__(fields, transform)


class RandomApply(PILImageTorchvisionTransforms):
    """# TODO"""
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.RandomApply(*args, **kwargs)
        super(RandomApply, self).__init__(fields, transform)


class RandomChoice(PILImageTorchvisionTransforms):
    """# TODO"""
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.RandomChoice(*args, **kwargs)
        super(RandomChoice, self).__init__(fields, transform)


class RandomCrop(PILImageTorchvisionTransforms):
    """# TODO"""
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.RandomCrop(*args, **kwargs)
        super(RandomCrop, self).__init__(fields, transform)


class RandomGrayscale(PILImageTorchvisionTransforms):
    """# TODO"""
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.RandomGrayscale(*args, **kwargs)
        super(RandomGrayscale, self).__init__(fields, transform)


class RandomHorizontalFlip(PILImageTorchvisionTransforms):
    """# TODO"""
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.RandomHorizontalFlip(*args, **kwargs)
        super(RandomHorizontalFlip, self).__init__(fields, transform)


class RandomPerspective(PILImageTorchvisionTransforms):
    """# TODO"""
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.RandomPerspective(*args, **kwargs)
        super(RandomPerspective, self).__init__(fields, transform)


class RandomResizedCrop(PILImageTorchvisionTransforms):
    """# TODO"""
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.RandomResizedCrop(*args, **kwargs)
        super(RandomResizedCrop, self).__init__(fields, transform)


class RandomRotation(PILImageTorchvisionTransforms):
    """# TODO"""
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.RandomRotation(*args, **kwargs)
        super(RandomRotation, self).__init__(fields, transform)


class RandomSizedCrop(PILImageTorchvisionTransforms):
    """# TODO"""
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.RandomSizedCrop(*args, **kwargs)
        super(RandomSizedCrop, self).__init__(fields, transform)


class RandomVerticalFlip(PILImageTorchvisionTransforms):
    """# TODO"""
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.RandomVerticalFlip(*args, **kwargs)
        super(RandomVerticalFlip, self).__init__(fields, transform)


class Resize(PILImageTorchvisionTransforms):
    """# TODO"""
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.Resize(*args, **kwargs)
        super(Resize, self).__init__(fields, transform)


class Scale(PILImageTorchvisionTransforms):
    """# TODO"""
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.Scale(*args, **kwargs)
        super(Scale, self).__init__(fields, transform)


class TenCrop(PILImageTorchvisionTransforms):
    """# TODO"""
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.TenCrop(*args, **kwargs)
        super(TenCrop, self).__init__(fields, transform)


class ToTensor(PILImageTorchvisionTransforms):
    """# TODO"""
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.ToTensor(*args, **kwargs)
        super(ToTensor, self).__init__(fields, transform)
        self.expected_type = (Image, np.ndarray)

    def _check_data_type(self, instance):
        """# TODO"""
        if not isinstance(instance, self.expected_type):
            raise TypeError('Input must be of type PIL.Image or numpy.ndarray. Type %s is the current input.' % type(
                instance))


class TensorTorchvisionTransforms:
    """# TODO Torchvision transforms for torch.tensor input.
    """
    def __init__(self, fields, transform):
        self.fields = fields
        self.transform = transform
        self.expected_type = torch.Tensor

    def __call__(self, data: dict) -> dict:
        for field in self.fields:
            self._check_data_type(data[field])
            data[field] = self.transform(data[field])
        return data

    def _check_data_type(self, instance):
        """# TODO"""
        if not isinstance(instance, self.expected_type):
            raise TypeError('Input must be of type torch Tensor. Type %s is the current input.' % type(instance))


class LinearTransformation(TensorTorchvisionTransforms):
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.LinearTransformation(*args, **kwargs)
        super(LinearTransformation, self).__init__(fields, transform)


class Normalize(TensorTorchvisionTransforms):
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.Normalize(*args, **kwargs)
        super(Normalize, self).__init__(fields, transform)


class RandomErasing(TensorTorchvisionTransforms):
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.RandomErasing(*args, **kwargs)
        super(RandomErasing, self).__init__(fields, transform)


class ToPILImage(TensorTorchvisionTransforms):
    def __init__(self, fields: List[str], *args, **kwargs):
        transform = tvt.ToPILImage(*args, **kwargs)
        super(ToPILImage, self).__init__(fields, transform)
        self.expected_type = (torch.Tensor, np.ndarray)
