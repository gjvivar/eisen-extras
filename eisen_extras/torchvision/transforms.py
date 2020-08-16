import numpy as np
from typing import List
from PIL.Image import Image

import torch
from torchvision import transforms as tvt


class PILImageTorchvisionTransforms:
    r"""Base class for torchvision transforms on PIL Image input."""
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
        if not isinstance(instance, self.expected_type):
            raise TypeError('Input must be of type PIL Image. Type %s is the current input.' % type(instance))


class CenterCrop(PILImageTorchvisionTransforms):
    r"""Crop given input at the center.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import CenterCrop
        transform = CenterCrop(['input'], size=10)
        output = transform(data)

    """
    def __init__(self, fields: List[str], *args, **kwargs):
        r"""
        :param fields: list of keynames in data dictionary to work on
        :type fields: list of str
        :param \*args: Additional argument list for this transform.
        :param \**kwargs: Additional keyword arguments for this transform.
        """
        transform = tvt.CenterCrop(*args, **kwargs)
        super(CenterCrop, self).__init__(fields, transform)


class ColorJitter(PILImageTorchvisionTransforms):
    r"""Randomly change the brightness, contrast and saturation of a given PIL Image.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import ColorJitter
        transform = ColorJitter(['input'], brightness=0, contrast=0, saturation=0, hue=0)
        output = transform(data)

    """
    def __init__(self, fields: List[str], *args, **kwargs):
        r"""
        :param fields: list of keynames in data dictionary to work on
        :type fields: list of str
        :param \*args: Additional argument list for this transform.
        :param \**kwargs: Additional keyword arguments for this transform.
        """
        transform = tvt.ColorJitter(*args, **kwargs)
        super(ColorJitter, self).__init__(fields, transform)


class FiveCrop(PILImageTorchvisionTransforms):
    r"""Crop PIL Image into four corners and its central crop.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import FiveCrop
        transform = FiveCrop(['input'], size=4)
        output = transform(data)

    """
    def __init__(self, fields: List[str], *args, **kwargs):
        r"""
        :param fields: list of keynames in data dictionary to work on
        :type fields: list of str
        :param \*args: Additional argument list for this transform.
        :param \**kwargs: Additional keyword arguments for this transform.
        """
        transform = tvt.FiveCrop(*args, **kwargs)
        super(FiveCrop, self).__init__(fields, transform)


class Grayscale(PILImageTorchvisionTransforms):
    r"""Convert given PIL image to grayscale.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import Grayscale
        transform = Grayscale(['input'], num_output_channels=1)
        output = transform(data)

    """
    def __init__(self, fields: List[str], *args, **kwargs):
        r"""
        :param fields: list of keynames in data dictionary to work on
        :type fields: list of str
        :param \*args: Additional argument list for this transform.
        :param \**kwargs: Additional keyword arguments for this transform.
        """
        transform = tvt.Grayscale(*args, **kwargs)
        super(Grayscale, self).__init__(fields, transform)


class Pad(PILImageTorchvisionTransforms):
    r"""Pad given PIL Image on all sides.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import Pad
        transform = Pad(['input'], padding, fill=0, padding_mode='constant')
        output = transform(data)

    """
    def __init__(self, fields: List[str], *args, **kwargs):
        r"""
        :param fields: list of keynames in data dictionary to work on
        :type fields: list of str
        :param \*args: Additional argument list for this transform.
        :param \**kwargs: Additional keyword arguments for this transform.
        """
        transform = tvt.Pad(*args, **kwargs)
        super(Pad, self).__init__(fields, transform)


class RandomAffine(PILImageTorchvisionTransforms):
    r"""Random affine transformation.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import RandomAffine
        transform = RandomAffine(['input'], degrees, translate=None, scale=None, shear=None, resample=False,
                                 fillcolor=0)
        output = transform(data)

    """
    def __init__(self, fields: List[str], *args, **kwargs):
        r"""
        :param fields: list of keynames in data dictionary to work on
        :type fields: list of str
        :param \*args: Additional argument list for this transform.
        :param \**kwargs: Additional keyword arguments for this transform.
        """
        transform = tvt.RandomAffine(*args, **kwargs)
        super(RandomAffine, self).__init__(fields, transform)


class RandomApply(tvt.RandomApply):
    r"""Randomly apply a list of transformations with a given probability of p.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import RandomApply
        transform = RandomApply(list_of_transforms, p=0.5)
        output = transform(data)

    """

    def __init__(self, transforms, p=0.5):
        r"""
        :param transforms: list of transforms
        :type transforms: list of PILImageTorchvisionTransforms or TensorTorchvisionTransforms
        :param p: probability
        :type p: float
        """
        super(RandomApply, self).__init__(transforms, p=p)


class RandomChoice(tvt.RandomChoice):
    r"""Randomly pick a single transformation from a given list of transformation then apply this transformation.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import RandomChoice
        transform = RandomChoice(list_of_transforms)
        output = transform(data)

    """

    def __init__(self, transforms):
        r"""
        :param transforms: list of transforms
        :type transforms: list of PILImageTorchvisionTransforms or TensorTorchvisionTransforms
        """
        super(RandomChoice, self).__init__(transforms)


class RandomOrder(tvt.RandomOrder):
    r"""Apply in a random order a given list of transformations.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import RandomOrder
        transform = RandomOrder(list_of_transforms)
        output = transform(data)

    """
    def __init__(self, transforms):
        r"""
        :param transforms: list of transforms
        :type transforms: list of PILImageTorchvisionTransforms or TensorTorchvisionTransforms
        """
        super(RandomOrder, self).__init__(transforms)


class RandomCrop(PILImageTorchvisionTransforms):
    r"""Crop the given PIL Image at a random location.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import RandomCrop
        transform = RandomCrop(['input'], size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
        output = transform(data)

    """
    def __init__(self, fields: List[str], *args, **kwargs):
        r"""
        :param fields: list of keynames in data dictionary to work on
        :type fields: list of str
        :param \*args: Additional argument list for this transform.
        :param \**kwargs: Additional keyword arguments for this transform.
        """
        transform = tvt.RandomCrop(*args, **kwargs)
        super(RandomCrop, self).__init__(fields, transform)


class RandomGrayscale(PILImageTorchvisionTransforms):
    r"""Randomly convert given PIL Image to grayscale with a given probability of p.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import RandomGrayscale
        transform = RandomGrayscale(['input'], p=0.1)
        output = transform(data)

    """
    def __init__(self, fields: List[str], *args, **kwargs):
        r"""
        :param fields: list of keynames in data dictionary to work on
        :type fields: list of str
        :param \*args: Additional argument list for this transform.
        :param \**kwargs: Additional keyword arguments for this transform.
        """
        transform = tvt.RandomGrayscale(*args, **kwargs)
        super(RandomGrayscale, self).__init__(fields, transform)


class RandomHorizontalFlip(PILImageTorchvisionTransforms):
    r"""Horizontally flip the given PIL Image randomly with a given probability of p.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import RandomHorizontalFlip
        transform = RandomHorizontalFlip(['input'], p=0.5)
        output = transform(data)

    """
    def __init__(self, fields: List[str], *args, **kwargs):
        r"""
        :param fields: list of keynames in data dictionary to work on
        :type fields: list of str
        :param \*args: Additional argument list for this transform.
        :param \**kwargs: Additional keyword arguments for this transform.
        """
        transform = tvt.RandomHorizontalFlip(*args, **kwargs)
        super(RandomHorizontalFlip, self).__init__(fields, transform)


class RandomPerspective(PILImageTorchvisionTransforms):
    r"""Random Perspective transformation of given PIL Image with a given probability of p.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import RandomPerspective
        transform = RandomPerspective(['input'], distortion_scale=0.5, p=0.5, interpolation=Image.BICUBIC, fill=0)
        output = transform(data)

    """
    def __init__(self, fields: List[str], *args, **kwargs):
        r"""
        :param fields: list of keynames in data dictionary to work on
        :type fields: list of str
        :param \*args: Additional argument list for this transform.
        :param \**kwargs: Additional keyword arguments for this transform.
        """
        transform = tvt.RandomPerspective(*args, **kwargs)
        super(RandomPerspective, self).__init__(fields, transform)


class RandomResizedCrop(PILImageTorchvisionTransforms):
    r"""Crop the given PIL Image to random size and aspect ratio.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import RandomResizedCrop
        transform = RandomResizedCrop(['input'], size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                                      interpolation=PIL.Image.BILINEAR)
        output = transform(data)

    """
    def __init__(self, fields: List[str], *args, **kwargs):
        r"""
        :param fields: list of keynames in data dictionary to work on
        :type fields: list of str
        :param \*args: Additional argument list for this transform.
        :param \**kwargs: Additional keyword arguments for this transform.
        """
        transform = tvt.RandomResizedCrop(*args, **kwargs)
        super(RandomResizedCrop, self).__init__(fields, transform)


class RandomRotation(PILImageTorchvisionTransforms):
    r"""Rotate given PIL Image in degrees.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import RandomRotation
        transform = RandomRotation(['input'], degrees, resample=False, expand=False, center=None, fill=None)
        output = transform(data)

    """
    def __init__(self, fields: List[str], *args, **kwargs):
        r"""
        :param fields: list of keynames in data dictionary to work on
        :type fields: list of str
        :param \*args: Additional argument list for this transform.
        :param \**kwargs: Additional keyword arguments for this transform.
        """
        transform = tvt.RandomRotation(*args, **kwargs)
        super(RandomRotation, self).__init__(fields, transform)


class RandomVerticalFlip(PILImageTorchvisionTransforms):
    r"""Vertically flip the given PIL Image randomly with a given probability of p.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import RandomVerticalFlip
        transform = RandomVerticalFlip(['input'], p=0.5)
        output = transform(data)

    """
    def __init__(self, fields: List[str], *args, **kwargs):
        r"""
        :param fields: list of keynames in data dictionary to work on
        :type fields: list of str
        :param \*args: Additional argument list for this transform.
        :param \**kwargs: Additional keyword arguments for this transform.
        """
        transform = tvt.RandomVerticalFlip(*args, **kwargs)
        super(RandomVerticalFlip, self).__init__(fields, transform)


class Resize(PILImageTorchvisionTransforms):
    r"""Resized given PIL Image to given size.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import Resize
        transform = Resize(['input'], size, interpolation=PIL.Image.BILINEAR)
        output = transform(data)

    """
    def __init__(self, fields: List[str], *args, **kwargs):
        r"""
        :param fields: list of keynames in data dictionary to work on
        :type fields: list of str
        :param \*args: Additional argument list for this transform.
        :param \**kwargs: Additional keyword arguments for this transform.
        """
        transform = tvt.Resize(*args, **kwargs)
        super(Resize, self).__init__(fields, transform)


class TenCrop(PILImageTorchvisionTransforms):
    r"""Crop given PIL Image into ten parts, four corners and the central crop plus the flipped version of these.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import TenCrop
        transform = TenCrop(['input'], size, vertical_flip=False)
        output = transform(data)

    """
    def __init__(self, fields: List[str], *args, **kwargs):
        r"""
        :param fields: list of keynames in data dictionary to work on
        :type fields: list of str
        :param \*args: Additional argument list for this transform.
        :param \**kwargs: Additional keyword arguments for this transform.
        """
        transform = tvt.TenCrop(*args, **kwargs)
        super(TenCrop, self).__init__(fields, transform)


class ToTensor(PILImageTorchvisionTransforms):
    r"""Convert given PIL Image or numpy.ndarray to tensor.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import ToTensor
        transform = ToTensor(['input'], img)
        output = transform(data)

    """
    def __init__(self, fields: List[str]):
        r"""
        :param fields: list of keynames in data dictionary to work on
        :type fields: list of str
        """

        transform = tvt.ToTensor()
        super(ToTensor, self).__init__(fields, transform)
        self.expected_type = (Image, np.ndarray)

    def _check_data_type(self, instance):
        if not isinstance(instance, self.expected_type):
            raise TypeError('Input must be of type PIL.Image or numpy.ndarray. Type %s is the current input.' % type(
                instance))


class TensorTorchvisionTransforms:
    """Base class for torchvision transforms on torch.*Tensor input.
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
        if not isinstance(instance, self.expected_type):
            raise TypeError('Input must be of type torch Tensor. Type %s is the current input.' % type(instance))


class LinearTransformation(TensorTorchvisionTransforms):
    r"""Transform given tensor image with a pre-calculated square transformation matrix and a mean_vector.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import LinearTransformation
        transform = LinearTransformation(['input'], transformation_matrix, mean_vector)
        output = transform(data)

    """
    def __init__(self, fields: List[str], *args, **kwargs):
        r"""
        :param fields: list of keynames in data dictionary to work on
        :type fields: list of str
        :param \*args: Additional argument list for this transform.
        :param \**kwargs: Additional keyword arguments for this transform.
        """
        transform = tvt.LinearTransformation(*args, **kwargs)
        super(LinearTransformation, self).__init__(fields, transform)


class Normalize(TensorTorchvisionTransforms):
    r"""Normalize a given tensor image using given mean and standard deviation.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import Normalize
        transform = Normalize(['input'], mean, std, inplace=False)
        output = transform(data)

    """
    def __init__(self, fields: List[str], *args, **kwargs):
        r"""
        :param fields: list of keynames in data dictionary to work on
        :type fields: list of str
        :param \*args: Additional argument list for this transform.
        :param \**kwargs: Additional keyword arguments for this transform.
        """
        transform = tvt.Normalize(*args, **kwargs)
        super(Normalize, self).__init__(fields, transform)


class RandomErasing(TensorTorchvisionTransforms):
    r"""Randomly erase a rectangular region in an image.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import RandomErasing
        transform = RandomErasing(['input'], p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
        output = transform(data)

    """
    def __init__(self, fields: List[str], *args, **kwargs):
        r"""
        :param fields: list of keynames in data dictionary to work on
        :type fields: list of str
        :param \*args: Additional argument list for this transform.
        :param \**kwargs: Additional keyword arguments for this transform.
        """
        transform = tvt.RandomErasing(*args, **kwargs)
        super(RandomErasing, self).__init__(fields, transform)


class ToPILImage(TensorTorchvisionTransforms):
    r"""Convert given tensor.*Tensor or numpy.ndarray to PIL Image.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import ToPILImage
        transform = ToPILImage(['input'], mode=None)
        output = transform(data)

    """
    def __init__(self, fields: List[str], *args, **kwargs):
        r"""
        :param fields: list of keynames in data dictionary to work on
        :type fields: list of str
        :param \*args: Additional argument list for this transform.
        :param \**kwargs: Additional keyword arguments for this transform.
        """
        transform = tvt.ToPILImage(*args, **kwargs)
        super(ToPILImage, self).__init__(fields, transform)
        self.expected_type = (torch.Tensor, np.ndarray)


class Lambda(tvt.Lambda):
    r"""User-defined lambda function as a transform.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import Lambda
        input_tensor = torch.ones(3, 32, 32, dtype=torch.uint8).random_(0, 255)
        input_data = {'input': input_tensor}
        transform = x_transforms.Compose([
            x_transforms.ToPILImage(['input']),
            x_transforms.TenCrop(['input'], 3),
            x_transforms.Lambda(['input'], lambda crops: [tvt.ToTensor()(crop) for crop in crops]),
        ])
        output = transform(input_data)

    """
    def __init__(self, fields, fn):
        r"""
        # :param fields: list of keynames in data dictionary to work on
        # :type fields: list of str
        :param fn: Lambda/function to be used for transform
        :type fn: function
        """
        super(Lambda, self).__init__(fn)
        self.fields = fields

    def __call__(self, data: dict) -> dict:
        for field in self.fields:
            data[field] = self.lambd(data[field])
        return data


class Compose(tvt.Compose):
    r"""Compose given list of transforms together.

    .. code-block:: python

        from eisen_extras.torchvision.transforms import Compose
        input_tensor = torch.ones(3, 32, 32, dtype=torch.uint8).random_(0, 255)
        input_data = {'input': input_tensor}
        transform = x_transforms.Compose([
            x_transforms.ToPILImage(['input']),
            x_transforms.TenCrop(['input'], 3),
            x_transforms.Lambda(['input'], lambda crops: [tvt.ToTensor()(crop) for crop in crops]),
        ])
        output = transform(input_data)

    """
    def __init__(self, t_list):
        r"""
        :param t_list: List of transforms
        :type t_list: list of PILImageTorchvisionTransforms or TensorTorchvisionTransforms
        """
        super(Compose, self).__init__(t_list)
