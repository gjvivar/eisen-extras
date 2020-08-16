import numpy as np
import pytest
import random

import torch
from torchvision import transforms as tvt

from eisen_extras.torchvision import transforms as x_transforms


class TestConversionTransforms:
    def test_to_tensor(self):
        input_tensor = torch.ones(3, 9, 9, dtype=torch.uint8).random_(0, 255)
        input_img = tvt.ToPILImage()(input_tensor)
        input_data = {'input': input_img}
        expected_output = input_tensor.float().div_(255)
        expected_output = expected_output

        transform = x_transforms.ToTensor(['input'])
        output = transform(input_data)
        output = output['input']
        assert torch.allclose(output, expected_output)

    def test_to_pil_image(self):
        input_tensor = torch.ones(3, 9, 9, dtype=torch.uint8).random_(0, 255)
        input_data = {'input': input_tensor}
        expected_output = input_tensor.numpy().transpose(1, 2, 0)

        transform = x_transforms.ToPILImage(['input'])
        output = transform(input_data)
        output = output['input']
        assert np.allclose(output, expected_output)


class TestGenericTransforms:
    def test_compose(self):
        input_tensor = torch.ones(1, 32, 32, dtype=torch.uint8).random_(0, 255)
        input_tensor_3 = torch.ones(3, 32, 32, dtype=torch.uint8).random_(0, 255)
        input_data = {'first_input': input_tensor, 'second_input': input_tensor_3}
        first_expected_output = input_tensor.float().div_(255)
        second_expected_output = tvt.ToPILImage()(input_tensor_3)

        transform = x_transforms.Compose([
            x_transforms.ToPILImage(['first_input']),
            x_transforms.ToTensor(['first_input']),
            x_transforms.ToPILImage(['second_input']),
        ])
        output = transform(input_data)

        assert len(output) == 2

        for i in ['first_input', 'second_input']:
            assert i in output

        assert torch.allclose(output['first_input'], first_expected_output)
        assert np.allclose(output['second_input'], second_expected_output)

    def test_lambda(self):
        input_tensor = torch.ones(3, 32, 32, dtype=torch.uint8).random_(0, 255)
        input_data = {'input': input_tensor}
        expected_output = input_tensor.float().div_(255)

        transform = x_transforms.Compose([
            x_transforms.ToPILImage(['input']),
            x_transforms.Lambda(['input'], lambda inp: tvt.ToTensor()(inp)),
        ])
        output = transform(input_data)
        output = output['input']
        assert torch.allclose(output, expected_output)


class TestPILImageTransforms:
    def test_check_data_type(self):
        with pytest.raises(TypeError):
            input_tensor = torch.rand((1, 5, 5))
            input_data = {'input': input_tensor}
            transform = x_transforms.ToTensor(['input'])
            transform(input_data)

    def test_center_crop(self):
        input_tensor = torch.ones(3, 9, 9, dtype=torch.uint8).random_(0, 255)
        input_data = {'input': input_tensor}
        expected_output = input_tensor.float().div_(255)
        expected_output = expected_output[:, 3:6, 3:6]
        transform = x_transforms.Compose([
            x_transforms.ToPILImage(['input']),
            x_transforms.CenterCrop(['input'], (3, 3)),
            x_transforms.ToTensor(['input']),
        ])
        output = transform(input_data)
        output = output['input']
        assert torch.allclose(output, expected_output)

    def test_color_jitter(self):
        input_tensor = torch.ones(3, 32, 32, dtype=torch.uint8).random_(0, 255)
        input_data = {'input': input_tensor}
        expected_output = input_tensor.float().div_(255)

        transform = x_transforms.Compose([
            x_transforms.ToPILImage(['input']),
            x_transforms.ColorJitter(['input'], brightness=0, contrast=0, saturation=0, hue=0),
            x_transforms.ToTensor(['input']),
        ])
        output = transform(input_data)
        output = output['input']
        assert torch.allclose(output, expected_output)

    def test_five_crop(self):
        input_tensor = torch.ones(3, 9, 9, dtype=torch.uint8).random_(0, 255)
        input_data = {'input': input_tensor}
        expected_output = input_tensor.float().div_(255)
        ul = expected_output[:, :3, :3]
        ur = expected_output[:, :3, -3:]
        ll = expected_output[:, -3:, :3]
        lr = expected_output[:, -3:, -3:]
        c = expected_output[:, 3:6, 3:6]
        expected_output = [ul, ur, ll, lr, c]
        expected_output = torch.stack(expected_output, -1)

        transform = x_transforms.Compose([
            x_transforms.ToPILImage(['input']),
            x_transforms.FiveCrop(['input'], 3),
            x_transforms.Lambda(['input'], lambda crops: [tvt.ToTensor()(crop) for crop in crops]),
            x_transforms.Lambda(['input'], lambda crops: torch.stack(crops, -1))
        ])
        output = transform(input_data)
        output = output['input']
        assert torch.allclose(output, expected_output)

    def test_grayscale(self):
        input_tensor_1 = torch.ones(1, 32, 32, dtype=torch.uint8).random_(0, 255)
        input_tensor_2 = torch.ones(3, 32, 32, dtype=torch.uint8).random_(0, 255)

        input_data = {'input_1': input_tensor_1, 'input_2': input_tensor_2}
        expected_output_1 = input_tensor_1.float().div_(255)
        expected_output_2 = tvt.Compose([
            tvt.ToPILImage(),
            tvt.Grayscale(3),
            tvt.ToTensor()
        ])(input_tensor_2)

        transform = x_transforms.Compose([
            x_transforms.ToPILImage(['input_1']),
            x_transforms.Grayscale(['input_1'], num_output_channels=1),
            x_transforms.ToTensor(['input_1']),
            x_transforms.ToPILImage(['input_2']),
            x_transforms.Grayscale(['input_2'], num_output_channels=3),
            x_transforms.ToTensor(['input_2']),
        ])
        output = transform(input_data)
        assert torch.allclose(output['input_1'], expected_output_1)
        assert torch.allclose(output['input_2'], expected_output_2)

    def test_pad(self):
        input_tensor = torch.ones(3, 10, 10, dtype=torch.uint8).random_(0, 255).float().div_(255)
        input_data = {'input': input_tensor}
        tvt_transform = tvt.Compose([tvt.ToPILImage(),
                                     tvt.Pad(1, fill=0, padding_mode='edge'),
                                     tvt.Pad(2, fill=1, padding_mode='constant'),
                                     tvt.Pad(3, fill=2, padding_mode='reflect'),
                                     tvt.Pad(4, fill=3, padding_mode='symmetric'),
                                     tvt.ToTensor()
                                 ])
        expected_output = tvt_transform(input_tensor)

        transform = tvt.Compose([x_transforms.ToPILImage(['input']),
                                 x_transforms.Pad(['input'], 1, fill=0, padding_mode='edge'),
                                 x_transforms.Pad(['input'], 2, fill=1, padding_mode='constant'),
                                 x_transforms.Pad(['input'], 3, fill=2, padding_mode='reflect'),
                                 x_transforms.Pad(['input'], 4, fill=3, padding_mode='symmetric'),
                                 x_transforms.ToTensor(['input'])
                                 ])
        output = transform(input_data)
        assert torch.allclose(output['input'], expected_output)

    def test_random_transforms_for_pil_image(self):
        input_tensor = torch.ones(3, 32, 32, dtype=torch.uint8).random_(0, 255).float().div_(255)
        input_data = {'input': input_tensor}
        random.seed(0)
        tvt_transform = tvt.Compose([tvt.ToPILImage(),
                                     tvt.RandomAffine(10.),
                                     tvt.RandomApply([tvt.RandomAffine(1), tvt.RandomAffine(1.)]),
                                     tvt.RandomChoice([tvt.RandomAffine(1), tvt.RandomAffine(1.)]),
                                     tvt.RandomOrder([tvt.RandomAffine(1), tvt.RandomAffine(1.)]),
                                     tvt.RandomCrop(3),
                                     tvt.RandomGrayscale(p=0.1),
                                     tvt.RandomHorizontalFlip(p=0.5),
                                     tvt.RandomPerspective(),
                                     tvt.RandomResizedCrop(16),
                                     tvt.RandomRotation(1.1),
                                     tvt.RandomVerticalFlip(),
                                     tvt.ToTensor()
                                     ])
        expected_output = tvt_transform(input_tensor)

        random.seed(0)
        transform = x_transforms.Compose([x_transforms.ToPILImage(['input']),
                                          x_transforms.RandomAffine(['input'], 10.),
                                          x_transforms.RandomApply([x_transforms.RandomAffine(['input'], 1),
                                                                    x_transforms.RandomAffine(['input'], 1.)]),
                                          x_transforms.RandomChoice([x_transforms.RandomAffine(['input'], 1),
                                                                     x_transforms.RandomAffine(['input'], 1.)]),
                                          x_transforms.RandomOrder([x_transforms.RandomAffine(['input'], 1),
                                                                    x_transforms.RandomAffine(['input'], 1.)]),
                                          x_transforms.RandomCrop(['input'], 3),
                                          x_transforms.RandomGrayscale(['input'], p=0.1),
                                          x_transforms.RandomHorizontalFlip(['input'], p=0.5),
                                          x_transforms.RandomPerspective(['input']),
                                          x_transforms.RandomResizedCrop(['input'], 16),
                                          x_transforms.RandomRotation(['input'], 1.1),
                                          x_transforms.RandomVerticalFlip(['input']),
                                          x_transforms.ToTensor(['input'])
                                          ])
        output = transform(input_data)
        assert torch.allclose(output['input'], expected_output)

    def test_resize(self):
        input_tensor = torch.ones(3, 32, 32, dtype=torch.uint8).random_(0, 255)
        input_data = {'input': input_tensor}
        expected_output = tvt.Compose([
            tvt.ToPILImage(),
            tvt.Resize(5),
            tvt.ToTensor()
        ])(input_tensor)

        transform = x_transforms.Compose([
            x_transforms.ToPILImage(['input']),
            x_transforms.Resize(['input'], 5),
            x_transforms.ToTensor(['input'])
        ])
        output = transform(input_data)
        assert torch.allclose(output['input'], expected_output)

    def test_ten_crop(self):
        input_tensor = torch.ones(3, 32, 32, dtype=torch.uint8).random_(0, 255)
        input_data = {'input': input_tensor}
        expected_output = tvt.Compose([
            tvt.ToPILImage(),
            tvt.TenCrop(3),
            tvt.Lambda(lambda crops: [tvt.ToTensor()(crop) for crop in crops]),
            tvt.Lambda(lambda crops: torch.stack(crops, -1))
        ])(input_tensor)

        transform = x_transforms.Compose([
            x_transforms.ToPILImage(['input']),
            x_transforms.TenCrop(['input'], 3),
            x_transforms.Lambda(['input'], lambda crops: [tvt.ToTensor()(crop) for crop in crops]),
            x_transforms.Lambda(['input'], lambda crops: torch.stack(crops, -1))
        ])
        output = transform(input_data)
        assert torch.allclose(output['input'], expected_output)


class TestTensorTransforms:
    def test_check_data_type(self):
        with pytest.raises(TypeError):
            input_tensor = torch.rand((1, 5, 5))
            input_tensor = tvt.ToPILImage()(input_tensor)
            input_data = {'input': input_tensor}
            transform = x_transforms.ToPILImage(['input'])
            transform(input_data)

    def test_linear_transformation(self):
        input_tensor = torch.ones(1, 9, 9, dtype=torch.uint8).random_(0, 255)
        input_data = {'input': input_tensor}
        trans_matrix = torch.randn(81).diag()
        mean_vector = torch.randn(81)
        expected_output = tvt.Compose([
            tvt.LinearTransformation(trans_matrix, mean_vector),
        ])(input_tensor)

        transform = x_transforms.Compose([
            x_transforms.LinearTransformation(['input'], trans_matrix, mean_vector),
        ])
        output = transform(input_data)
        assert torch.allclose(output['input'], expected_output)

    def test_normalize(self):
        input_tensor = torch.ones(3, 9, 9, dtype=torch.uint8).random_(0, 255).float().div_(255)
        input_data = {'input': input_tensor}
        mean_ = torch.mean(input_tensor, (1, 2))
        std_ = torch.std(input_tensor, (1, 2))
        expected_output = tvt.Compose([
            tvt.Normalize(mean_, std_),
        ])(input_tensor)

        transform = x_transforms.Compose([
            x_transforms.Normalize(['input'], mean_, std_)
        ])
        output = transform(input_data)
        assert torch.allclose(output['input'], expected_output)

    def test_random_erasing(self):
        input_tensor = torch.ones(3, 9, 9, dtype=torch.uint8).random_(0, 255).float().div_(255)
        input_data = {'input': input_tensor}
        random.seed(0)
        expected_output = tvt.Compose([
            tvt.RandomErasing(),
        ])(input_tensor)

        random.seed(0)
        transform = x_transforms.Compose([
            x_transforms.RandomErasing(['input'])
        ])
        output = transform(input_data)
        assert torch.allclose(output['input'], expected_output)
