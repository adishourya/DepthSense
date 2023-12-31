import torch
import torch.nn as nn


# Based on
# https://github.com/tensorflow/models/blob/master/research/struct2depth/model.py#L625-L641


class InverseDepthSmoothnessLoss(nn.Module):
    r"""Criterion that computes image-aware inverse depth smoothness loss.
    """

    def __init__(self) -> None:
        super(InverseDepthSmoothnessLoss, self).__init__()

    @staticmethod
    def gradient_x(img: torch.Tensor) -> torch.Tensor:
        assert len(img.shape) == 4, img.shape
        return img[:, :, :, :-1] - img[:, :, :, 1:]

    @staticmethod
    def gradient_y(img: torch.Tensor) -> torch.Tensor:
        assert len(img.shape) == 4, img.shape
        return img[:, :, :-1, :] - img[:, :, 1:, :]

    def forward(
            self,
            idepth: torch.Tensor,
            image: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(idepth):
            raise TypeError("Input idepth type is not a torch.Tensor. Got {}"
                            .format(type(idepth)))
        if not torch.is_tensor(image):
            raise TypeError("Input image type is not a torch.Tensor. Got {}"
                            .format(type(image)))
        if not len(idepth.shape) == 4:
            raise ValueError("Invalid idepth shape, we expect BxCxHxW. Got: {}"
                             .format(idepth.shape))
        if not len(image.shape) == 4:
            raise ValueError("Invalid image shape, we expect BxCxHxW. Got: {}"
                             .format(image.shape))
        if not idepth.shape[-2:] == image.shape[-2:]:
            raise ValueError("idepth and image shapes must be the same. Got: {}"
                             .format(idepth.shape, image.shape))
        if not idepth.device == image.device:
            raise ValueError(
                "idepth and image must be in the same device. Got: {}" .format(
                    idepth.device, image.device))
        if not idepth.dtype == image.dtype:
            raise ValueError(
                "idepth and image must be in the same dtype. Got: {}" .format(
                    idepth.dtype, image.dtype))
        # compute the gradients
        idepth_dx: torch.Tensor = self.gradient_x(idepth)
        idepth_dy: torch.Tensor = self.gradient_y(idepth)
        image_dx: torch.Tensor = self.gradient_x(image)
        image_dy: torch.Tensor = self.gradient_y(image)

        # compute image weights
        weights_x: torch.Tensor = torch.exp(
            -torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
        weights_y: torch.Tensor = torch.exp(
            -torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

        # apply image weights to depth
        smoothness_x: torch.Tensor = torch.abs(idepth_dx * weights_x)
        smoothness_y: torch.Tensor = torch.abs(idepth_dy * weights_y)
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)
