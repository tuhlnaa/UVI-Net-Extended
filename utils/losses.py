import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from math import exp
from torch import Tensor
from torch.autograd import Variable
from typing import Optional, Union, List, Tuple

def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1 - _ssim_3D(
            img1, img2, window, self.window_size, channel, self.size_average
        )


def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = (
        _1D_window.mm(_2D_window.reshape(1, -1))
        .reshape(window_size, window_size, window_size)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )
    window = Variable(
        _3D_window.expand(
            channel, 1, window_size, window_size, window_size
        ).contiguous()
    )
    return window


def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim3D(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_3D(img1, img2, window, window_size, channel, size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class MSE(torch.nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, img1, img2):
        return torch.mean((img1 - img2) ** 2)


class L1(torch.nn.Module):
    def __init__(self):
        super(L1, self).__init__()

    def forward(self, img1, img2):
        return torch.mean(torch.abs(img1 - img2))


class L1_norm(torch.nn.Module):
    def __init__(self):
        super(L1_norm, self).__init__()

    def forward(self, img1):
        return torch.mean(torch.abs(img1))
    

class Grad(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty="l1", loss_mult=None):
        super(Grad, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == "l2":
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class Grad3d(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty="l1", loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == "l2":
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class DiceLoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self, num_class=36):
        super().__init__()
        self.num_class = num_class

    def forward(self, y_pred, y_true):
        # y_pred = torch.round(y_pred)
        # y_pred = nn.functional.one_hot(torch.round(y_pred).long(), num_classes=7)
        # y_pred = torch.squeeze(y_pred, 1)
        # y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
        y_true = nn.functional.one_hot(y_true, num_classes=self.num_class)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
        intersection = y_pred * y_true
        intersection = intersection.sum(dim=[2, 3, 4])
        union = torch.pow(y_pred, 2).sum(dim=[2, 3, 4]) + torch.pow(y_true, 2).sum(
            dim=[2, 3, 4]
        )
        dsc = (2.0 * intersection) / (union + 1e-5)
        dsc = 1 - torch.mean(dsc)
        return dsc


# class NCC(torch.nn.Module):
#     """
#     Local (over window) normalized cross correlation loss.
#     """

#     def __init__(self, win=9, gpu=True):
#         super(NCC, self).__init__()
#         self.win = win
#         if gpu:
#             self.device = "cuda:0"
#         else:
#             self.device = "cpu"

#     def forward(self, y_true, y_pred):
#         Ii = y_true
#         Ji = y_pred

#         # get dimension of volume
#         # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
#         ndims = len(list(Ii.size())) - 2
#         assert ndims in [1, 2, 3], (
#             "volumes should be 1 to 3 dimensions. found: %d" % ndims
#         )

#         # set window size
#         win = [self.win] * ndims

#         # compute filters
#         sum_filt = torch.ones([1, 1, *win]).to(self.device)

#         pad_no = math.floor(win[0] / 2)

#         if ndims == 1:
#             stride = 1
#             padding = pad_no
#         elif ndims == 2:
#             stride = (1, 1)
#             padding = (pad_no, pad_no)
#         else:
#             stride = (1, 1, 1)
#             padding = (pad_no, pad_no, pad_no)

#         # get convolution function
#         conv_fn = getattr(F, "conv%dd" % ndims)

#         # compute CC squares
#         I2 = Ii * Ii
#         J2 = Ji * Ji
#         IJ = Ii * Ji

#         I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
#         J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
#         I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
#         J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
#         IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

#         win_size = np.prod(win)
#         u_I = I_sum / win_size
#         u_J = J_sum / win_size

#         cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
#         I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
#         J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

#         cc = cross * cross / (I_var * J_var + 1e-5)

#         return -torch.mean(cc)


class NCCLoss(nn.Module):
    """Normalized Cross Correlation (NCC) Loss for image similarity measurement.
    
    This loss function computes the local normalized cross-correlation over a window,
    which is particularly useful for medical image registration tasks.
    
    Args:
        window_size (Union[int, List[int]], optional): Size of the local window for computing correlation.
            Can be a single integer for isotropic window or list of integers for each dimension.
            Defaults to 9.
        device (Optional[Union[str, torch.device]], optional): Device to run computations on.
            Defaults to None (will use input tensor's device).
    """

    def __init__(self, window_size: Union[int, List[int]] = 9, device: Optional[Union[str, torch.device]] = None) -> None:
        super().__init__()
        self.window_size = window_size
        self.device = device

    def _setup_conv_params(self, ndims: int, window_size: Union[int, List[int]]) -> Tuple[List[int], Tuple, Tuple]:
        """Set up convolution parameters based on input dimensionality.
        
        Args:
            ndims (int): Number of spatial dimensions (1D, 2D, or 3D)
            window_size (Union[int, List[int]]): Window size for local correlation
            
        Returns:
            Tuple[List[int], Tuple, Tuple]: Window sizes, stride, and padding
        """
        if isinstance(window_size, int):
            win_sizes = [window_size] * ndims
        else:
            win_sizes = list(window_size)
            assert len(win_sizes) == ndims, f"Expected {ndims} window sizes, got {len(win_sizes)}"
        
        pad_size = win_sizes[0] // 2
        if ndims == 1:
            stride = (1,)
            padding = (pad_size,)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_size, pad_size)
        else:  # ndims == 3
            stride = (1, 1, 1)
            padding = (pad_size, pad_size, pad_size)
            
        return win_sizes, stride, padding

    def forward(self, y_pred: Tensor, y_true: Tensor, eps: float = 1e-5) -> Tensor:
        """Compute NCC loss between predicted and target images.
        
        Args:
            y_pred (Tensor): Predicted image tensor of shape [batch_size, *vol_shape, nb_feats]
            y_true (Tensor): Target image tensor of shape [batch_size, *vol_shape, nb_feats]
            eps (float, optional): Small constant for numerical stability. Defaults to 1e-5.
            
        Returns:
            Tensor: Computed NCC loss (negative mean correlation)
        """
        # Validate input dimensions
        ndims = len(y_true.shape) - 2
        if ndims not in {1, 2, 3}:
            raise ValueError(f"Expected 1-3 dimensions, got {ndims}")
            
        # Get device from input if not specified
        device = self.device or y_true.device
        
        # Setup convolution parameters
        win_sizes, stride, padding = self._setup_conv_params(ndims, self.window_size)
        
        # Create convolution filter
        sum_filt = torch.ones(1, 1, *win_sizes, device=device)
        
        # Get appropriate convolution function
        conv_fn = getattr(F, f'conv{ndims}d')
        
        # Compute local sums via convolution
        win_size = torch.prod(torch.tensor(win_sizes))
        I, J = y_true, y_pred
        
        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I * I, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J * J, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(I * J, sum_filt, stride=stride, padding=padding)
        
        # Compute local means
        I_mean = I_sum / win_size
        J_mean = J_sum / win_size
        
        # Compute cross correlation
        cross = IJ_sum - J_mean * I_sum - I_mean * J_sum + I_mean * J_mean * win_size
        I_var = I2_sum - 2 * I_mean * I_sum + I_mean * I_mean * win_size
        J_var = J2_sum - 2 * J_mean * J_sum + J_mean * J_mean * win_size
        
        cc = cross * cross / (I_var * J_var + eps)
        
        return -torch.mean(cc)
    

def CharbonnierLoss(predict, target, eps=1e-3):
    return torch.mean(torch.sqrt((predict - target) ** 2 + eps**2))


class CensusLoss(nn.Module):
    def __init__(self, patch_size=7):
        super(CensusLoss, self).__init__()
        patch_size = patch_size
        out_channels = patch_size * patch_size * patch_size
        self.w = np.eye(out_channels).reshape(
            (patch_size, patch_size, patch_size, 1, out_channels)
        )
        self.w = np.transpose(self.w, (4, 3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().to("cuda:0")

    def transform(self, img):
        patches = F.conv3d(img, self.w, padding=3, bias=None)
        transf = patches - img
        transf_norm = transf / torch.sqrt(0.81 + transf**2)
        return transf_norm

    def hamming(self, t1, t2):
        dist = (t1 - t2) ** 2
        dist_norm = torch.mean(dist / (0.1 + dist), 1, True)
        return dist_norm

    def valid_mask(self, t, padding):
        n, _, d, h, w = t.size()
        inner = torch.ones(
            n, 1, d - 2 * padding, h - 2 * padding, w - 2 * padding
        ).type_as(t)
        mask = F.pad(inner, [padding] * 6)
        return mask

    def forward(self, img0, img1):
        img0 = self.transform(img0)
        img1 = self.transform(img1)
        return (self.hamming(img0, img1) * self.valid_mask(img0, 1)).mean()
