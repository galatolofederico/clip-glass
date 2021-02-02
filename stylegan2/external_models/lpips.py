"""
Code adapted from https://github.com/richzhang/PerceptualSimilarity

Original License:
Copyright (c) 2018, Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import torch
from torch import nn
import torchvision


class LPIPS_VGG16(nn.Module):
    _FEATURE_IDX = [0, 4, 9, 16, 23, 30]
    _LINEAR_WEIGHTS_URL = 'https://github.com/richzhang/PerceptualSimilarity' + \
                          '/blob/master/lpips/weights/v0.1/vgg.pth?raw=true'

    def __init__(self, pixel_min=-1, pixel_max=1):
        super(LPIPS_VGG16, self).__init__()
        features = torchvision.models.vgg16(pretrained=True).features
        self.slices = nn.ModuleList()
        linear_weights = torch.utils.model_zoo.load_url(self._LINEAR_WEIGHTS_URL)
        for i in range(1, len(self._FEATURE_IDX)):
            idx_range = range(self._FEATURE_IDX[i - 1], self._FEATURE_IDX[i])
            self.slices.append(nn.Sequential(*[features[j] for j in idx_range]))
        self.linear_layers = nn.ModuleList()
        for weight in torch.utils.model_zoo.load_url(self._LINEAR_WEIGHTS_URL).values():
            weight = weight.view(1, -1)
            linear = nn.Linear(weight.size(1), 1, bias=False)
            linear.weight.data.copy_(weight)
            self.linear_layers.append(linear)
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188]).view(1, -1, 1, 1))
        self.register_buffer('scale', torch.Tensor([.458,.448,.450]).view(1, -1, 1, 1))
        self.pixel_min = pixel_min
        self.pixel_max = pixel_max
        self.requires_grad_(False)
        self.eval()

    def _scale(self, x):
        if self.pixel_min != -1 or self.pixel_max != 1:
            x = (2*x - self.pixel_min - self.pixel_max) \
                / (self.pixel_max - self.pixel_min)
        return (x - self.shift) / self.scale

    @staticmethod
    def _normalize_tensor(feature_maps, eps=1e-8):
        rnorm = torch.rsqrt(torch.sum(feature_maps ** 2, dim=1, keepdim=True) + eps)
        return feature_maps * rnorm

    def forward(self, x0, x1, eps=1e-8):
        x0, x1 = self._scale(x0), self._scale(x1)
        dist = 0
        for slice, linear in zip(self.slices, self.linear_layers):
            x0, x1 = slice(x0), slice(x1)
            _x0, _x1 = self._normalize_tensor(x0, eps), self._normalize_tensor(x1, eps)
            dist += linear(torch.mean((_x0 - _x1) ** 2, dim=[-1, -2]))
        return dist.view(-1)
