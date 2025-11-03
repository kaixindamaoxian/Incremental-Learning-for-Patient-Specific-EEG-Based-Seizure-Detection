"""
    EEGNet PyTorch implementation
    Original implementation - https://github.com/vlawhern/arl-eegmodels
    Original paper: https://iopscience.iop.org/article/10.1088/1741-2552/aace8c

    ---
    EEGNet Parameters:

      nb_classes      : int, number of classes to classify
      Chans           : number of channels in the EEG data
      Samples         : sample frequency (Hz) in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer.
                        ARL recommends to set this parameter to be half of the sampling rate.
                        For the SMR dataset in particular since the data was high-passed at 4Hz ARL used a kernel length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim


class EEGNet(nn.Module):
    def __init__(self, n_channels, sample_rate):
        super(EEGNet, self).__init__()
        kernel = int(sample_rate * 0.25) - 1
        # Layer 1
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 16, (1, kernel), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1)
        )
        # Layer 2
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(16, 32, (n_channels, 1), stride=(1, 1), groups=16),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.5)
        )
        # Layer 3
        self.separable_conv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 16), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.5)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=32, out_features=2, bias=True)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (64, 18, 256) ->  (64, 1, 18, 256)
        x = self.temporal_conv(x)   # (64, 1, 18, 256) ->  (64, 16, 18, 256)
        x = self.depthwise_conv(x)  # (64, 16, 18, 256) ->  (64, 32, 1, 256 /4)
        x = self.separable_conv(x)  # (64, 32, 1, 256 /4) -> (64, 32, 1, 256 /4 / 8)
        x = torch.mean(x, 3)    # (64, 32, 1)
        x = x.permute(0, 2, 1).contiguous().squeeze(1)  # (64, 1, 32)
        out = self.fc(x)
        return x, out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = torch.randn(64, 19, 256*4).to(device)

    model = EEGNet(19, 256).to(device)
    features, outputs = model(input_data)