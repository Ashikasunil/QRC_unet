import torch.nn as nn
import torch

# Dummy placeholder class (replace with your real QRC_UNet if you're using state_dict)
class QRC_UNet(nn.Module):
    def __init__(self):
        super(QRC_UNet, self).__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)