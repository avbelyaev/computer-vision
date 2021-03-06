# -*- coding: utf-8 -*-
"""Alexarch.ipynb

Automatically generated by Colaboratory.
"""

import torch
import numpy as np

class AlexFromScratch:
    """Alexnet architecture implemented from scratch"""

    def __init__(self):
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=224, kernel_size=11, stride=4, padding=2),
            torch.nn.Conv2d(in_channels=224, out_channels=48, kernel_size=5, padding=2),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, padding=2),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(in_channels=128, out_channels=192, kernel_size=5, padding=2),
            torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            torch.nn.Conv2d(in_channels=192, out_channels=128, kernel_size=5, padding=2),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Linear(in_features=6, out_features=2048),
            torch.nn.Linear(in_features=2048, out_features=2048),
            torch.nn.Linear(in_features=2048, out_features=3)
        )   

    def forward(self, x):
        return self.net(x)

alex = AlexFromScratch()

alex_in = torch.randn(3, 3, 224, 224)
alex_in.shape
# --> torch.Size([3, 3, 224, 224])


alex_out = alex.forward(alex_in)


alex_out.shape
# --> torch.Size([3, 128, 6, 3])
