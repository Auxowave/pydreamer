from decimal import DivisionImpossible
from typing import Optional, Union
from numpy import size
import torch
import torch.nn as nn
import torch.distributions as D
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from actnorm import ActNorm2d
import pickle

from .functions import *
from .common import *

class NewCNN(nn.Module):

    def __init__(self, in_channels=3, cnn_depth=32, activation=nn.ELU):
        super().__init__()
        self.out_dim = cnn_depth * 32
        d = cnn_depth

        self.model = nn.Sequential(
            nn.Conv2d(9, d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d),
            activation(),
            nn.Conv2d(d, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            activation()
        )

        self.gif = nn.Parameter(self.load_gif())

        self.x_0 = None
        self.x_1 = None
        self.x_2 = None
        self.x_3 = None
        self.iter = 0
        self.picture_every = 5040

    @staticmethod
    def load_gif():
        with open("images.txt", "rb") as f:
            images = pickle.load(f)
        tensors = []
        for image in images:
            tensor = torch.tensor(image)
            tensors.append(tensor)
        b = torch.Tensor(60, 3, 64, 64)
        torch.cat(tensors, out=b)
        return b

    def forward(self, x: Tensor) -> Tensor:
        if self.x_0 is not None and self.x_0.size() != x.size():
            print("reset history")
            self.x_0 = None

        gif_image = self.gif[self.iter % 60]

        x = x + gif_image

        if self.x_0 is None:
            self.x_0 = x
            self.x_1 = x
            self.x_2 = x
            self.x_3 = x
        else:
            self.x_3 = self.x_2
            self.x_2 = self.x_1
            self.x_1 = self.x_0
            self.x_0 = x

        combined_history = torch.cat((self.x_1, self.x_2, self.x_3), -3)
        combined_history, bd = flatten_batch(combined_history, 3)
        y = self.model(combined_history)
        y = unflatten_batch(y, bd)



        self.iter += 1
        if self.iter == self.picture_every:
            print("Creating pictures New CNN")
            fig, (ax1, ax2, ax3) = plt.subplots(1,3)
            ax1.imshow(np.clip(x.cpu().detach().numpy().astype('float64')[0][0].transpose((1,2,0)), 0, 1), interpolation='nearest')
            ax1.set_title("Input")
            ax2.imshow(np.clip(y.cpu().detach().numpy().astype('float64')[0][0].transpose((1,2,0)), 0, 1), interpolation='nearest')
            ax2.set_title("CNN_out 1")
            ax3.imshow(np.clip(np.mean(np.mean(y.cpu().detach().numpy().astype('float64'), 0), 0).transpose((1,2,0)), 0, 1), interpolation='nearest')
            ax3.set_title("CNN_out mean")
            plt.savefig('pictures/NewCNN_out.png')
            plt.close(fig)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(np.clip(self.x_1.cpu().detach().numpy().astype('float64')[0][0].transpose((1,2,0)), 0, 1), interpolation='nearest')
            ax1.set_title("x_1")
            ax2.imshow(np.clip(self.x_2.cpu().detach().numpy().astype('float64')[0][0].transpose((1,2,0)), 0, 1), interpolation='nearest')
            ax2.set_title("x_2")
            ax3.imshow(np.clip(self.x_3.cpu().detach().numpy().astype('float64')[0][0].transpose((1,2,0)), 0, 1), interpolation='nearest')
            ax3.set_title("x_3")
            plt.savefig('pictures/history.png')
            plt.close(fig)
            self.iter = 0
        

        return y

