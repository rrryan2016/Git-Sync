import torch 
import torchvision 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torchvision import datasets, transforms 
import torch.utils.data as tud 
import numpy as np 
import time

(pretrai=False)

trainset = datasets.CIFAR10(root='./data',train=True,transform= , download=False)




