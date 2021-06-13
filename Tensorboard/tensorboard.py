#Imports
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import sys 
import os
import numpy as np
import math

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter





writer = SummaryWriter(f'runs/Vulcan')




step = 0



writer.add_scalar('Training Loss', loss, global_step=step)
writer.__ne__