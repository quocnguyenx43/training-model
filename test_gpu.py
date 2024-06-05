import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import datasets as dst
import models as md
import functions as func

import argparse as arg

import warnings
warnings.filterwarnings('ignore')


try:
    # Attempt to use GPU for training if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except RuntimeError as e:
    # Handle potential errors (e.g., no compatible GPU found)
    print(f"Error: {e}")
    print("Using CPU for training.")
    device = 'cpu'

print(f"Using device: {device}")