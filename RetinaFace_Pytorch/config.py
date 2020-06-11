import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
image_w = 112
image_h = 112
channel = 3
emb_size = 512

# Training parameters
num_workers = 4  # for data-loading; right now, only 1 works with h5py
grad_clip = 1.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
rgb_mean = (104, 117, 123)  # bgr order
num_classes = 2

DATA_DIR = 'data'
train_label_file = './data/widerface/train/label.txt'
valid_label_file = './data/widerface/val/label.txt'
