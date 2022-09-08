import numpy as np
import torch
from skimage import io
from torch.nn.functional import interpolate


def im_reader(im_path):
    return io.imread(im_path)


def im_preprocess(im, size):
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    im_tensor = torch.tensor(im.copy(), dtype=torch.float32)
    im_tensor = torch.transpose(torch.transpose(im_tensor, 1, 2), 0, 1)
    if len(size) < 2:
        return im_tensor, im.shape[0:2]
    else:
        im_tensor = torch.unsqueeze(im_tensor, 0)
        im_tensor = interpolate(im_tensor, size, mode="bilinear")
        im_tensor = torch.squeeze(im_tensor, 0)

    return im_tensor.type(torch.uint8), im.shape[0:2]
