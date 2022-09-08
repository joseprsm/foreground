import numpy as np
import torch
from skimage import io
from torch.nn.functional import interpolate
from torchvision.transforms import Compose
from torchvision.transforms.functional import normalize

from foreground.models import ISNetDIS


class GOSNormalize(object):
    """
    Normalize the Image using torch.transforms.
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

    def __call__(self, image):
        image = normalize(image, self.mean, self.std)
        return image


transform = Compose([GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])])


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


def load_image(im_path, hypar):
    im = io.imread(im_path)
    im, im_shp = im_preprocess(im, hypar["cache_size"])
    im = torch.divide(im, 255.0)
    shape = torch.from_numpy(np.array(im_shp))
    transformed = transform(im).unsqueeze(0)
    shape = shape.unsqueeze(0)
    return transformed, shape


def load_hp():
    # Set Parameters
    hp = {}  # paramters for inferencing

    hp["model_path"] = "./saved_models"  # load trained weights from this path
    hp["restore_model"] = "isnet.pth"  # name of the to-be-loaded weights
    hp["interm_sup"] = False  # indicate if activate intermediate feature supervision

    #  choose floating point accuracy
    hp["model_digit"] = "full"  # indicates "half" or "full" accuracy of float number
    hp["seed"] = 0

    # cached input spatial resolution, can be configured into different size
    hp["cache_size"] = [1024, 1024]

    # data augmentation parameters ---
    # model input spatial size, usually use the same value hypar["cache_size"], which means we don't further resize the images
    hp["input_size"] = [1024, 1024]

    # random crop size from the input, it is usually set as smaller than hypar["cache_size"], e.g., [920,920] for data augmentation
    hp["crop_size"] = [1024, 1024]

    hp["model"] = ISNetDIS()
    return hp
