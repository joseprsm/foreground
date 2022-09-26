import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torch.nn.functional import interpolate

from foreground.load import load_hp, load_image

dvc = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(hypar, device):
    net = hypar["model"]  # GOSNETINC(3,1)

    # convert to half precision
    if hypar["model_digit"] == "half":
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    net.to(device)

    if hypar["restore_model"] != "":
        net.load_state_dict(
            torch.load(
                hypar["model_path"] + "/" + hypar["restore_model"], map_location=device
            )
        )
        net.to(device)
    net.eval()
    return net


def predict(net, inputs_val, shapes_val, hypar, device):
    """
    Given an Image, predict the mask
    """
    net.eval()

    if hypar["model_digit"] == "full":
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)

    inputs_val_v = Variable(inputs_val, requires_grad=False).to(
        device
    )  # wrap inputs in Variable

    ds_val = net(inputs_val_v)[0]  # list of 6 results

    pred_val = ds_val[0][
        0, :, :, :
    ]  # B x 1 x H x W    # we want the first one which is the most accurate prediction

    # recover the prediction spatial size to the orignal image size
    pred_val = torch.squeeze(
        interpolate(
            torch.unsqueeze(pred_val, 0),
            (shapes_val[0][0], shapes_val[0][1]),
            mode="bilinear",
        )
    )

    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val - mi) / (ma - mi)  # max = 1

    if device == "cuda":
        torch.cuda.empty_cache()
    return (pred_val.detach().cpu().numpy() * 255).astype(
        np.uint8
    )  # it is the mask we need


# Build Model
hp = load_hp()
model = build_model(hp, dvc)


def inference(image: Image):
    image_path = image

    image_tensor, orig_size = load_image(image_path, hp)
    mask = predict(model, image_tensor, orig_size, hp, dvc)

    pil_mask = Image.fromarray(mask).convert("L")
    im_rgb = Image.open(image).convert("RGB")
    im_rgb.putalpha(pil_mask)

    return im_rgb
