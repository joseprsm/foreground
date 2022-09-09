from torch.nn.functional import interpolate


def upsample_like(src, tar):
    """Up-sample tensor 'src' to have the same spatial size with tensor 'tar'"""
    return interpolate(src, size=tar.shape[2:], mode="bilinear")
