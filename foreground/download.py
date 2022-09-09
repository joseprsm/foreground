import os

import gdown

MODEL_PATH_URL = "https://drive.google.com/uc?id=1nV57qKuy--d5u1yvkng9aXW1KS4sOpOi"


def download_weights():
    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")
        gdown.download(MODEL_PATH_URL, "saved_models/isnet.pth", use_cookies=False)


if __name__ == "__main__":
    download_weights()
