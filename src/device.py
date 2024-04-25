import torch


def get_device() -> str:
    """
    Determine and return the device on which PyTorch will run.

    Returns:
    ----------
        str: Either "cuda" if CUDA is available, otherwise "cpu".
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device.type
