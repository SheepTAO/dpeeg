# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

import platform

import torch
from torch import Tensor
from torch.nn import Module


def get_device(nGPU: int = 0) -> torch.device:
    """Get the available devices.

    Parameters
    ----------
    nGPU : int
        GPU number to use.

    Returns
    -------
    device : torch.device
        Available device.
    """
    if not torch.cuda.is_available():
        dev = torch.device("cpu")
    else:
        if (nGPU > torch.cuda.device_count() - 1) or (nGPU < 0):
            raise ValueError(f"GPU: {nGPU} does not exit.")
        dev = torch.device(f"cuda:{nGPU}")
    return dev


def get_device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    elif device.type == "cpu":
        return platform.processor() or "Unknown CPU"
    else:
        return "Unsupported device type"


def model_predict(
    model: Module,
    data: Tensor,
    event_id: dict[str, int] | None = None,
    no_grad: bool = True,
) -> tuple[Tensor, list[str]] | tuple[Tensor, None]:
    """Compute model predictions.

    Parameters
    ----------
    model : Module
        Inherit Module and should define the forward method. The first
        parameter returned by model forward propagation is the prediction.
    data : Tensor (N, ...)
        Data to be predicted.
    event_id : dict, optional
        A dictionary containing label and corresponding id.
    no_grad : bool
        Whether to turn off autograd calculation graph recording.

    Returns
    -------
    preds : Tensor (N, ...)
        Prediction corresponding to the input data.
    labels : list of str
        If event_id not None, corresponding predicted label will be returned.
    """
    model.eval()
    if no_grad:
        with torch.no_grad():
            out = model(data)
    else:
        out = model(data)

    out = out[0] if isinstance(out, tuple) else out
    preds = torch.argmax(out, dim=1).detach()

    if event_id:
        reversed_event_id = {v: k for k, v in event_id.items()}
        labels = [reversed_event_id[pred] for pred in preds.tolist()]
        return preds, labels
    else:
        return preds, None


def model_depth(model: Module) -> int:
    """Compute the maximum depth of the model."""
    names = dict(model.named_modules()).keys()
    name_depth = [len(name.split(".")) for name in names]
    max_depth = max(name_depth)
    return max_depth
