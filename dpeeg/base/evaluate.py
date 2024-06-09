#!/usr/bin/env python
# coding: utf-8

"""
    This module is used to evaluate the metrics of the model and save.

    @author: SheepTAO
    @data: 2023-4-13
"""


import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch import Tensor
from torch.nn import Module
from torchinfo import summary
from torch.utils.hooks import RemovableHandle
from torchmetrics.functional.classification.confusion_matrix import confusion_matrix


def save_cm_img(
    preds : Tensor,
    target : Tensor,
    cls_name : list | tuple,
    fig_path : str,
) -> None:
    '''Calculate and save the corresponding confusion matrix figure to the given
    img path.

    Parameters
    ----------
    preds : Tensor
        Predicted labels, as returned by a classifier.
    target : Tensor
        Ground truth (correct) labels.
    cls_name : list, tuple
        The name of dataset labels.
    fig_path : str
        Path to store the figure.
    '''
    cm = confusion_matrix(preds, target, 'multiclass', num_classes=len(cls_name))
    ax = sns.heatmap(cm / cm.sum(1, keepdim=True), annot=True, cmap='Blues',
                      xticklabels=cls_name, yticklabels=cls_name) # type: ignore
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.savefig(fig_path)
    # # !!! Clear the current Axes
    # plt.clf()
    # Clear all
    plt.close()


def model_predict(
    model : Module, 
    data : Tensor,
    event_id : dict[str, int] | None = None,
    no_grad : bool = True,
) -> tuple[Tensor, list[str]] | tuple[Tensor, None]:
    '''Compute model predictions.

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
    '''
    model.eval()
    if no_grad:
        with torch.no_grad():
            out = model(data)
    else:
        out = model(data)

    out = out[0] if isinstance(out, tuple) else out
    preds = torch.argmax(out, dim=1).detach()

    if event_id:
        reversed_event_id = {v : k for k, v in event_id.items()}
        labels = [reversed_event_id[pred] for pred in preds.tolist()]
        return preds, labels
    else:
        return preds, None


def model_depth(model : Module) -> int:
    '''Compute the maximum depth of the model.
    '''
    names = dict(model.named_modules()).keys()
    name_depth = [len(name.split('.')) for name in names]
    max_depth = max(name_depth)
    return max_depth


def model_summary(model : Module):
    '''Output the model structure according to the model depth.

    This function is usually used in conjunction with the `Activation` class. 
    You can use this function to view the specific inter-layer relationships of
    the model, and then obtain the corresponding middle layer name by adding '.'
    to the inter-layer name as the names parameter of the Activation class.

    Examples
    --------
    >>> class MyModel(nn.Module):
    ...     def __init__(self) -> None:
    ...         super().__init__()
    ...         self.conv = nn.Sequential(
    ...             nn.Conv2d(3, 10, 3),
    ...             nn.Conv2d(10, 20, 4)
    ...         )
    ... 
    ...     def forward(self, x):
    ...         return self.conv(x)
    ... 
    >>> model = MyModel()
    >>> print(model_summary(model))
    MyModel (MyModel)                        --
    ├─Sequential (conv)                      --
    │    └─Conv2d (0)                        280
    │    └─Conv2d (1)                        3,220

    Then you can use `conv.0` and `conv.1` as the name of intermediate layer.
    '''
    s = summary(
        model=model,
        depth=model_depth(model),
        row_settings=['var_names'],
        verbose=0
    )
    return s


class Activation:
    def __init__(self, model : Module, names : list[str]) -> None:
        '''Get model's intermediate result.

        Allows the context manager to be used to obtain the forward propagation
        output feature map of any intermediate layer of the model.

        Parameters
        ----------
        model : Module
            Obtain the intermediate layer output of the model.
        names : list of str
            A list of intermediate module names whose outputs will be obtained.
            You can get all module names through `model.named_modules()` or 
            `dpeeg.base.evaluate.model_summary`.

        Examples
        --------
        >>> with Activation(net, ['conv.0', 'conv.1']) as act:
        >>>     model_predict(net, data)
        >>>     actmaps = act.get_actmaps()

        you can also manually remove the handler registered on the model

        >>> act = Activation(net, ['conv.0', 'conv.1'])
        >>> model_predict(net, data)
        >>> actmaps = act.get_actmaps()
        >>> act.close()
        '''        
        self.names = names
        self.handle : list[RemovableHandle] = list()
        self.activation_maps  = dict()

        for name in self.names:
            module = model.get_submodule(name)
            handle = module.register_forward_hook(self._hook_func(name))
            self.handle.append(handle)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _hook_func(self, name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                actmaps = list()
                for out in output:
                    if isinstance(out, Tensor):
                        actmaps.append(out.detach())
                    else:
                        actmaps.append(out)
                actmap = tuple(actmaps)
            elif isinstance(output, Tensor):
                actmap = output.detach()
            else:
                raise TypeError(f'Unsupport type {type(output)} as output, in '
                                f'{model._get_name()}.')
            self.activation_maps[name] = actmap
        return hook

    def get_actmaps(self) -> dict[str, Tensor]:
        '''Get intermediate activation maps.

        Returns
        -------
        dict
            Returns the name of the intermediate layer and its corresponding
            activation feature maps.
        '''
        return self.activation_maps

    def close(self):
        '''Remove the handler and clear the activation maps.
        '''
        for handle in self.handle:
            handle.remove()
        self.handle.clear()
