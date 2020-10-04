import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn


def inference(model: nn.Module, test_loader: DataLoader,
              device: str, logit=False) -> np.array:
    """
    Runs prediction on test dataset
    :param model: pytorch model
    :param test_loader: test dataset pytorch loader
    :param device: device to use (cpu, gpu etc.)
    :param logit: predict logit (True) or probabilities
    :return: array with predictions
    """
    with torch.no_grad():
        logits = []

        for inputs in test_loader:
            inputs = inputs.to(device)
            model.eval()
            outputs = model(inputs).cpu()
            logits.append(outputs)

    if logit:
        return torch.cat(logits).numpy()
    else:
        return nn.Sigmoid()(logits).numpy()
