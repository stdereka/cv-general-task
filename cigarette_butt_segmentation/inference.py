import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn


def inference(model: nn.Module, test_dataset,
              device: str, logit=False) -> np.array:
    """
    Runs prediction on unlabeled dataset
    :param model: pytorch model
    :param test_dataset: test pytorch dataset
    :param device: device to use (cpu, gpu etc.)
    :param logit: predict logit (True) or probabilities
    :return: array with predictions
    """
    test_loader = DataLoader(test_dataset)

    with torch.no_grad():
        logits = []

        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            model.eval()
            outputs = model(inputs).cpu()
            logits.append(outputs)

    if logit:
        return torch.cat(logits).numpy()
    else:
        return nn.Sigmoid()(torch.cat(logits)).numpy()


def inference_tta(model: nn.Module, test_dataset,
                  device: str, logit=False, augs=(), n_pass=4) -> np.array:
    """
    Runs n_pass predictions on unlabeled dataset with test time augmentations (TTA)
    :param model: pytorch model
    :param test_dataset: test pytorch dataset
    :param device: device to use (cpu, gpu etc.)
    :param logit: predict logit (True) or probabilities
    :param augs: augmentation pipeline
    :param n_pass: list of n_pass predictions on augmented dataset
    :return:
    """
    test_dataset.augs = augs

    preds = []
    for i in range(n_pass):
        pred = inference(model, test_dataset, device, logit)
        preds.append(pred)

    return preds
