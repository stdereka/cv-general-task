import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from lib import get_dice
import albumentations as a


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
    :return: list of arrays with predictions
    """
    test_dataset.augs = augs

    preds = []
    for i in range(n_pass):
        pred = inference(model, test_dataset, device, logit)
        preds.append(pred)

    test_dataset.augs = a.Compose([])

    return preds


def inference_with_metric(model: nn.Module, test_dataset, device: str, thresh=0.5):
    """
    Runs prediction and computes metric for every picture on labeled dataset
    :param model: pytorch model
    :param test_dataset: test pytorch dataset (labeled)
    :param device: device to use (cpu, gpu etc.)
    :param thresh: class 1 threshold
    :return: (predicted masks, metric values)
    """
    preds = inference(model, test_dataset, device, logit=False)
    trues = [item[1].numpy() for item in test_dataset]
    dices = [get_dice(trues[i], preds[i] >= thresh) for i in range(len(trues))]

    return preds, dices


def inference_with_rotations(model: nn.Module, test_dataset,
                             device: str, logit=False):
    """
    Returns 4 predictions on rotated images (0, 90, 180, 270 degrees)
    :param model: pytorch model
    :param test_dataset: test pytorch dataset
    :param device: device to use (cpu, gpu etc.)
    :param logit: predict logit (True) or probabilities
    :return: 4 predictions
    """

    preds = []
    for k in range(4):
        test_loader = DataLoader(test_dataset)

        with torch.no_grad():
            logits = []

            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                inputs = torch.rot90(inputs, k=k, dims=(2, 3))
                model.eval()
                outputs = model(inputs).cpu()
                logits.append(outputs)

        if logit:
            pred = torch.cat(logits).numpy()
        else:
            pred = nn.Sigmoid()(torch.cat(logits)).numpy()

        preds.append(np.rot90(pred, k=4-k, axes=(2, 3)))

    return preds
