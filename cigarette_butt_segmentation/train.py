import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from lib import get_dice
import matplotlib.pyplot as plt


def fit_epoch(model: nn.Module, train_loader: DataLoader,
              criterion, optimizer, device: str) -> (float, float):
    """
    This function performs one epoch training
    :param model: pytorch model
    :param train_loader: pytorch data loader
    :param criterion: loss function to optimize
    :param optimizer: optimizer to use
    :param device: device to train on
    :return: loss and metric on train set
    """
    running_loss = 0.0
    processed_data = 0

    ground = []
    predicted = []
    for inputs, labels in train_loader:
        inputs = inputs.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(labels, outputs)
        loss.backward()
        optimizer.step()
        preds = nn.Sigmoid()(outputs)

        ground.append(labels.cpu())
        predicted.append(preds.cpu().detach().numpy())
        running_loss += loss.item() * inputs.size(0)
        processed_data += inputs.size(0)

    ground = np.hstack(ground)
    predicted = np.hstack(predicted)

    train_loss = running_loss / processed_data
    train_metric = get_dice(ground, predicted >= 0.5)

    return train_loss, train_metric


def eval_epoch(model: nn.Module, val_loader: DataLoader,
               criterion, device: str) -> (float, float):
    """
    This function performs one epoch evaluation
    :param model: pytorch model
    :param val_loader: pytorch data loader
    :param criterion: loss function
    :param device: device to train on
    :return: loss and metric on validation set
    """
    model.eval()
    running_loss = 0.0
    processed_size = 0

    ground = []
    predicted = []
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(labels, outputs)
            preds = nn.Sigmoid()(outputs)

        ground.append(labels.cpu())
        predicted.append(preds.cpu().detach().numpy())

        running_loss += loss.item() * inputs.size(0)
        processed_size += inputs.size(0)

    ground = np.hstack(ground)
    predicted = np.hstack(predicted)

    val_loss = running_loss / processed_size
    val_metric = get_dice(ground, predicted >= 0.5)

    return val_loss, val_metric


def train(train_dataset, val_dataset, model, epochs: int,
          batch_size: int, device: str, opt, criterion) -> np.array:
    """
    Runs training loop
    :param train_dataset: training set
    :param val_dataset: validation set
    :param model: pytorch model
    :param epochs: number of epochs
    :param batch_size: size of batch
    :param device: device to train on
    :param opt: optimizer to use
    :param criterion: loss function
    :return: numpy array with loss and accuracy values (each row corresponds to one epoch)
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    history = []
    log_template = "Epoch: {ep}, train_loss: {t_loss:0.4f}, val_loss: {v_loss:0.4f}, " \
                   "train_metric: {t_acc:0.4f}, val_metric: {v_acc:0.4f}"

    for epoch in range(epochs):

        train_loss, train_metric = fit_epoch(model, train_loader, criterion, opt, device)

        val_loss, val_metric = eval_epoch(model, val_loader, criterion, device)

        history.append([train_loss, train_metric, val_loss, val_metric])

        print(log_template.format(ep=epoch + 1, t_loss=train_loss, v_loss=val_loss,
                                  t_acc=train_metric, v_acc=val_metric))

    return np.array(history)


def plot_training_history(history: np.array, figsize=(20, 5)):
    """
    Draws model training history on a graph
    :param history: output of train function
    :param figsize: figure size
    :return:
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax = ax.flatten()

    ax[0].plot(history[:, 0], label="Train")
    ax[0].plot(history[:, 2], label="Val")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(history[:, 1], label="Train")
    ax[1].plot(history[:, 3], label="Val")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Metric")
    ax[1].legend()
    ax[1].grid()

    plt.show()
