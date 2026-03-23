# %%

import numpy as np
import torch
from tqdm import tqdm


# %%

def run_batch_train(
        device: torch.device,
        scaler: torch.amp.grad_scaler.GradScaler,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.modules.loss._Loss,
        x: torch.Tensor,
        y_true: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad()

    x, y_true = x.to(device), y_true.to(device).squeeze(1)

    with torch.amp.autocast_mode.autocast(device_type=device.type):
        y_pred = model(x)
        y_loss = loss_fn(y_pred, y_true)

    scaler.scale(y_loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return y_loss.item()


def run_batch_valid(
        device: torch.device,
        model: torch.nn.Module,
        loss_fn: torch.nn.modules.loss._Loss,
        x: torch.Tensor,
        y_true: torch.Tensor,
) -> float:
    model.eval()

    x, y_true = x.to(device), y_true.to(device).squeeze(1)

    with torch.no_grad() and torch.amp.autocast_mode.autocast(device_type=device.type):
        y_pred = model(x)
        y_loss = loss_fn(y_pred, y_true)

    return y_loss.item()


def run_epoch(
        device: torch.device,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.modules.loss._Loss,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_valid: torch.utils.data.DataLoader,
) -> tuple[float, float]:
    scaler = torch.amp.grad_scaler.GradScaler(device=device.type)

    train_loss = []
    valid_loss = []

    for i, (batch_x, batch_y) in tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc="train"):
        loss = run_batch_train(device, scaler, model, optimizer, loss_fn, batch_x, batch_y)
        train_loss.append(loss)

    for i, (batch_x, batch_y) in tqdm(enumerate(dataloader_valid), total=len(dataloader_valid), desc="valid"):
        loss = run_batch_valid(device, model, loss_fn, batch_x, batch_y)
        valid_loss.append(loss)

    return float(np.mean(train_loss)), float(np.mean(valid_loss))
