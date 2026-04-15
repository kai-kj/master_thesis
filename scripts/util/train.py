# %%

import numpy as np
import torch
from tqdm.auto import tqdm
import gc


# %%

def run_batch_train(
        device: torch.device,
        scaler: torch.amp.grad_scaler.GradScaler,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
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
        loss_fn: torch.nn.Module,
        x: torch.Tensor,
        y_true: torch.Tensor,
) -> float:
    model.eval()

    x, y_true = x.to(device), y_true.to(device).squeeze(1)

    with torch.inference_mode(), torch.amp.autocast_mode.autocast(device_type=device.type):
        y_pred = model(x)
        y_loss = loss_fn(y_pred, y_true)

    return y_loss.item()


def run_epoch(
        device: torch.device,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_valid: torch.utils.data.DataLoader,
        scaler: torch.amp.grad_scaler.GradScaler,
) -> tuple[float, float]:
    train_loss = []
    valid_loss = []

    for batch_x, batch_y in tqdm(dataloader_train, total=len(dataloader_train), colour="white", desc="  train"):
        loss = run_batch_train(device, scaler, model, optimizer, loss_fn, batch_x, batch_y)
        train_loss.append(loss)

    gc.collect()

    for batch_x, batch_y in tqdm(dataloader_valid, total=len(dataloader_valid), colour="white", desc="  valid"):
        loss = run_batch_valid(device, model, loss_fn, batch_x, batch_y)
        valid_loss.append(loss)

    gc.collect()

    return float(np.mean(train_loss)), float(np.mean(valid_loss))
