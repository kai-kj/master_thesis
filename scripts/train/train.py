# %%

import gc
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


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
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: torch.nn.Module,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_valid: torch.utils.data.DataLoader,
        scaler: torch.amp.grad_scaler.GradScaler,
) -> tuple[float, float, float]:
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

    scheduler.step()

    return float(np.mean(train_loss)), float(np.mean(valid_loss)), scheduler.get_last_lr()[0]


def run_loop(
        device: torch.device,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: torch.nn.Module,
        num_epochs: int,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_valid: torch.utils.data.DataLoader,
        model_path: Path,
):
    print(f"training model {model_path.name}")

    model_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    scaler = torch.amp.grad_scaler.GradScaler(device=device.type)
    history = []

    for epoch in range(num_epochs):
        print(f"epoch {epoch + 1}/{num_epochs}")

        curr_train_loss, curr_valid_loss, curr_lr = run_epoch(
            device,
            model,
            optimizer, scheduler, loss_fn,
            dataloader_train, dataloader_valid,
            scaler
        )

        history.append((epoch, curr_lr, curr_train_loss, curr_valid_loss))

        print(f"  lr:         {history[-1][1]:.6f}")
        print(f"  train loss: {history[-1][2]:.6f}")
        print(f"  valid loss: {history[-1][3]:.6f}")

        if len(history) == 1 or history[-1][3] < np.min([h[3] for h in history[:-1]]):
            print("  best model, saving")
            torch.save(model.state_dict(), model_path / "model.pt")

        pd.DataFrame(history, columns=["epoch", "lr", "train_loss", "valid_loss"]) \
            .to_csv(model_path / "history.csv", index=False)

        current_time = time.time()
        time_per_epoch = (current_time - start_time) / (epoch + 1)
        remaining_time = time_per_epoch * (num_epochs - epoch - 1)

        print(f"  {time.strftime("%H:%M:%S", time.gmtime(time_per_epoch))} per epoch")
        print(f"  {time.strftime("%H:%M:%S", time.gmtime(remaining_time))} remaining")

    end_time = time.time()
    total_time = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))

    print(f"done, training time for {num_epochs} epochs: {total_time}")
