# %%

import gc
from pathlib import Path
import time

import numpy as np
import sklearn.metrics as skm
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
        return_y: bool = False,
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    model.train()
    optimizer.zero_grad()

    x, y_true = x.to(device), y_true.to(device).squeeze(1)

    with torch.amp.autocast_mode.autocast(device_type=device.type):
        y_pred = model(x)
        y_loss = loss_fn(y_pred, y_true)

    scaler.scale(y_loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if return_y:
        return y_loss.item(), y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
    else:
        return y_loss.item(), None, None


def run_batch_test(
        device: torch.device,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        x: torch.Tensor,
        y_true: torch.Tensor,
        return_y: bool = False,
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    model.eval()

    x, y_true = x.to(device), y_true.to(device).squeeze(1)

    with torch.inference_mode(), torch.amp.autocast_mode.autocast(device_type=device.type):
        y_pred = model(x)
        y_loss = loss_fn(y_pred, y_true)

    if return_y:
        y_true_np = y_true.detach().cpu().numpy().flatten()
        y_pred_np = np.argmax(y_pred.detach().cpu().numpy(), axis=1).flatten()
        return y_loss.item(), y_true_np, y_pred_np
    else:
        return y_loss.item(), None, None


def run_epoch(
        device: torch.device,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: torch.nn.Module,
        dl_train: torch.utils.data.DataLoader,
        dl_valid: torch.utils.data.DataLoader,
        scaler: torch.amp.grad_scaler.GradScaler,
        n_classes: int,
        calc_confusion: bool = False,
        silent: bool = True,
) -> tuple[float, float, float, np.ndarray | None]:
    train_loss = []
    valid_loss = []

    for batch_x, batch_y in tqdm(dl_train, total=len(dl_train), colour="white", desc="  train", disable=silent):
        loss, _, _ = run_batch_train(device, scaler, model, optimizer, loss_fn, batch_x, batch_y)
        train_loss.append(loss)

    gc.collect()

    y_pred_all = []
    y_true_all = []

    for batch_x, batch_y in tqdm(dl_valid, total=len(dl_valid), colour="white", desc="  valid", disable=silent):
        loss, y_true, y_pred = run_batch_test(device, model, loss_fn, batch_x, batch_y, return_y=calc_confusion)
        valid_loss.append(loss)
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

    gc.collect()

    scheduler.step()

    return (
        float(np.mean(train_loss)),
        float(np.mean(valid_loss)),
        float(scheduler.get_last_lr()[0]),
        skm.confusion_matrix(
            np.concatenate(y_true_all, axis=0),
            np.concatenate(y_pred_all, axis=0),
            labels=list(range(n_classes))
        ) if calc_confusion else None
    )


def run_loop(
        device: torch.device,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: torch.nn.Module,
        num_epochs: int,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_valid: torch.utils.data.DataLoader,
        model_path: str | Path,
        n_classes: int,
        silent: bool = False,
):
    model_path = Path(model_path)

    print(f"training model {model_path.name}")

    model_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    scaler = torch.amp.grad_scaler.GradScaler(device=device.type)
    history = []

    for epoch in range(num_epochs):
        if not silent:
            print(f"epoch {epoch + 1}/{num_epochs}")

        curr_train_loss, curr_valid_loss, curr_lr, confusion_matrix = run_epoch(
            device,
            model,
            optimizer, scheduler, loss_fn,
            dataloader_train, dataloader_valid,
            scaler,
            n_classes,
            calc_confusion=True,
            silent=silent
        )

        assert confusion_matrix is not None
        confusion_matrix = confusion_matrix.flatten()

        history.append((epoch, curr_lr, curr_train_loss, curr_valid_loss, *confusion_matrix))

        if not silent:
            #     print(f"  lr:         {history[-1][1]:.6f}")
            print(f"  loss: {history[-1][2]:.6f} (train), {history[-1][3]:.6f} (valid)")

        if len(history) == 1 or history[-1][3] < np.min([h[3] for h in history[:-1]]):
            if not silent: print("  best model, saving")
            model.save(model_path / "model.json", model_path / "model.pt")  # type: ignore

        confusion_matrix_cols = [f"cm_{i:03d}" for i in range(len(confusion_matrix))]

        pd.DataFrame(history, columns=["epoch", "lr", "train_loss", "valid_loss"] + confusion_matrix_cols) \
            .to_csv(model_path / "history.csv", index=False)

        current_time = time.time()
        time_per_epoch = (current_time - start_time) / (epoch + 1)
        remaining_time = time_per_epoch * (num_epochs - epoch - 1)

        if not silent:
            print(
                f"  {time.strftime("%H:%M:%S", time.gmtime(remaining_time))} remaining" +
                f" ({time.strftime("%H:%M:%S", time.gmtime(time_per_epoch))} per epoch)"
            )

        if silent:
            print(
                f"- epoch {epoch + 1}/{num_epochs} " +
                f"- loss: {history[-1][2]:.4f} (train), {history[-1][3]:.4f} (valid) " +
                f"- {time.strftime("%H:%M:%S", time.gmtime(remaining_time))} remaining"
            )

    end_time = time.time()
    total_time = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))

    print(f"done, training time for {num_epochs} epochs: {total_time}")
