import copy
import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from timm.utils import adaptive_clip_grad
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    training_config: dict[str, Any] | None = None,
    train_log_path: str = "output/train_log.csv",
):
    training_config = training_config or {}

    num_epochs = int(training_config.get("num_epochs", 30))
    learning_rate = float(training_config.get("learning_rate", 3e-4))
    weight_decay = float(training_config.get("weight_decay", 1e-2))
    label_smoothing = float(training_config.get("label_smoothing", 0.1))
    min_lr = float(training_config.get("min_lr", 1e-6))
    grad_clip_factor = float(training_config.get("grad_clip_factor", 0.01))

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    best_acc = 0.0

    weights = None
    log_path = Path(train_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open(mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])

        for epoch in range(1, num_epochs + 1):
            model.train()

            train_loss = 0
            correct = 0
            total = 0

            loop = tqdm(train_loader, leave=True)
            for images, labels in loop:
                optimizer.zero_grad(set_to_none=True)
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                # Enables autocasting for the forward pass (model + loss)
                with autocast(device_type=device.type, enabled=use_amp):
                    # Forward
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                # Scales loss. Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                scaler.scale(loss).backward()

                if grad_clip_factor > 0:
                    scaler.unscale_(optimizer)
                    adaptive_clip_grad(params, clip_factor=grad_clip_factor)

                scaler.step(optimizer)
                scaler.update()
                # Compute metrics include accuracy, loss
                train_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loop.set_description(f"Epoch [{epoch}]")
                loop.set_postfix(loss=train_loss / total)
            train_acc = correct / total
            train_loss = train_loss / total

            # validation
            model.eval()
            correct = 0
            total = 0
            val_loss = 0

            # disable gradient computation during validation
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * labels.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = correct / total
            val_loss = val_loss / total

            loop.write(f"Epoch {epoch}/{num_epochs}  Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                weights = copy.deepcopy(model.state_dict())
                print("Best accuracy: {:.4f}".format(best_acc * 100))
            scheduler.step()
            writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc])
            f.flush()

    if weights is not None:
        model.load_state_dict(weights)

    print(f"Best accuracy: {best_acc * 100:.2f}%")
