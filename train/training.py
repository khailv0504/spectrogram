import copy
import csv

import torch
import torch.nn as nn
from timm.utils import adaptive_clip_grad
from torch.amp import GradScaler
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

def train_model(model, train_loader, val_loader, DEVICE):
    NUM_EPOCHS = 30
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr = 3e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = GradScaler("cuda")
    best_acc = -1.0
    weights = None
    checked_missing_grad = False
    with open(r"D:\deep_learning\project_spectrogram\output\train_log.csv", mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()

            train_loss = 0
            correct = 0
            total = 0


            loop = tqdm(train_loader, leave=True)
            for batch in loop:
                optimizer.zero_grad(set_to_none=True)

                images = batch["image"].to(DEVICE, non_blocking=True)
                metadata = batch["meta"].to(DEVICE, non_blocking=True)
                labels = batch["label"].to(DEVICE, non_blocking=True)
                # Enables autocasting for the forward pass (model + loss)
                with autocast(device_type="cuda"):
                    # Forward
                    outputs = model(images, metadata)
                    loss = criterion(outputs, labels)

                # Scales loss. Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                scaler.scale(loss).backward()

                if not checked_missing_grad:
                    missing = [
                        name for name, p in model.named_parameters()
                        if p.requires_grad and p.grad is None
                    ]
                    if missing:
                        raise RuntimeError(
                            "Trainable parameters without gradients: " + ", ".join(missing)
                        )
                    checked_missing_grad = True

                scaler.unscale_(optimizer)
                adaptive_clip_grad(params, clip_factor=0.01)

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

                for batch in val_loader:
                    images = batch["image"].to(DEVICE, non_blocking=True)
                    metadata = batch["meta"].to(DEVICE, non_blocking=True)
                    labels = batch["label"].to(DEVICE, non_blocking=True)
                    outputs = model(images, metadata)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * labels.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = correct / total
            val_loss = val_loss / total

            loop.write(f"Epoch {epoch}/{NUM_EPOCHS}  Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")
            if (val_acc > best_acc):
                best_acc = val_acc
                weights = copy.deepcopy(model.state_dict())
                print("✹ Best accuracy: {:.4f}".format(best_acc*100))
            scheduler.step()
            writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc])
            f.flush()

    model.load_state_dict(weights)
    print(f"Best accuracy: {best_acc*100:.2f}%")
    return model
