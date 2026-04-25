import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


class Visualize:
    def __init__(
        self,
        val_loader,
        model,
        DEVICE,
        class_names,
        save_dir="output",
        train_log_path=None,
    ):
        self.all_labels = []
        self.all_predictions = []
        self.class_names = class_names
        self.save_dir = save_dir
        self.train_log_path = train_log_path or os.path.join(save_dir, "train_log.csv")

        os.makedirs(self.save_dir, exist_ok=True)

        model.eval()
        with torch.no_grad():
            progress = tqdm(val_loader, leave=True)
            for batch in progress:
                if isinstance(batch, dict):
                    images = batch["image"].to(DEVICE, non_blocking=True)
                    labels = batch["label"].to(DEVICE, non_blocking=True)
                    metadata = batch.get("meta")

                    if metadata is not None:
                        metadata = metadata.to(DEVICE, non_blocking=True)
                        outputs = model(images, metadata)
                    else:
                        outputs = model(images)
                else:
                    images, labels = batch
                    images = images.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)
                    outputs = model(images)

                _, predicted = torch.max(outputs, 1)
                self.all_labels.extend(labels.cpu().numpy())
                self.all_predictions.extend(predicted.cpu().numpy())

    def display_confusion_matrix(self, threshold=3):
        cm = confusion_matrix(self.all_labels, self.all_predictions)
        cm_norm = confusion_matrix(self.all_labels, self.all_predictions, normalize="true") * 100

        plt.figure(figsize=(8, 6))
        ax1 = sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )

        plt.title("Confusion Matrix", pad=20, fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        cbar1 = ax1.collections[0].colorbar
        cbar1.set_label("Count", rotation=0, labelpad=20, verticalalignment="center")

        plt.savefig(os.path.join(self.save_dir, "raw_cm.pdf"), format="pdf", bbox_inches="tight")
        plt.show()

        plt.figure(figsize=(8, 6))
        ax2 = sns.heatmap(
            cm_norm,
            annot=np.where(cm_norm < threshold, "", np.round(cm_norm, 2)),
            fmt="",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            vmin=0,
            vmax=100,
        )

        plt.title("Normalized Confusion Matrix", pad=20, fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        cbar2 = ax2.collections[0].colorbar
        cbar2.set_label("Accuracy (%)", rotation=0, labelpad=-40, y=1.05, ha="left")

        plt.savefig(os.path.join(self.save_dir, "confusion_matrix.pdf"), format="pdf", bbox_inches="tight")
        plt.show()

    def display_curve(self):
        if not os.path.exists(self.train_log_path):
            raise FileNotFoundError(f"Training log not found: {self.train_log_path}")

        df = pd.read_csv(self.train_log_path)
        train_loss = df["train_loss"]
        val_loss = df["val_loss"]
        train_acc = df["train_acc"]
        val_acc = df["val_acc"]
        epochs = range(1, len(train_loss) + 1)

        plt.figure()
        plt.plot(
            epochs,
            train_loss,
            label="Train Loss",
            marker="o",
            markersize=4,
            markevery=10,
            linestyle="-",
            linewidth=1.5,
        )
        plt.plot(
            epochs,
            val_loss,
            label="Validation Loss",
            marker="s",
            markersize=4,
            markevery=10,
            linestyle="-",
            linewidth=1.5,
        )

        min_train_epoch = np.argmin(train_loss) + 1
        min_val_epoch = np.argmin(val_loss) + 1

        plt.scatter(min_train_epoch, train_loss[min_train_epoch - 1], color="blue", zorder=5)
        plt.scatter(min_val_epoch, val_loss[min_val_epoch - 1], color="orange", zorder=5)

        plt.annotate(
            f"{train_loss[min_train_epoch - 1]:.4f}",
            (min_train_epoch, train_loss[min_train_epoch - 1]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
        )
        plt.annotate(
            f"{val_loss[min_val_epoch - 1]:.4f}",
            (min_val_epoch, val_loss[min_val_epoch - 1]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(os.path.join(self.save_dir, "loss_curve.pdf"), format="pdf", bbox_inches="tight")
        plt.show()

        plt.figure()
        plt.plot(
            epochs,
            train_acc,
            label="Train Accuracy",
            marker="o",
            markersize=4,
            markevery=10,
            linestyle="-",
            linewidth=1.5,
        )
        plt.plot(
            epochs,
            val_acc,
            label="Validation Accuracy",
            marker="s",
            markersize=4,
            markevery=10,
            linestyle="-",
            linewidth=1.5,
        )

        max_train_epoch = np.argmax(train_acc) + 1
        max_val_epoch = np.argmax(val_acc) + 1

        plt.scatter(max_train_epoch, train_acc[max_train_epoch - 1], color="blue", zorder=5)
        plt.scatter(max_val_epoch, val_acc[max_val_epoch - 1], color="orange", zorder=5)

        plt.annotate(
            f"{train_acc[max_train_epoch - 1]:.4f}",
            (max_train_epoch, train_acc[max_train_epoch - 1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        plt.annotate(
            f"{val_acc[max_val_epoch - 1]:.4f}",
            (max_val_epoch, val_acc[max_val_epoch - 1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(os.path.join(self.save_dir, "accuracy_curve.pdf"), format="pdf", bbox_inches="tight")
        plt.show()

    def display_report(self):
        return classification_report(self.all_labels, self.all_predictions)
