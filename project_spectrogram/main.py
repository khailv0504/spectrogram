import os
import pathlib
import torch

from project_spectrogram.config import load_config
from project_spectrogram.module.spectrum_expert_model import SpectrumExpertModel
from project_spectrogram.datasets.processing import Preprocessing
from project_spectrogram.train.training import train_model
from project_spectrogram.utils.visualize import Visualize


def resolve_train_root_dir(config: dict) -> str:
    paths_cfg = config.setdefault("paths", {})

    train_root_dir = paths_cfg.get("train_root_dir")
    if train_root_dir:
        return str(train_root_dir)

    dataset_dir_from_env = os.environ.get("KAGGLE_TRAIN_DATASET_DIR")
    if dataset_dir_from_env:
        paths_cfg["train_root_dir"] = dataset_dir_from_env
        return dataset_dir_from_env

    kaggle_dataset_id = paths_cfg.get("kaggle_dataset_id")
    if kaggle_dataset_id:
        dataset_slug = str(kaggle_dataset_id).split("/", 1)[-1]
        kaggle_input_dir = pathlib.Path("/kaggle/input") / dataset_slug
        if kaggle_input_dir.exists():
            paths_cfg["train_root_dir"] = str(kaggle_input_dir)
            return str(kaggle_input_dir)

        try:
            import kagglehub  # type: ignore
        except ModuleNotFoundError as exc:
            raise ValueError(
                "Cannot resolve dataset path. Set paths.train_root_dir, "
                "or set KAGGLE_TRAIN_DATASET_DIR, "
                "or install kagglehub to use paths.kaggle_dataset_id."
            ) from exc

        dataset_path = kagglehub.dataset_download(str(kaggle_dataset_id))
        paths_cfg["train_root_dir"] = dataset_path
        return dataset_path

    raise ValueError(
        "Missing dataset configuration. Set one of: "
        "paths.train_root_dir, KAGGLE_TRAIN_DATASET_DIR, or paths.kaggle_dataset_id."
    )


if __name__ == "__main__":
    config = load_config()
    train_root_dir = resolve_train_root_dir(config)
    print("Train dataset root:", train_root_dir)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipelinePreprocessing = Preprocessing(
        root_dir=train_root_dir,
        data_config=config["data"],
    )
    train_loader, val_loader, class_names = pipelinePreprocessing.process()

    num_classes = int(config["model"].get("num_classes", len(class_names)))
    if num_classes != len(class_names):
        print(
            f"Warning: model.num_classes={num_classes}, "
            f"but found {len(class_names)} classes in dataset."
        )

    model = SpectrumExpertModel(num_classes=num_classes).to(DEVICE)
    print(sum(p.numel() for p in model.parameters()))

    train_model(
        model,
        train_loader,
        val_loader,
        DEVICE,
        training_config=config["training"],
        train_log_path=config["paths"]["train_log_path"],
    )

    visualizing = Visualize(
        val_loader=val_loader,
        model=model,
        DEVICE=DEVICE,
        class_names=class_names,
        save_dir=config["visualization"]["save_dir"],
        train_log_path=config["paths"]["train_log_path"],
    )
    print(visualizing.display_report())
    visualizing.display_curve()
    visualizing.display_confusion_matrix(
        threshold=float(config["visualization"].get("confusion_threshold", 3))
    )

    model.eval()
    example_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    traced_model = torch.jit.trace(model, example_input)

    # save model
    model_name = config["paths"]["model_output_path"]
    traced_model.save(model_name)
    print("Model saved:", model_name)
