import torch

from project_spectrogram.config import load_config
from project_spectrogram.module.spectrum_expert_model import SpectrumExpertModel
from project_spectrogram.datasets.processing import Preprocessing
from project_spectrogram.train.training import train_model
from project_spectrogram.utils.visualize import Visualize

if __name__ == "__main__":
    config = load_config()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipelinePreprocessing = Preprocessing(
        root_dir=config["paths"]["train_root_dir"],
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
