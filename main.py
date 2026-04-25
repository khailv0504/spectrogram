import torch

from project_spectrogram.module.spectrum_expert_model import SpectrumExpertModel
from project_spectrogram.datasets.processing import Preprocessing
from project_spectrogram.train.training import train_model
from project_spectrogram.utils.visualize import Visualize

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = r"D:\deep_learning\kagglehub\datasets\huynhthethien\radarcommunsignaldata2026train\versions\1"  # folder gốc
    pipelinePreprocessing = Preprocessing(root_dir)
    train_loader, val_loader, class_names = pipelinePreprocessing.process()
    model = SpectrumExpertModel(num_classes=12).to(DEVICE)

    model = train_model(model, train_loader, val_loader, DEVICE)
    visualizing = Visualize(val_loader, model, DEVICE, class_names)
    visualizing.display_report()
    visualizing.display_curve()
    visualizing.display_confusion_matrix()

    model.eval()
    example_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    # convert PyTorch model into TorchScript format
    traced_model = torch.jit.trace(model, example_input)

    # save model
    model_name = f"TrainedModel.pt"
    traced_model.save(model_name)
    print("Model saved:", model_name)
