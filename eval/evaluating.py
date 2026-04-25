import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pathlib
from tqdm import tqdm


if __name__ == "__main__":
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # sinh viên thay đường dẫn
    MODEL_PATH = pathlib.Path("")

    TEST_DATASET_DIR = pathlib.Path(r"D:\deep_learning\kagglehub\datasets\huynhthethien\radarcommunsignaldata2026test\versions\1")

    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])


    test_dataset = datasets.ImageFolder(
        root=str(TEST_DATASET_DIR),
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    print("\nTest dataset loaded")
    print("Classes:", test_dataset.classes)
    print("Total test samples:", len(test_dataset))

    print("\n==============================")
    print("EVALUATING MODEL")
    print("==============================")

    print("Model:", MODEL_PATH.name)


    def evaluate_model(model_path):

        try:
            model = torch.jit.load(model_path)
            model.to(DEVICE)
            model.eval()

            correct = 0
            total = 0

            with torch.no_grad():
                pbar = tqdm(test_loader)
                for images, labels in pbar:
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)

                    outputs = model(images)

                    _, predicted = torch.max(outputs, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    pbar.set_postfix(acc=correct / total)

            accuracy = correct / total

            return accuracy, correct, total
        except Exception as e:
            print(f"\nERROR loading model: {model_path}")
            print(e)
            return None, None, None

    acc, correct, total = evaluate_model(MODEL_PATH)

    if acc is not None:

        print("\n==============================")
        print("RESULT")
        print("==============================")

        print(f"Accuracy : {acc:.4f}")
        print(f"Correct  : {correct}/{total}")

