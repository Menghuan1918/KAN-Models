import cv2
import numpy as np
import torch
import torchvision
from kan import KAN
import matplotlib.pyplot as plt

def preprocess_data(data):
    images = []
    labels = []
    for img, label in data:
        img = cv2.resize(np.array(img), (7, 7))
        img = img.flatten() / 255.0
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

train_data = torchvision.datasets.MNIST(
    root="./mnist_data", train=True, download=True, transform=None
)
test_data = torchvision.datasets.MNIST(
    root="./mnist_data", train=False, download=True, transform=None
)

train_images, train_labels = preprocess_data(train_data)
test_images, test_labels = preprocess_data(test_data)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device} device")

dataset = {
    "train_input": torch.from_numpy(train_images).float().to(device),
    "train_label": torch.from_numpy(train_labels).to(device),
    "test_input": torch.from_numpy(test_images).float().to("cpu"),
    "test_label": torch.from_numpy(test_labels).to("cpu"),
}

model = KAN(width=[49, 10, 10], device=device)

results = model.train(
    dataset,
    opt="Adam",
    lr=0.05,
    steps=100,
    batch=512,
    loss_fn=torch.nn.CrossEntropyLoss(),
)
torch.save(model.state_dict(), "kan.pth")


del model
model = KAN(width=[49, 10, 10], device="cpu")
model.load_state_dict(torch.load("kan.pth"))

def test_acc():
    with torch.no_grad():
        predictions = torch.argmax(model(dataset["test_input"]), dim=1)
        correct = (predictions == dataset["test_label"]).float()
        accuracy = correct.mean()
    return accuracy

acc = test_acc()
print(f"Test accuracy: {acc.item() * 100:.2f}%")

plt.plot(results["train_loss"], label="train")
plt.plot(results["test_loss"], label="test")
plt.legend()
plt.savefig("kan.png")