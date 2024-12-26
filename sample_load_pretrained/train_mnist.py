import torch
import torchvision
import numpy as np

from classifier import ClassifierMNIST

from torchvision import datasets
from torch.utils.data import DataLoader


EPOCHS = 1


def _load_mnist():
    dataset = datasets.MNIST(
        root="/tmp/mnist",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    return dataset


def _train(dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ClassifierMNIST()
    model.to(device)
    
    criteria = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_history = []
    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criteria(output, target)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

        print("Epoch: {} loss: {}".format(
            epoch,
            np.mean(loss_history).item()
        ))

    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    dataset = _load_mnist()
    _train(dataset)
    