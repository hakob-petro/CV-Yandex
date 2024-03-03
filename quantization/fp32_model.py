import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = nn.Conv2d(1, 12, 3)  # 12 filters of size 1x3x3
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2028, 10)

    def forward(self, input):
        x = self.conv(input)  # [B,  1, 28, 28] -> [B, 12, 26, 26]
        x = self.relu(x)  # [B, 12, 26, 26]
        x = self.maxpool(x)  # [B, 12, 26, 26] -> [B, 12, 13, 13]
        x = self.flatten(x)  # [B, 12, 13, 13] -> [B, 2028]
        x = self.fc(x)  # [B, 2028] -> [B, 10]
        return x


def train(model, train_loader, optimizer, loss_fn, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} "
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)}]"
                f"\tLoss: {loss.item():.4f}"
            )
    torch.save(model.state_dict(), "fp32_model.pth")


def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(1)
            correct += (pred == target).sum()
    print(
        f"Test accuracy: {correct}/{len(test_loader.dataset)} "
        f"({100. * correct / len(test_loader.dataset):.0f}%)\n"
    )


def main():
    model = SimpleNet()

    n_epochs = 3
    batch_size = 64

    train_dataset = torchvision.datasets.MNIST(
        "mnist",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataset = torchvision.datasets.MNIST(
        "mnist",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs + 1):
        train(model, train_loader, optimizer, loss_fn, epoch)
        test(model, test_loader)


if __name__ == "__main__":
    main()
