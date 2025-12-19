import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MLP(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 10),
        )

    def forward(self, x):
        return self.net(x)


def get_datasets(data_root: Path):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_sum += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return loss_sum / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(__file__).resolve().parent / "data"
    train_ds, test_ds = get_datasets(data_root)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)

    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 11):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch:02d}: train_loss={train_loss:.4f} test_loss={test_loss:.4f} accuracy={test_acc:.4%}"
        )

    start = time.perf_counter()
    evaluate(model, test_loader, device)
    end = time.perf_counter()
    print(f"Original MLP time: {end - start:.6f} seconds")

    from TASO_generated_rules import rules
    import onnx
    from onnxscript.rewriter import pattern
    from onnxscript import ir
    from onnx2pytorch import ConvertModel
    import onnxscript

    torch.onnx.export(
        model,
        (torch.randn(1, 1, 28, 28).to(device),),
        "mlp.onnx",
        opset_version=15,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
    )

    rewritten = onnxscript.rewriter.rewrite(
        model=onnx.load("mlp.onnx"),
        pattern_rewrite_rules=rules,
    )

    re_torch = ConvertModel(rewritten)
    start = time.perf_counter()
    evaluate(re_torch, test_loader, device)
    end = time.perf_counter()
    print(f"Rewritten MLP time: {end - start:.6f} seconds")


if __name__ == "__main__":
    main()
