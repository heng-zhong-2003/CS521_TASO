import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# -----------------------------
# LeNet-5 definition
# -----------------------------
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# -----------------------------
# Dataset utilities (MNIST)
# -----------------------------
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


# -----------------------------
# Main experiment
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(__file__).resolve().parent / "data"
    train_ds, test_ds = get_datasets(data_root)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    model = LeNet5().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- Train briefly (can shorten for faster tests)
    for epoch in range(1, 6):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch:02d}: train_loss={train_loss:.4f} "
            f"test_loss={test_loss:.4f} accuracy={test_acc:.4%}"
        )

    # --- Baseline timing ---
    start = time.perf_counter()
    evaluate(model, test_loader, device)
    end = time.perf_counter()
    print(f"Original LeNet5 time: {end - start:.6f} seconds")

    # -----------------------------
    # Apply TASO-generated rewrites
    # -----------------------------
    from TASO_generated_rules import rules
    import onnx
    from onnxscript.rewriter import pattern
    from onnxscript import ir
    from onnx2pytorch import ConvertModel
    import onnxscript

    # --- Export to ONNX ---
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    torch.onnx.export(
        model,
        (dummy_input,),
        "lenet5.onnx",
        opset_version=15,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
    )

    # --- Apply all generated rewrite rules ---
    # rewritten = onnxscript.rewriter.rewrite(
        # model=onnx.load("lenet5.onnx"),
        # pattern_rewrite_rules=rules,
    # )

    rewritten = onnx.load("lenet5.onnx")
    # --- Convert back to PyTorch ---
    re_torch = ConvertModel(rewritten).to(device)

    # --- Timing after rewrite ---
    start = time.perf_counter()
    evaluate(re_torch, test_loader, device)
    end = time.perf_counter()
    print(f"Rewritten LeNet5 time: {end - start:.6f} seconds")


if __name__ == "__main__":
    main()
