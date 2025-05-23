import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_data_loaders(num_devices, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    size = len(full) // num_devices
    loaders = {}
    for i in range(num_devices):
        idx = list(range(i*size, (i+1)*size))
        subset = Subset(full, idx)
        loaders[f'device_{i}'] = DataLoader(subset, batch_size=batch_size, shuffle=True)
    test = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test, batch_size=1000)
    return loaders, test_loader

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(10, 20, 5), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(320, 50), nn.ReLU(),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total

def fedavg(num_rounds=5, num_devices=3, lr=0.01, epochs=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders, test_loader = get_data_loaders(num_devices)
    global_model = SimpleCNN().to(device)

    for r in range(num_rounds):
        local_states = []
        for ld in loaders.values():
            m = SimpleCNN().to(device)
            m.load_state_dict(global_model.state_dict())
            opt = optim.SGD(m.parameters(), lr=lr)
            m.train()
            for _ in range(epochs):
                for x, y in ld:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    loss = nn.functional.cross_entropy(m(x), y)
                    loss.backward()
                    opt.step()
            local_states.append(m.state_dict())
        # average
        new_state = {}
        for k in global_model.state_dict().keys():
            new_state[k] = torch.stack([s[k] for s in local_states], 0).mean(0)
        global_model.load_state_dict(new_state)
        acc = evaluate(global_model, test_loader, device)
        print(f"FedAvg Round {r+1}: Accuracy={acc:.4f}")

def dla_ai_simulator():
    print("Extend SimulationRunner to accept a real model and dataset for DLA‑AI end-to-end.")

if __name__ == '__main__':
    print("=== FedAvg on MNIST ===")
    fedavg()
    print("=== DLA-AI Placeholder ===")
    dla_ai_simulator()
