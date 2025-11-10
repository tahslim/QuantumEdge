# export_mobilenet_onnx.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# Simple MobileNet-v1 block
class DepthwiseSepConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layers(x)

class TinyMobileNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DepthwiseSepConv(32, 64),
            DepthwiseSepConv(64, 128, stride=2),
            DepthwiseSepConv(128, 128),
            DepthwiseSepConv(128, 256, stride=2),
            DepthwiseSepConv(256, 256),
            DepthwiseSepConv(256, 512, stride=2),
            *[DepthwiseSepConv(512, 512) for _ in range(5)],
            DepthwiseSepConv(512, 1024, stride=2),
            DepthwiseSepConv(1024, 1024),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def main():
    # Use CIFAR-10 for lightweight training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    model = TinyMobileNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training TinyMobileNet on CIFAR-10 (1 epoch for demo)...")
    model.train()
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        break  # Just 1 batch for demo; remove in real use

    # Export to ONNX
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model,
        dummy_input,
        "models/mobilenet_cifar10.onnx",
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    )
    print("✅ ONNX model saved to models/mobilenet_cifar10.onnx")

if __name__ == "__main__":
    main()
