import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

from torchvision import transforms
import torch.nn.functional as F

from loss.InverseDepthLoss import InverseDepthSmoothnessLoss as IDSL
from data.loaders.DataLoader import RedWebDataset, Rescale
import torch.optim as optim


class ResNetWithFeatureFusion(nn.Module):
    def __init__(self):
        super(ResNetWithFeatureFusion, self).__init__()

        # Load the pretrained ResNet50 model
        self.resnet50 = models.resnet50(pretrained=True)

        # Extract features from intermediate layers
        self.layer1_features = nn.Sequential(
            self.resnet50.conv1,
            self.resnet50.bn1,
            self.resnet50.relu,
            self.resnet50.maxpool,
            self.resnet50.layer1
        )

        self.layer2_features = self.resnet50.layer2
        self.layer3_features = self.resnet50.layer3

        # Additional layers for feature fusion
        self.fusion_conv = nn.Conv2d(
            in_channels=1024, out_channels=3, kernel_size=1)
        self.fusion_bn = nn.BatchNorm2d(3)
        self.fusion_relu = nn.ReLU(inplace=True)

    def __call__(self, x):
        # Backbone ResNet as suggested Xian et.al
        x = self.layer1_features(x)
        x = self.layer2_features(x)
        x = self.layer3_features(x)

        # Feature Fusion (from Xian et.al)
        fused_features = self.fusion_conv(x)
        fused_features = self.fusion_bn(fused_features)
        fused_features = self.fusion_relu(fused_features)
        fused_features = F.interpolate(fused_features, size=(
            256, 256), mode='bilinear', align_corners=False)

        return fused_features


def training_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 10
    learning_rate = 0.001
    batch_size = 32

    dataset = normal_dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ResNetWithFeatureFusion().to(device)

    idsl = IDSL()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            inputs = batch['mono'].to(device)
            targets = batch['heat'].to(device)
            inputs = inputs.float()

            # flush the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            outputs = outputs.float()
            targets = targets.float()
            # Calculate the loss
            # print(targets.shape , outputs.shape)
            loss = idsl(targets, outputs)
            print("loss ->", loss)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print the average loss for the epoch
        average_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

    torch.save(model.state_dict(), 'resnet_with_feature_fusion.pth')

    pass


if __name__ == "__main__":
    normal_dataset = RedWebDataset(
        root_dir="ReDWeb_V1",
        transform=transforms.Compose([Rescale((256, 256))]))
    batcher = DataLoader(normal_dataset, batch_size=1, shuffle=True)
    model = ResNetWithFeatureFusion()
    training_loop()
