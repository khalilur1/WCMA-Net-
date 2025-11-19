import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-GUI backend

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Channel Attention
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return x * out

# Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, in_channels, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * x

# Ablation Model
class AblationModel(nn.Module):
    def __init__(self, use_wavelet=False, use_denoising=False, use_multiscale=True, use_channel_attention=False, use_spatial_attention=False):
        super(AblationModel, self).__init__()
        self.use_wavelet = use_wavelet
        self.use_denoising = use_denoising
        self.use_multiscale = use_multiscale
        self.use_channel_attention = use_channel_attention
        self.use_spatial_attention = use_spatial_attention

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        if self.use_multiscale:
            self.conv3x3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv5x5 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
            self.conv7x7 = nn.Conv2d(32, 64, kernel_size=7, padding=3)
        else:
            self.conv = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        if self.use_channel_attention:
            self.channel_attention = ChannelAttention(192 if self.use_multiscale else 64)

        if self.use_spatial_attention:
            self.spatial_attention = SpatialAttention(192 if self.use_multiscale else 64)

        self.conv2 = nn.Conv2d(192 if self.use_multiscale else 64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))

        if self.use_multiscale:
            x3 = torch.relu(self.conv3x3(x))
            x5 = torch.relu(self.conv5x5(x))
            x7 = torch.relu(self.conv7x7(x))
            x = torch.cat([x3, x5, x7], dim=1)
        else:
            x = torch.relu(self.conv(x))

        if self.use_channel_attention:
            x = self.channel_attention(x)
        if self.use_spatial_attention:
            x = self.spatial_attention(x)

        x = self.pool2(torch.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training and Evaluation
def train_and_evaluate_variant(data_file, epochs=20, batch_size=16, lr=0.0001, **variant_params):
    data = torch.load(data_file)
    features, labels = data["features"], data["labels"]
    features = features.view(-1, 1, 64, 64)

    dataset = TensorDataset(features, labels)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = AblationModel(**variant_params).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    return model

# Main Program
if __name__ == "__main__":
    data_file = "processed_features.pt"
    epochs = 20
    batch_size = 16
    lr = 0.0001

    variants = [
        {"use_wavelet": False, "use_denoising": False, "use_multiscale": False, "use_channel_attention": False,
         "use_spatial_attention": False, "name": "base_model"},
        {"use_wavelet": False, "use_denoising": True, "use_multiscale": False, "use_channel_attention": False,
         "use_spatial_attention": False, "name": "Denoising_model"},
        {"use_wavelet": True, "use_denoising": False, "use_multiscale": False, "use_channel_attention": False,
         "use_spatial_attention": False, "name": "Wavelet_model"},
        {"use_wavelet": False, "use_denoising": False, "use_multiscale": True, "use_channel_attention": False,
         "use_spatial_attention": False, "name": "Multi-scale_model"},
        {"use_wavelet": False, "use_denoising": False, "use_multiscale": False, "use_channel_attention": True,
         "use_spatial_attention": False, "name": "channel_attention_model"},
        {"use_wavelet": False, "use_denoising": False, "use_multiscale": False, "use_channel_attention": False,
         "use_spatial_attention": True, "name": "spatial_attention_model"},
        {"use_wavelet": True, "use_denoising": True, "use_multiscale": True, "use_channel_attention": True,
         "use_spatial_attention": True, "name": "full_model"}
    ]

    for variant in variants:
        name = variant.pop("name")
        try:
            model = train_and_evaluate_variant(data_file, epochs, batch_size, lr, **variant)
            torch.save(model.state_dict(), f"{name}.pth")
            logger.info(f"Model for {name} saved.")
        except Exception as e:
            logger.error(f"Error training {name}: {e}")
