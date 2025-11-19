# File: Proposed_Wav_Mam_Pytorch.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Channel-Wise Attention
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
        self.conv1 = nn.Conv2d(2, in_channels, kernel_size=7, padding=3, bias=False)  # Preserve in_channels
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * x

# Wavelet-Based Mamba Attention Network
class MambaAttentionNetwork(nn.Module):
    def __init__(self):
        super(MambaAttentionNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Multi-scale Convolutions
        self.conv3x3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(32, 64, kernel_size=7, padding=3)

        # Attention
        self.channel_attention = ChannelAttention(192)
        self.spatial_attention = SpatialAttention(192)

        # Feature Refinement
        self.conv2 = nn.Conv2d(192, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Fully Connected
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x3 = torch.relu(self.conv3x3(x))
        x5 = torch.relu(self.conv5x5(x))
        x7 = torch.relu(self.conv7x7(x))
        x = torch.cat([x3, x5, x7], dim=1)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        x = self.pool2(torch.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to train and evaluate the model
def train_and_evaluate(data_file, epochs=20, batch_size=8, lr=0.0001):
    data = torch.load(data_file)
    features, labels = data["features"], data["labels"]
    features = features.view(-1, 1, 64, 64)

    dataset = TensorDataset(features, labels)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MambaAttentionNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_accuracy = 0
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

        # Validation
        val_loss, val_accuracy = evaluate_model(model, test_loader, criterion)
        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "mamba_attention_model.pth")
            logger.info(f"Best model saved with accuracy: {best_val_accuracy:.4f}")

    evaluate_final_model(model, test_loader)

# Evaluate Final Model
def evaluate_final_model(model, test_loader):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_score.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    conf_matrix = confusion_matrix(y_true, y_pred)

    logger.info(f"Accuracy: {acc}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Sensitivity (Recall): {recall}")
    logger.info(f"Specificity: {specificity}")
    logger.info(f"F1-Score: {f1}")
    logger.info(f"Area Under Curve (AUC): {auc}")
    logger.info(f"Confusion Matrix:\n {conf_matrix}")

    # Save confusion matrix plot
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    logger.info("Confusion matrix saved as confusion_matrix.png")

    # Save ROC curve
    plot_roc_curve(y_true, y_score, "roc_curve.png")

# Function to plot the ROC curve
def plot_roc_curve(y_true, y_score, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_score):.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    logger.info(f"ROC curve saved as {save_path}")

# Function to evaluate the model during training
def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    return total_loss / len(data_loader), accuracy

# Main Program
if __name__ == "__main__":
    train_and_evaluate("processed_features.pt", epochs=20, batch_size=8, lr=0.0001)
