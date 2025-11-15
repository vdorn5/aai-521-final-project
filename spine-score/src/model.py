import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


# ===========================
# 3D UNet Model
# ===========================
class DoubleConv(nn.Module):
    """(Conv3D => ReLU => Conv3D => ReLU)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SimpleUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, features=[32, 64, 128]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)
        prev_ch = in_channels

        # Encoder
        for f in features:
            self.encoder.append(DoubleConv(prev_ch, f))
            prev_ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(prev_ch, prev_ch * 2)
        prev_ch = prev_ch * 2

        # Decoder
        self.upconv = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for f in reversed(features):
            self.upconv.append(nn.ConvTranspose3d(prev_ch, f, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(prev_ch, f))
            prev_ch = f

        # Final layer
        self.final_conv = nn.Conv3d(prev_ch, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.upconv)):
            x = self.upconv[idx](x)
            skip = skip_connections[idx]

            # Pad if needed
            if x.shape != skip.shape:
                diffZ = skip.shape[2] - x.shape[2]
                diffY = skip.shape[3] - x.shape[3]
                diffX = skip.shape[4] - x.shape[4]
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2,
                              diffZ // 2, diffZ - diffZ // 2])

            x = torch.cat([skip, x], dim=1)
            x = self.decoder[idx](x)

        return self.final_conv(x)


# ===========================
# Dice Loss + CrossEntropy
# ===========================
class DiceCELoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, outputs, targets):
        # outputs: (B,C,H,W,D)
        # targets: (B,H,W,D)
        ce_loss = self.ce(outputs, targets)

        num_classes = outputs.shape[1]
        outputs_soft = torch.softmax(outputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()

        smooth = 1e-5
        intersection = torch.sum(outputs_soft * targets_one_hot)
        union = torch.sum(outputs_soft + targets_one_hot)
        dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)

        return ce_loss + dice_loss


# ===========================
# Dice coefficient metric
# ===========================
def dice_coeff(pred, target, num_classes=3, smooth=1e-5):
    """
    Compute per-class Dice coefficient
    pred: (B,H,W,D)
    target: (B,H,W,D)
    """
    dice = 0
    pred_one_hot = F.one_hot(pred, num_classes).permute(0, 4, 1, 2, 3)
    target_one_hot = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3)
    for c in range(num_classes):
        inter = torch.sum(pred_one_hot[:, c] * target_one_hot[:, c])
        union = torch.sum(pred_one_hot[:, c]) + torch.sum(target_one_hot[:, c])
        dice += (2 * inter + smooth) / (union + smooth)
    return dice / num_classes


# ===========================
# Training Loop
# ===========================
def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=1e-3, num_classes=3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = DiceCELoss()

    history = {
        "train_loss": [], 
        "val_loss": [], 
        "val_dice": []
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        history["train_loss"].append(train_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0
        all_dice = []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                dice_score = dice_coeff(preds, masks, num_classes=num_classes)
                all_dice.append(dice_score.item())

        history["val_loss"].append(val_loss / len(val_loader))
        history["val_dice"].append(np.mean(all_dice))

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {history['train_loss'][-1]:.4f} "
              f"Val Loss: {history['val_loss'][-1]:.4f} "
              f"Val Dice: {history['val_dice'][-1]:.4f}")

    return model, history

def compute_segmentation_metrics(preds, masks, num_classes=3):
    """
    Compute Accuracy, Precision, Recall, F1 per class.
    
    preds: torch.Tensor, shape [B,H,W,D], predicted class indices
    masks: torch.Tensor, shape [B,H,W,D], ground truth class indices
    """
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for cls in range(num_classes):
        pred_cls = (preds == cls)
        mask_cls = (masks == cls)

        TP = (pred_cls & mask_cls).sum().item()
        FP = (pred_cls & (~mask_cls)).sum().item()
        FN = ((~pred_cls) & mask_cls).sum().item()
        TN = ((~pred_cls) & (~mask_cls)).sum().item()

        acc = (TP + TN) / (TP + TN + FP + FN + 1e-6)
        prec = TP / (TP + FP + 1e-6)
        rec = TP / (TP + FN + 1e-6)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)

        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1'].append(f1)

    return metrics

def evaluate_model(model, dataloader, device, num_classes=3):
    """
    Evaluate a trained segmentation model on a dataset.
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader for the dataset
        device: 'cuda' or 'cpu'
        num_classes: number of segmentation classes
    
    Returns:
        mean_dice: mean Dice coefficient over all samples
        class_dice: list of Dice per class
    """
    model.eval()
    all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    all_dice = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)           # (B, C, H, W, D)
            masks  = masks.squeeze(1).to(device) # remove channel dim if needed: (B, H, W, D)

            outputs = model(images)              # (B, num_classes, H, W, D)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)  # (B, H, W, D)

            batch_dice = []
            for cls in range(num_classes):
                pred_cls = (preds == cls).float()
                mask_cls = (masks == cls).float()
                intersection = (pred_cls * mask_cls).sum()
                union = pred_cls.sum() + mask_cls.sum()
                dice = (2 * intersection + 1e-6) / (union + 1e-6)
                batch_dice.append(dice.item())
            all_dice.append(batch_dice)
            metrics = compute_segmentation_metrics(preds, masks, num_classes=3)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
    
    # Average over batches
    for key in all_metrics:
        all_metrics[key] = torch.tensor(all_metrics[key], dtype=torch.float32).mean(dim=0)

    all_dice = torch.tensor(all_dice)          # shape: (num_samples, num_classes)
    mean_dice = all_dice.mean().item()
    class_dice = all_dice.mean(dim=0).tolist() # Dice per class

    return mean_dice, class_dice, metrics

def print_metrics(metrics, class_names=None):
    """
    metrics: dict with keys 'accuracy', 'precision', 'recall', 'f1'
             each value is a list per class
    class_names: optional list of class names
    """
    import pandas as pd

    # If no class names were provided, number them
    num_classes = len(metrics["accuracy"])
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    df = pd.DataFrame({
        "Class": class_names,
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1 Score": metrics["f1"]
    })

    print("\n===== Evaluation Metrics per Class =====\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n========================================\n")