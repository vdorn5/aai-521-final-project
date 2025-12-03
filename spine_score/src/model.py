import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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


# ==========================================
# V-Net Model Definition (Light Version)
# ==========================================

class VNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, kernel_size=3, padding=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            input_ch = in_channels if i == 0 else out_channels
            self.layers.append(nn.Sequential(
                nn.Conv3d(input_ch, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm3d(out_channels),
                nn.PReLU()
            ))

        if in_channels != out_channels:
            self.project_residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.project_residual = None

    def forward(self, x):
        residual = x
        out = x
        for layer in self.layers:
            out = layer(out)
        if self.project_residual is not None:
            residual = self.project_residual(residual)
        return out + residual

class VNetDeepSup(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, base_filters=16):
        super().__init__()
        
        # --- Encoder ---
        self.in_conv = nn.Conv3d(in_channels, base_filters, kernel_size=3, padding=1)
        self.block1 = VNetBlock(base_filters, base_filters, num_layers=1)
        self.down1 = nn.Conv3d(base_filters, base_filters*2, kernel_size=2, stride=2)
        
        self.block2 = VNetBlock(base_filters*2, base_filters*2, num_layers=2)
        self.down2 = nn.Conv3d(base_filters*2, base_filters*4, kernel_size=2, stride=2)
        
        self.block3 = VNetBlock(base_filters*4, base_filters*4, num_layers=3)
        self.down3 = nn.Conv3d(base_filters*4, base_filters*8, kernel_size=2, stride=2)
        
        # --- Bottleneck ---
        self.block4 = VNetBlock(base_filters*8, base_filters*8, num_layers=3)
        self.up4 = nn.ConvTranspose3d(base_filters*8, base_filters*4, kernel_size=2, stride=2)

        # --- Decoder ---
        self.block3_dec = VNetBlock(base_filters*8, base_filters*8, num_layers=3)
        self.up3 = nn.ConvTranspose3d(base_filters*8, base_filters*2, kernel_size=2, stride=2)
        
        self.block2_dec = VNetBlock(base_filters*4, base_filters*4, num_layers=2)
        self.up2 = nn.ConvTranspose3d(base_filters*4, base_filters, kernel_size=2, stride=2)
        
        self.block1_dec = VNetBlock(base_filters*2, base_filters*2, num_layers=1)
        
        # --- Output Heads ---
        self.final_conv = nn.Conv3d(base_filters*2, out_channels, kernel_size=1)
        
        # Deep Supervision Heads
        self.ds3_conv = nn.Conv3d(base_filters*8, out_channels, kernel_size=1)
        self.ds2_conv = nn.Conv3d(base_filters*4, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.in_conv(x)
        x1 = self.block1(x1)
        x2 = self.down1(x1)
        x2 = self.block2(x2)
        x3 = self.down2(x2)
        x3 = self.block3(x3)
        
        # Bottleneck
        x4 = self.down3(x3)
        x4 = self.block4(x4)
        
        # Decoder Level 3
        up4 = self.up4(x4)
        if up4.shape != x3.shape: up4 = self._pad_to_match(up4, x3)
        cat4 = torch.cat([x3, up4], dim=1)
        dec3 = self.block3_dec(cat4)
        
        # Decoder Level 2
        up3 = self.up3(dec3)
        if up3.shape != x2.shape: up3 = self._pad_to_match(up3, x2)
        cat3 = torch.cat([x2, up3], dim=1)
        dec2 = self.block2_dec(cat3)
        
        # Decoder Level 1
        up2 = self.up2(dec2)
        if up2.shape != x1.shape: up2 = self._pad_to_match(up2, x1)
        cat2 = torch.cat([x1, up2], dim=1)
        dec1 = self.block1_dec(cat2)
        
        final_out = self.final_conv(dec1)

        # Return list if training, single tensor if validation
        if self.training:
            return [final_out, self.ds2_conv(dec2), self.ds3_conv(dec3)]
        else:
            return final_out

    def _pad_to_match(self, src, target):
        diffZ = target.shape[2] - src.shape[2]
        diffY = target.shape[3] - src.shape[3]
        diffX = target.shape[4] - src.shape[4]
        return F.pad(src, [diffX // 2, diffX - diffX // 2,
                           diffY // 2, diffY - diffY // 2,
                           diffZ // 2, diffZ - diffZ // 2])


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

# ==========================================
# Deep Supervision Loss
# ==========================================
class DeepSupDiceCELoss(nn.Module):
    def __init__(self, weights=[1.0, 0.5, 0.25]):
        super().__init__()
        self.weights = weights
        self.base_loss = DiceCELoss() 

    def forward(self, outputs, targets):
        if not isinstance(outputs, list):
            return self.base_loss(outputs, targets)

        total_loss = 0
        target_shape = targets.shape[-3:]
        
        for i, pred in enumerate(outputs):
            if pred.shape[-3:] != target_shape:
                pred = F.interpolate(pred, size=target_shape, mode='trilinear', align_corners=False)
            
            layer_loss = self.base_loss(pred, targets)
            w = self.weights[i] if i < len(self.weights) else 0.0
            total_loss += w * layer_loss

        return total_loss



class GeneralizedDiceFocalLoss3D(nn.Module):
    """
    Combines:
      - Focal Cross Entropy (class-balanced)
      - Generalized Dice (volume-balanced)
    """
    def __init__(self, ce_weights, dice_weight=0.0, gamma=1.0, eps=1e-6):
        super().__init__()
        self.register_buffer("ce_weights", ce_weights.float())
        self.dice_weight = dice_weight
        self.gamma = gamma
        self.eps = eps

    def focal_ce(self, logits, targets):
        """
        logits: [B, C, D, H, W]
        targets: [B, D, H, W]
        """
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        targets_flat = targets.long()
        pt = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        log_pt = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)

        focal_term = (1 - pt) ** self.gamma
        class_w = self.ce_weights[targets_flat]

        loss = -class_w * focal_term * log_pt
        return loss.mean()

    def generalized_dice(self, logits, targets):
        """
        logits: [B, C, D, H, W]
        targets: [B, D, H, W]
        """
        probs = torch.softmax(logits, dim=1)
        num_classes = logits.shape[1]

        # One-hot encode
        target_1hot = F.one_hot(targets, num_classes=num_classes)
        target_1hot = target_1hot.permute(0, 4, 1, 2, 3).float()

        # Flatten spatial dims
        probs_f = probs.contiguous().view(logits.size(0), num_classes, -1)
        target_f = target_1hot.contiguous().view(logits.size(0), num_classes, -1)

        w = 1.0 / (target_f.sum(-1) ** 2 + self.eps)

        intersection = (probs_f * target_f).sum(-1)
        union = probs_f.sum(-1) + target_f.sum(-1)

        dice = (2 * intersection + self.eps) / (union + self.eps)
        gdice = 1.0 - (w * dice).sum() / w.sum()

        return gdice

    def forward(self, logits, targets):
        ce = self.focal_ce(logits, targets)
        dice = self.generalized_dice(logits, targets)
        total = ce + self.dice_weight * dice
        return total, ce, dice

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
    # One-hot encoding
    pred_one_hot = F.one_hot(pred, num_classes).permute(0, 4, 1, 2, 3)
    target_one_hot = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3)
    
    for c in range(num_classes):
        inter = torch.sum(pred_one_hot[:, c] * target_one_hot[:, c])
        union = torch.sum(pred_one_hot[:, c]) + torch.sum(target_one_hot[:, c])
        dice += (2 * inter + smooth) / (union + smooth)
    return dice / num_classes

# def dice_coeff(pred, target, num_classes=3, smooth=1e-5):
#     """
#     Fast Dice computation using label masks (no one-hot).
#     pred:   (B,H,W,D)
#     target: (B,H,W,D)
#     """
#     dice_scores = []

#     for c in range(num_classes):
#         p = (pred == c).float()
#         t = (target == c).float()

#         inter = (p * t).sum()
#         denom = p.sum() + t.sum()

#         dice = (2 * inter + smooth) / (denom + smooth)
#         dice_scores.append(dice)

#     return sum(dice_scores) / num_classes

# ===========================
# Training Loop
# ===========================
# def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=5e-5, num_classes=3):
#     model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     class_weights = torch.tensor([1.02, 54.9, 237.0], dtype=torch.float32)
#     class_weights = class_weights / class_weights.mean()
#     criterion = WeightedDiceCELoss3D(class_weights).to(device) # DiceCELoss()

#     history = {
#         "train_loss": [], 
#         "val_loss": [], 
#         "val_dice": []
#     }

#     for epoch in range(num_epochs):
#         model.train()

#         train_loss = 0
#         for images, masks, patient_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
#             images = images.to(device)
#             masks = masks.squeeze(1).to(device)
#             # patient_ids can be used for saving predictions or debugging

#             optimizer.zero_grad()

#             # New-style autocast for PyTorch 2.x
#             with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
#                 outputs = model(images)
#                 loss = criterion(outputs, masks)

#             # outputs = model(images)
#             # loss = criterion(outputs, masks)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()

#         history["train_loss"].append(train_loss / len(train_loader))

#         # Validation
#         model.eval()
#         val_loss = 0
#         all_dice = []
#         with torch.no_grad():
#             for images, masks, _ in val_loader:
#                 images = images.to(device)
#                 masks = masks.squeeze(1).to(device)
#                 outputs = model(images)
#                 loss = criterion(outputs, masks)
#                 val_loss += loss.item()

#                 probs = torch.softmax(outputs, dim=1)
#                 preds = torch.argmax(probs, dim=1)

#                 # Added (Edited for debugging)
#                 unique, counts = torch.unique(preds, return_counts=True)
#                 print("VAL PRED UNIQUE:", dict(zip(unique.cpu().numpy(), counts.cpu().numpy())))


#                 dice_score = dice_coeff(preds, masks, num_classes=num_classes)
#                 all_dice.append(dice_score.item())

#         history["val_loss"].append(val_loss / len(val_loader))
#         history["val_dice"].append(np.mean(all_dice))

#         print(f"Epoch [{epoch+1}/{num_epochs}] "
#               f"Train Loss: {history['train_loss'][-1]:.4f} "
#               f"Val Loss: {history['val_loss'][-1]:.4f} "
#               f"Val Dice: {history['val_dice'][-1]:.4f}")

#     return model, history


def show_epoch_preview(images, masks, outputs, epoch, save_dir=None, slice_mode="middle"):
    """
    Displays a single validation slice inline during training.
    Shows Original, Ground Truth, and Prediction.
    Also saves validation slice during training.

    images:  [B,1,D,H,W]
    masks:   [B,D,H,W]
    outputs: [B,C,D,H,W]
    """
    cmap = ListedColormap(["black", "yellow", "red"])

    # pick first sample in batch
    vol = images[0,0].cpu()
    true_mask = masks[0].cpu()
    
    # predicted
    probs = torch.softmax(outputs[0], dim=0)
    pred_mask = torch.argmax(probs, dim=0).cpu()

    # Slice Selection
    if slice_mode == "middle":
        mid = vol.shape[-1] // 2
    else:
        fg_slices = torch.where(true_mask.sum(dim=(0,1)) > 0)[0]
        mid = int(fg_slices[len(fg_slices)//2]) if len(fg_slices) > 0 else vol.shape[-1]//2

    # Plot
    plt.figure(figsize=(14,5))
    plt.suptitle(f"Epoch {epoch+1} — Validation Preview", fontsize=14)

    # Original
    plt.subplot(1,3,1)
    plt.title("Original slice")
    plt.imshow(vol[:,:,mid], cmap="gray")
    plt.axis("off")

    # Ground truth
    plt.subplot(1,3,2)
    plt.title("Ground truth mask")
    plt.imshow(vol[:,:,mid], cmap="gray")
    plt.imshow(true_mask[:,:,mid], cmap=cmap, alpha=0.6, vmin=0, vmax=2)
    plt.axis("off")

    # Prediction
    plt.subplot(1,3,3)
    plt.title("Prediction")
    plt.imshow(vol[:,:,mid], cmap="gray")
    plt.imshow(pred_mask[:,:,mid], cmap=cmap, alpha=0.6, vmin=0, vmax=2)
    plt.axis("off")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"epoch_{epoch+1:03d}.png")
        plt.savefig(save_path, dpi=120, bbox_inches="tight")

    plt.show()


def train_model_UNet(model, train_loader, val_loader, device,
                save_name="best_UNet.pth",
                num_epochs=20, lr=5e-5, num_classes=3):

    # ------------------------------
    # Directory for training outputs
    # ------------------------------
    run_name = save_name.replace(".pth", "")
    save_dir = os.path.join("../weights", run_name)
    os.makedirs(save_dir, exist_ok=True)

    best_model_path = os.path.join(save_dir, save_name)

    # ------------------------------
    # Setup model & optimizers
    # ------------------------------
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6
    )

    # Stable class weights
    # You can plug your custom values here instead
    ce_weights = torch.tensor([0.1, 1.0, 1.0], device=device).float()

    criterion = GeneralizedDiceFocalLoss3D(
        ce_weights=ce_weights,
        dice_weight=0.0,      # will be overwritten below
        gamma=1.0
    )

    scaler = amp.GradScaler()

    best_dice = -1
    history = {
        "train_loss": [], "train_ce": [], "train_dice": [],
        "val_loss": [], "val_dice": [], "val_per_class": []
    }

    def dice_warmup(epoch):
        """Smooth warm-up 0 → 0.4 over 20 epochs"""
        return min(0.4, epoch / 20.0)
    
    # --------------------------------------
    # Training Loop
    # --------------------------------------
    for epoch in range(num_epochs):
        # Hardcoded below
        criterion.dice_weight = 0.0 # dice_warmup(epoch)

        model.train()
        train_loss = train_ce = train_dice = 0

        for images, masks, _ in tqdm(train_loader,
                                     desc=f"Epoch {epoch+1}/{num_epochs}",
                                     leave=False):
            images = images.to(device)
            masks = masks.squeeze(1).to(device)

            optimizer.zero_grad()

            with amp.autocast(device_type=device):
                outputs = model(images)
                loss, ce_val, dice_val = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_ce += ce_val.item()
            train_dice += dice_val.item()

        n = len(train_loader)
        history["train_loss"].append(train_loss / n)
        history["train_ce"].append(train_ce / n)
        history["train_dice"].append(train_dice / n)

        # ---------------------
        # Validation
        # ---------------------
        model.eval()
        val_loss = 0
        dice_scores = []
        per_class_list = []

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.squeeze(1).to(device)

                outputs = model(images)
                loss, _, _ = criterion(outputs, masks)
                val_loss += loss.item()

                preds = outputs.argmax(1)

                # ----- Dice -------
                per_class = []
                for c in range(num_classes):
                    inter = ((preds == c) & (masks == c)).sum().item()
                    union = (preds == c).sum().item() + (masks == c).sum().item()
                    if union == 0:
                        per_class.append(1.0)
                    else:
                        per_class.append(2 * inter / union)
                per_class_list.append(per_class)

                dice_scores.append(np.mean(per_class))

        per_class_avg = np.mean(per_class_list, axis=0).tolist()

        # --------------------------
        # Save visualization image
        # --------------------------
        with torch.no_grad():
            val_images, val_masks, _ = next(iter(val_loader))
            val_images = val_images.to(device)
            val_masks = val_masks.squeeze(1).to(device)
            val_outputs = model(val_images)

        # Show inline preview
        show_epoch_preview(val_images, val_masks, val_outputs, epoch, save_dir=save_dir)

        # Store val metrics
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_dice"].append(np.mean(dice_scores))
        history["val_per_class"].append(per_class_avg)

        scheduler.step(history["val_dice"][-1])

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss={history['train_loss'][-1]:.4f} "
              f"CE={history['train_ce'][-1]:.4f} "
              f"DiceLoss={history['train_dice'][-1]:.4f} "
              f"ValDice={history['val_dice'][-1]:.4f} "
              f"(dice_w={criterion.dice_weight:.3f})")

        # ---------------------------------------
        # Save best model (via calculated dice)
        # ---------------------------------------
        if history["val_dice"][-1] > best_dice:
            best_dice = history["val_dice"][-1]
            torch.save(model.state_dict(), best_model_path)
            print(f"🔥 Saved new best model → {best_model_path}")

    return model, history


def train_vnet_deep_sup(model, train_loader, val_loader, device, 
                        save_name="best_VNet.pth", 
                        num_epochs=10, lr=1e-3, num_classes=3):

    # ------------------------------
    # Directory for training outputs
    # ------------------------------
    run_name = save_name.replace(".pth", "")
    save_dir = os.path.join("../weights", run_name)
    os.makedirs(save_dir, exist_ok=True)

    best_model_path = os.path.join(save_dir, save_name)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = DeepSupDiceCELoss(weights=[1.0, 0.5, 0.25])
    val_criterion = DiceCELoss() 
    
    best_dice = -1
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for images, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs_list = model(images) # Returns list of [High, Med, Low] res
            loss = criterion(outputs_list, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        history["train_loss"].append(train_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0
        all_dice = []
        
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                output = model(images) # Returns single tensor in eval()
                loss = val_criterion(output, masks)
                val_loss += loss.item()

                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                d = dice_coeff(preds, masks, num_classes=num_classes)
                all_dice.append(d.item())
        
        # Inline Validation Preview
        model.eval()
        with torch.no_grad():
            val_images, val_masks, _ = next(iter(val_loader))
            val_images = val_images.to(device)
            val_masks = val_masks.squeeze(1).to(device)
            val_outputs = model(val_images)

        show_epoch_preview(val_images, val_masks, val_outputs, epoch, save_dir=save_dir)

        history["val_loss"].append(val_loss / len(val_loader))
        history["val_dice"].append(np.mean(all_dice))

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {history['train_loss'][-1]:.4f} "
              f"Val Loss: {history['val_loss'][-1]:.4f} "
              f"Val Dice: {history['val_dice'][-1]:.4f}")
        
        # ---------------------------------------
        # Save best model (via calculated dice)
        # ---------------------------------------
        if history["val_dice"][-1] > best_dice:
            best_dice = history["val_dice"][-1]
            torch.save(model.state_dict(), best_model_path)
            print(f"🔥 Saved new best model → {best_model_path}")

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
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks  = masks.squeeze(1).to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

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
    
    # Average metrics over batches
    for key in all_metrics:
        all_metrics[key] = torch.tensor(all_metrics[key], dtype=torch.float32).mean(dim=0)

    all_dice = torch.tensor(all_dice)
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