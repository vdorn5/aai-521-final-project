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
# Weighted Dice Loss
# ===========================
# class WeightedDiceCELoss3D(nn.Module):
#     def __init__(self, class_weights=None):
#         """
#         class_weights: list or tensor of shape [C]
#                        e.g. [0.05, 1.0, 1.0]
#         """
#         super().__init__()

#         if class_weights is not None:
#             class_weights = torch.tensor(class_weights, dtype=torch.float32)
#         self.register_buffer("class_weights", class_weights)

#         # Cross entropy uses the weights directly
#         self.ce = nn.CrossEntropyLoss(weight=self.class_weights)

#     def forward(self, outputs, targets):
#         # outputs: (B,C,H,W,D)
#         # targets: (B,H,W,D)

#         ce_loss = self.ce(outputs, targets)

#         num_classes = outputs.shape[1]

#         # softmax probabilities
#         probs = torch.softmax(outputs, dim=1)

#         # one-hot targets
#         targets_1h = F.one_hot(targets, num_classes)   # (B,H,W,D,C)
#         targets_1h = targets_1h.permute(0, 4, 1, 2, 3).float()  # (B,C,H,W,D)

#         smooth = 1e-5

#         # --- PER CLASS Dice ---
#         dims = (0, 2, 3, 4)   # sum over batch + spatial dims

#         intersection = torch.sum(probs * targets_1h, dim=dims)
#         denominator = torch.sum(probs + targets_1h, dim=dims)

#         dice_per_class = (2 * intersection + smooth) / (denominator + smooth)

#         # convert weights
#         if self.class_weights is not None:
#             weights = self.class_weights
#             weights = weights / weights.sum()   # normalize to sum to 1
#         else:
#             weights = torch.ones(num_classes, dtype=torch.float32, device=outputs.device) / num_classes

#         # weighted DICE loss = 1 - Σ w_c * dice_c
#         weighted_dice = 1 - torch.sum(weights * dice_per_class)

#         return ce_loss + weighted_dice
    


# class WeightedDiceCELoss3D(nn.Module):
#     def __init__(self, class_weights):
#         super().__init__()

#         if not isinstance(class_weights, torch.Tensor):
#             class_weights = torch.tensor(class_weights, dtype=torch.float32)

#         # Normalize class weights (important!)
#         class_weights = class_weights / class_weights.mean()

#         # Store CE weights
#         self.register_buffer("class_weights", class_weights)

#         # Prepare dice weights (foreground only)
#         fg = class_weights[1:]                   # ignore background
#         dice_weights = 1.0 / fg                  # inverse-frequency
#         dice_weights = dice_weights / dice_weights.max()

#         self.register_buffer("dice_weights", dice_weights)

#         print("Initialized CE weights:", class_weights)
#         print("Initialized Dice weights:", dice_weights)

#     def forward(self, outputs, targets):
#         outputs = outputs.float()
#         targets = targets.long()

#         cw = self.class_weights.to(outputs.device)
#         dw = self.dice_weights.to(outputs.device)

#         # ----------------------------------------
#         # 1. Cross-entropy loss
#         # ----------------------------------------
#         ce_loss = F.cross_entropy(outputs, targets, weight=cw)

#         # ----------------------------------------
#         # 2. Dice loss (now includes background!)
#         # ----------------------------------------
#         smooth = 1e-5
#         num_classes = outputs.shape[1]

#         probs = torch.softmax(outputs, dim=1)
#         targets_1h = F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()

#         dice_loss = torch.tensor(0.0, device=outputs.device)
#         per_class_dice = []

#         for c in range(num_classes):
#             p = probs[:, c]
#             t = targets_1h[:, c]

#             inter = (p * t).sum()
#             union = p.sum() + t.sum()
#             dice_c = (2 * inter + smooth) / (union + smooth)

#             # background gets weight = 1.0
#             if c == 0:
#                 w = 1.0
#             else:
#                 w = dw[c - 1]

#             dice_loss += w * (1 - dice_c)
#             per_class_dice.append(dice_c.item())

#         # ----------------------------------------
#         # Total loss
#         # ----------------------------------------
#         total_loss = ce_loss + 5.0 * dice_loss

#         return total_loss, ce_loss.detach(), dice_loss.detach(), per_class_dice

# class WeightedDiceCELoss3D(nn.Module):
#     def __init__(self, class_weights):
#         super().__init__()

#         if not isinstance(class_weights, torch.Tensor):
#             class_weights = torch.tensor(class_weights, dtype=torch.float32)

#         # Normalize CE weights
#         class_weights = class_weights / class_weights.mean()
#         self.register_buffer("ce_weights", class_weights)

#         # Foreground dice weights (equal)
#         self.register_buffer("dice_weights", torch.ones(len(class_weights) - 1))

#         print("CE weights:", class_weights)
#         print("Dice weights (foreground only):", self.dice_weights)

#     def forward(self, outputs, targets):
#         outputs = outputs.float()
#         targets = targets.long()

#         ce_w = self.ce_weights.to(outputs.device)
#         dice_w = self.dice_weights.to(outputs.device)

#         # -----------------------------
#         # Cross-entropy
#         # -----------------------------
#         ce_loss = F.cross_entropy(outputs, targets, weight=ce_w)

#         # -----------------------------
#         # Dice using *label masks* (FAST)
#         # -----------------------------
#         probs = torch.softmax(outputs, dim=1)
#         num_classes = outputs.shape[1]

#         eps = 1e-5
#         dice_loss = 0.0
#         per_class_dice = []

#         # skip background class 0
#         for c in range(1, num_classes):
#             p = probs[:, c]                      # (B,H,W,D)
#             t = (targets == c).float()           # no one-hot

#             inter = (p * t).sum()
#             denom = p.sum() + t.sum()

#             dice_c = (2 * inter + eps) / (denom + eps)

#             w = dice_w[c - 1]
#             dice_loss += w * (1 - dice_c)
#             per_class_dice.append(dice_c.item())

#         total_loss = ce_loss + dice_loss

#         return total_loss, ce_loss.detach(), dice_loss.detach(), per_class_dice

class WeightedDiceCELoss3D(nn.Module):
    def __init__(self, class_weights, dice_weight=0.5):
        super().__init__()

        if not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)

        # Normalize CE weights
        class_weights = class_weights / class_weights.mean()
        self.register_buffer("ce_weights", class_weights)

        # Foreground dice weights (equal)
        self.register_buffer("dice_weights", torch.ones(len(class_weights) - 1))

        # ⭐ Store new dice weight
        self.dice_weight = dice_weight

        print("CE weights:", class_weights)
        print("Dice weights (foreground only):", self.dice_weights)
        print("Global dice_weight:", self.dice_weight)

    def forward(self, outputs, targets):
        outputs = outputs.float()
        targets = targets.long()

        ce_w = self.ce_weights.to(outputs.device)
        dice_w = self.dice_weights.to(outputs.device)

        # -----------------------------
        # Cross-entropy
        # -----------------------------
        ce_loss = F.cross_entropy(outputs, targets, weight=ce_w)

        # -----------------------------
        # Dice using masks (FAST)
        # -----------------------------
        probs = torch.softmax(outputs, dim=1)
        num_classes = outputs.shape[1]

        eps = 1e-5
        dice_loss = 0.0
        per_class_dice = []

        for c in range(1, num_classes):
            p = probs[:, c]
            t = (targets == c).float()

            inter = (p * t).sum()
            denom = p.sum() + t.sum()

            dice_c = (2 * inter + eps) / (denom + eps)

            w = dice_w[c - 1]
            dice_loss += w * (1 - dice_c)
            per_class_dice.append(dice_c.item())

        # ⭐ Apply dice_weight here
        total_loss = ce_loss + self.dice_weight * dice_loss

        return total_loss, ce_loss.detach(), dice_loss.detach(), per_class_dice


# class GeneralizedDiceCELoss3D(torch.nn.Module):
#     """
#     CE + Generalized Dice Loss (foreground only),
#     with manual CE weights and safe linear warm-up for Dice.
#     """
#     def __init__(self, ce_weights, temperature: float = 1.2, eps: float = 1e-6):
#         super().__init__()
#         if not isinstance(ce_weights, torch.Tensor):
#             ce_weights = torch.tensor(ce_weights, dtype=torch.float32)

#         # Apply manual scaling to emphasize rare classes
#         ce_w = ce_weights
#         ce_w = ce_w / ce_w.mean()
#         self.register_buffer("ce_weights", ce_w.float())

#         self.temperature = float(temperature)
#         self.eps = float(eps)
#         print("CE weights (manual + normalized):", self.ce_weights)

#     def forward(self, outputs, targets, dice_weight: float = 0.0):
#         outputs = outputs.float()
#         targets = targets.long()

#         # Cross-Entropy
#         ce_loss = F.cross_entropy(outputs, targets, weight=self.ce_weights)

#         # Generalized Dice Loss (foreground only)
#         temp_logits = outputs / self.temperature
#         probs = torch.softmax(temp_logits, dim=1)
#         num_classes = outputs.shape[1]

#         dice_numer, dice_denom = 0.0, 0.0

#         for c in range(1, num_classes):  # skip background
#             p_c = probs[:, c]
#             t_c = (targets == c).float()
#             v_c = t_c.sum()
#             w_c = 1.0 / (v_c + self.eps)

#             inter = (p_c * t_c).sum()
#             union = p_c.sum() + t_c.sum()

#             dice_numer += w_c * (2.0 * inter + self.eps)
#             dice_denom += w_c * (union + self.eps)

#         dice_coeff = dice_numer / dice_denom if dice_denom != 0 else torch.tensor(0.0, device=outputs.device)
#         dice_loss = 1.0 - dice_coeff

#         # Combine with linear warm-up multiplier
#         total_loss = ce_loss + dice_weight * dice_loss
#         return total_loss, ce_loss.detach(), dice_loss.detach(), dice_coeff.item()


class GeneralizedDiceFocalLoss3D(nn.Module):
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
# def dice_coeff(pred, target, num_classes=3, smooth=1e-5):
#     """
#     Compute per-class Dice coefficient
#     pred: (B,H,W,D)
#     target: (B,H,W,D)
#     """
#     dice = 0
#     pred_one_hot = F.one_hot(pred, num_classes).permute(0, 4, 1, 2, 3)
#     target_one_hot = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3)
#     for c in range(num_classes):
#         inter = torch.sum(pred_one_hot[:, c] * target_one_hot[:, c])
#         union = torch.sum(pred_one_hot[:, c]) + torch.sum(target_one_hot[:, c])
#         dice += (2 * inter + smooth) / (union + smooth)
#     return dice / num_classes

def dice_coeff(pred, target, num_classes=3, smooth=1e-5):
    """
    Fast Dice computation using label masks (no one-hot).
    pred:   (B,H,W,D)
    target: (B,H,W,D)
    """
    dice_scores = []

    for c in range(num_classes):
        p = (pred == c).float()
        t = (target == c).float()

        inter = (p * t).sum()
        denom = p.sum() + t.sum()

        dice = (2 * inter + smooth) / (denom + smooth)
        dice_scores.append(dice)

    return sum(dice_scores) / num_classes


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

# def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=5e-5, num_classes=3):
#     model = model.to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode='max',          # using val Dice as metric to maximize
#         factor=0.5,          # shrink LR by half
#         patience=5,          # wait 5 epochs of no improvement
#         min_lr=1e-5,         # prevents LR becoming tiny
#     )

#     # Normalized class weights (same as loss expects)
#     # class_weights = torch.tensor([1.02, 54.9, 237.0], dtype=torch.float32)
#     # class_weights = torch.tensor([0.9789, 0.0116, 0.0042], dtype=torch.float32)
#     # class_weights = class_weights / class_weights.mean()

#     freq = torch.tensor([0.9789, 0.0116, 0.0042], dtype=torch.float32)
#     inv_freq = 1.0 / freq
#     class_weights = inv_freq / inv_freq.mean()


#     criterion = WeightedDiceCELoss3D(class_weights, dice_weight=0.5).to(device)

#     history = {
#         "train_loss": [],
#         "train_ce": [],
#         "train_dice_loss": [],
#         "val_loss": [],
#         "val_dice": [],
#         "val_per_class_dice": []
#     }

#     def dice_weight_for_epoch(epoch):
#         """
#         Warm-up schedule:
#         - Epochs 0–2: No Dice
#         - Epochs 3–9: Dice = 0.3
#         - Epochs 10+: Dice = 0.5
#         """
#         if epoch < 3:
#             return 0.0
#         elif epoch < 10:
#             return 0.3
#         else:
#             return 0.5

#     best_dice = -1.0

#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0
#         train_ce_total = 0
#         train_dice_total = 0

#         for images, masks, patient_ids in tqdm(train_loader, 
#                                                desc=f"Epoch {epoch+1}/{num_epochs}", 
#                                                leave=False):
#             images = images.to(device)
#             masks = masks.squeeze(1).to(device)

#             optimizer.zero_grad()

#             with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
#                 outputs = model(images)

#                 # NEW loss signature
#                 loss, ce_loss_val, dice_loss_val, _ = criterion(outputs, masks)

#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()
#             train_ce_total += ce_loss_val.item()
#             train_dice_total += dice_loss_val.item()

#         # Store train stats
#         history["train_loss"].append(train_loss / len(train_loader))
#         history["train_ce"].append(train_ce_total / len(train_loader))
#         history["train_dice_loss"].append(train_dice_total / len(train_loader))

#         # ------------------------------------------------------------------
#         # Validation
#         # ------------------------------------------------------------------
#         model.eval()
#         val_loss = 0
#         all_dice = []
#         all_per_class = []

#         with torch.no_grad():
#             for images, masks, _ in val_loader:
#                 images = images.to(device)
#                 masks = masks.squeeze(1).to(device)

#                 outputs = model(images)

#                 loss, _, _, per_class_dice = criterion(outputs, masks)
#                 val_loss += loss.item()
#                 all_per_class.append(per_class_dice)

#                 # Compute dice on discrete predictions (optional)
#                 # probs = torch.softmax(outputs, dim=1)
#                 # preds = torch.argmax(probs, dim=1)
#                 preds = outputs.argmax(dim=1)

#                 dice_score = dice_coeff(preds, masks, num_classes=num_classes)
#                 all_dice.append(dice_score.item())

#         # Reduce per-class dice (list of lists → mean per class)
#         per_class_avg = np.mean(np.array(all_per_class), axis=0)

#         # Store validation stats
#         history["val_loss"].append(val_loss / len(val_loader))
#         history["val_dice"].append(np.mean(all_dice))
#         history["val_per_class_dice"].append(per_class_avg.tolist())

#         # Learning rate scheduler
#         scheduler.step(history["val_dice"][-1])
#         print(f"LR after epoch {epoch+1}: {optimizer.param_groups[0]['lr']:.6f}")

#         # ---------------------------
#         # Checkpointing (IMPORTANT)
#         # ---------------------------
#         if history["val_dice"][-1] > best_dice:
#             best_dice = history["val_dice"][-1]
#             torch.save(model.state_dict(), "best_model.pth")  # save checkpoint
#             print(f"🔥 Saved new best model (Dice={best_dice:.4f})")

#         print(f"Epoch [{epoch+1}/{num_epochs}] "
#               f"Train Loss: {history['train_loss'][-1]:.4f} "
#               f"CE: {history['train_ce'][-1]:.4f} "
#               f"Dice: {history['train_dice_loss'][-1]:.4f} "
#               f"Val Loss: {history['val_loss'][-1]:.4f} "
#               f"Val Dice: {history['val_dice'][-1]:.4f} "
#               f"Val Per-Class Dice: {per_class_avg.tolist()}")

#     return model, history


# def train_model(model, train_loader, val_loader, device, num_epochs=15, num_classes=3, lr=5e-5, max_dice_weight=0.6):
#     model = model.to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5
#     )

#     ce_weights = torch.tensor([0.1, 1.0, 2.0])
#     criterion = GeneralizedDiceCELoss3D(ce_weights).to(device)

#     history = {"train_loss": [], "train_ce": [], "train_dice_loss": [],
#                "val_loss": [], "val_dice": [], "val_per_class_dice": []}
#     best_dice = -1.0

#     for epoch in range(num_epochs):
#         # Linear warm-up for Dice weight
#         dice_weight = min(max_dice_weight, 0.2 + 0.2*epoch)  # hits 0.6 by epoch 2

#         model.train()
#         train_loss, train_ce, train_dice = 0.0, 0.0, 0.0

#         # Single tqdm for training batches
#         loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
#         for images, masks, _ in loop:
#             images, masks = images.to(device), masks.squeeze(1).to(device)
#             optimizer.zero_grad()

#             with torch.autocast(device_type='cuda'):
#                 outputs = model(images)
#                 loss, ce_val, dice_val, _ = criterion(outputs, masks, dice_weight=dice_weight)

#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()
#             train_ce += ce_val.item()
#             train_dice += dice_val.item()

#             loop.set_postfix({"Train Loss": train_loss / (loop.n+1), "Dice_w": dice_weight})

#         # Store training stats
#         history["train_loss"].append(train_loss / len(train_loader))
#         history["train_ce"].append(train_ce / len(train_loader))
#         history["train_dice_loss"].append(train_dice / len(train_loader))

#         # ----------------------------
#         # Validation
#         # ----------------------------
#         model.eval()
#         val_loss, val_dice_list, per_class_list = 0.0, [], []

#         with torch.no_grad():
#             for images, masks, _ in val_loader:
#                 images, masks = images.to(device), masks.squeeze(1).to(device)
#                 outputs = model(images)
#                 loss, _, _, per_class_dice = criterion(outputs, masks, dice_weight=dice_weight)
#                 val_loss += loss.item()
#                 per_class_list.append(per_class_dice)

#                 preds = outputs.argmax(dim=1)
#                 val_dice_list.append(dice_coeff(preds, masks, num_classes=num_classes).item())

#         per_class_avg = np.mean(np.array(per_class_list), axis=0)
#         val_dice_mean = np.mean(val_dice_list)

#         history["val_loss"].append(val_loss / len(val_loader))
#         history["val_dice"].append(val_dice_mean)
#         history["val_per_class_dice"].append(per_class_avg.tolist())

#         scheduler.step(val_dice_mean)
#         print(f"LR after epoch {epoch+1}: {optimizer.param_groups[0]['lr']:.6f}")

#         # Checkpoint if improved
#         if val_dice_mean > best_dice:
#             best_dice = val_dice_mean
#             torch.save(model.state_dict(), "best_model.pth")
#             print(f"🔥 Saved new best model (Dice={best_dice:.4f})")

#         print(f"Epoch [{epoch+1}/{num_epochs}] "
#               f"Train Loss: {history['train_loss'][-1]:.4f} CE: {history['train_ce'][-1]:.4f} "
#               f"DiceLoss: {history['train_dice_loss'][-1]:.4f} Val Loss: {history['val_loss'][-1]:.4f} "
#               f"Val Dice: {history['val_dice'][-1]:.4f} Val Per-Class Dice: {per_class_avg.tolist()} "
#               f"(dice_w={dice_weight:.3f})")

#     return model, history


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def show_epoch_preview(images, masks, outputs, epoch, slice_mode="middle"):
    """
    Visualizes one slice from the first item in the batch.
    Shows: original slice, ground truth, prediction.
    """
    cmap = ListedColormap(["black", "yellow", "red"])

    # pick first sample in batch
    vol = images[0,0].cpu()
    true_mask = masks[0].cpu()
    
    # predicted
    probs = torch.softmax(outputs[0], dim=0)
    pred_mask = torch.argmax(probs, dim=0).cpu()

    # Select a slice
    if slice_mode == "middle":
        mid = vol.shape[-1] // 2
    else:
        # Slice containing most foreground
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

    plt.show()


import torch
from torch import amp
import numpy as np
from tqdm import tqdm

def train_model(model, train_loader, val_loader, device,
                num_epochs=20, lr=5e-5, num_classes=3):

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6
    )

    # Stable class weights
    # You can plug your custom values here instead
    ce_weights = torch.tensor([0.1, 1.0, 2.0], device=device).float()

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

    for epoch in range(num_epochs):

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

        # Store train metrics
        n = len(train_loader)
        history["train_loss"].append(train_loss / n)
        history["train_ce"].append(train_ce / n)
        history["train_dice"].append(train_dice / n)

        # --------------------------
        # Validation
        # --------------------------
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

        # -----------------------------------------
        # Inline preview visualization (first batch)
        # -----------------------------------------
        model.eval()
        with torch.no_grad():
            val_images, val_masks, _ = next(iter(val_loader))
            val_images = val_images.to(device)
            val_masks = val_masks.squeeze(1).to(device)

            val_outputs = model(val_images)

        # Show inline preview
        show_epoch_preview(val_images, val_masks, val_outputs, epoch)


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

        # Save checkpoint
        if history["val_dice"][-1] > best_dice:
            best_dice = history["val_dice"][-1]
            torch.save(model.state_dict(), "best_model.pth")
            print(f"🔥 Saved new best model (Dice={best_dice:.4f})")

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
            images = images.to(device)           # (B, C, H, W, D)
            masks  = masks.squeeze(1).to(device) # remove channel dim if needed: (B, H, W, D)

            outputs = model(images)              # (B, num_classes, H, W, D)
            # probs = torch.softmax(outputs, dim=1)
            # preds = torch.argmax(probs, dim=1)  # (B, H, W, D)
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