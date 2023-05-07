import torch
import torchvision
from dataset import niiDataset, KfoldDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset

def save_checkpoint(state, filename="unet_checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint["state_dict"])

def custom_bce_with_logits_loss(input, target):
    sigmoid_input = torch.sigmoid(input)
    bce_loss = -(target * torch.log(sigmoid_input) + (1 - target) * torch.log(1 - sigmoid_input))
    return bce_loss.mean()

def dice_loss(input, target, eps=1e-7):
    input = torch.sigmoid(input)
    intersection = torch.sum(input * target)
    union = torch.sum(input * input) + torch.sum(target * target)
    dice = (2. * intersection + eps) / (union + eps)
    return 1 - dice

def combined_loss(input, target, alpha=0.5):
    bce_loss = custom_bce_with_logits_loss(input, target)
    dice = dice_loss(input, target)
    return alpha * bce_loss + (1 - alpha) * dice

def get_loaders(full_dataset, batch_size, train_transform, val_transform,
                num_workers=4, pin_memory=False):

    kfold = KFold(n_splits=4, shuffle=True)
    train_loaders = []
    val_loaders = []

    IMG_DIR = "F:/sarcopenia-unet/data/npy/ct_t12"
    MASK_DIR = "F:/sarcopenia-unet/data/npy/t12"

    for fold, (train_indices, val_indices) in enumerate(kfold.split(full_dataset)):

        train_dataset = KfoldDataset(IMG_DIR, MASK_DIR, train_indices, transform=train_transform)
        val_dataset = KfoldDataset(IMG_DIR, MASK_DIR, val_indices, transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
          num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
          num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    return train_loaders, val_loaders

def check_accuracy(loader, model, loss_fn, device="cuda"):
    dice_score = 0
    model.eval()
    dice_all = []
    fold_change = []
    test_loss = 0.0  # 新增：用于累加测试损失

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.float().to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))

            # 新增：计算并累加测试损失
            batch_loss = loss_fn(preds, y)
            test_loss += batch_loss.item()

            preds = (preds > 0.5).float()
            dice_one = (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            dice_all.append(float(dice_one))
            dice_score += dice_one
            if (y == 1).sum() == 0:
                z = 0
            else:
                z = (preds == 1).sum() / (y == 1).sum()
            fold_change.append(float(z))

    print(f"Dice Score: {dice_score / len(loader)}")
    print(f"Test Loss: {test_loss / len(loader)}")  # 新增：输出平均测试损失
    np.save('F:/sarcopenia-unet/output/fold_change.npy', np.array(fold_change))
    np.save('F:/sarcopenia-unet/output/dice_all.npy', np.array(dice_all))
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="F:/sarcopenia-unet/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        
    model.train()