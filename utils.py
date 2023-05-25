import os
import torch
import torchvision
from dataset import niiDataset, KfoldDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import KFold

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def save_checkpoint(state, filename="unet_checkpoint.pth.tar"):
    filename = os.path.join(ROOT_DIR, filename)
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(full_dataset, batch_size, train_transform, val_transform,
                num_workers=4, pin_memory=False):

    kfold = KFold(n_splits=4, shuffle=True)
    train_loaders = []
    val_loaders = []

    IMG_DIR = os.path.join(ROOT_DIR, "data/IMG/ct_t4")
    MASK_DIR = os.path.join(ROOT_DIR, "data/MASK/t4")

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
    test_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.float().to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))

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

    print(f"Train Dice: {dice_score / len(loader)}")
    print(f"Train Loss: {test_loss / len(loader)}")  # 新增：输出平均测试损失
    np.save(os.path.join(ROOT_DIR, 'output/fold_change.npy'), np.array(fold_change))
    np.save(os.path.join(ROOT_DIR, 'output/dice_all.npy'), np.array(dice_all))
    model.train()

def save_predictions_as_imgs(loader, model, folder="output/images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, os.path.join(ROOT_DIR, f"{folder}/pred_{idx}.png"))
        
    model.train()