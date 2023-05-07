import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import AttentionUNet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs, combined_loss
)
from dataset import niiDataset
import sys
from log import Logger

sys.stdout = Logger()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {DEVICE}")

# Hyperparameters etc.
LEARNING_RATE = 1e-4
BATCH_SIZE = 3
NUM_EPOCHS = 2
NUM_WORKERS = 8
# IMAGE_HEIGHT = 512
# IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = False
IMG_DIR = "F:/sarcopenia-unet/data/npy/ct_t12"
MASK_DIR = "F:/sarcopenia-unet/data/npy/t12"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose(
      [
        # A.RandomSizedCrop(min_max_height=(256, 512), 
        # height=512, width=512, p=0.5),
        A.Normalize(mean=[0.0],std=[1.0],),
        ToTensorV2(),
      ],
    )

    val_transforms = A.Compose(
        [
            A.Normalize(mean=[0.0],std=[1.0],),
            ToTensorV2(),
        ],
    )

    model = AttentionUNet(img_ch=1, output_ch=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    full_dataset = niiDataset(image_dir=IMG_DIR, mask_dir=MASK_DIR, transform=train_transform)

    train_loaders, val_loaders = get_loaders(full_dataset, BATCH_SIZE, train_transform,
                          val_transforms, NUM_WORKERS, PIN_MEMORY)

    if LOAD_MODEL:
        load_checkpoint(torch.load("F:/sarcopenia-unet/unet_checkpoint.pth.tar"), model)

    for fold, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
        print(f"Fold {fold + 1}/{4}")

        check_accuracy(val_loader, model, combined_loss, device=DEVICE)
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(NUM_EPOCHS):
            train_fn(train_loader, model, optimizer, combined_loss, scaler)

            # save model
            checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
            }

            save_checkpoint(checkpoint)
            check_accuracy(val_loader, model, combined_loss, device=DEVICE)  # 修改：传递损失函数作为参数
            save_predictions_as_imgs(
                val_loader, model, folder=f"F:/sarcopenia-unet/output/images/fold{fold + 1}", device=DEVICE)

if __name__ == "__main__":
    main()