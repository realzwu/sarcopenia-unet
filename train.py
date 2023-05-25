from albumentations.pytorch.transforms import F
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
from dataset import niiDataset
import os
import sys
from log import Logger

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.stdout = Logger(os.path.join(ROOT_DIR, "output/log_t4_2.txt"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hyperparameters etc.
LEARNING_RATE = 1e-5
BATCH_SIZE = 20
NUM_EPOCHS = 10
NUM_WORKERS = 8
# IMAGE_HEIGHT = 512
# IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = True
IMG_DIR = os.path.join(ROOT_DIR, "data/IMG/ct_t4")
MASK_DIR = os.path.join(ROOT_DIR, "data/MASK/t4")

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

    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    full_dataset = niiDataset(image_dir=IMG_DIR, mask_dir=MASK_DIR, transform=train_transform)

    train_loaders, val_loaders = get_loaders(full_dataset, BATCH_SIZE, train_transform,
                          val_transforms, NUM_WORKERS, PIN_MEMORY)

    if LOAD_MODEL:
        load_checkpoint(torch.load(os.path.join(ROOT_DIR, "unet_checkpoint.pth.tar")), model)


    for fold, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
        print(f"Fold: {fold + 1}/{4}")

        check_accuracy(val_loader, model, loss_fn, device=DEVICE)
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(NUM_EPOCHS):
            train_fn(train_loader, model, optimizer, loss_fn, scaler)

            # save model
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}

            save_checkpoint(checkpoint)
            check_accuracy(val_loader, model, loss_fn, device=DEVICE)  # 传递损失函数作为参数
            save_predictions_as_imgs(
              val_loader, model, folder=os.path.join(ROOT_DIR, f"output/images/fold{fold + 1}"), device=DEVICE)

if __name__ == "__main__":
    main()