import os
import random
from pathlib import Path
from typing import Tuple, Literal, Optional, Dict

import cv2
import numpy as np
import albumentations as A
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torchvision
from tqdm import tqdm

torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2 ** 32 - 1)
np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2 ** 32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2 ** 32 - 1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_FILE = Path(os.path.join(os.path.abspath(os.path.dirname(__file__)), "birds_model.chpt"))


def print_trainable_parameters(model: torch.nn.Module) -> None:
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
    )


def get_labels_dict(gt_file_path: str):
    img2class = dict()
    with open(gt_file_path) as f:
        next(f)  # skip first row
        for line in f:
            img_name = line.rstrip('\n').split(',')[0]
            bird_label = int(line.rstrip('\n').split(',')[1])
            img2class.update({img_name: bird_label})
    return img2class


class BirdDataset(Dataset):
    def __init__(self,
                 img_dir: str,
                 input_shape: Tuple[int, ...],
                 train_gt: Optional[Dict[str, int]] = None,
                 indexes: Optional[list[int]] = None,
                 mode: Literal["training", "validation", "inference"] = "training",
                 transform: Optional = None,
                 ):
        self.img_dir = img_dir
        self.img_names = sorted(os.listdir(img_dir))
        if indexes:
            self.images = [self.img_names[idx] for idx in indexes]
        self.classes = train_gt

        self.transform = transform
        self.normalize_transform = A.Compose([A.Normalize(always_apply=True)])

        self.input_shape = input_shape
        self.mode = mode

    def __len__(self):
        return len(self.img_names)

    def _resize_input(self, image: np.ndarray) -> np.ndarray:
        if image.shape[0] < self.input_shape[0] and image.shape[1] < self.input_shape[1]:
            image = cv2.resize(image, self.input_shape, interpolation=cv2.INTER_CUBIC)
        else:
            image = cv2.resize(image, self.input_shape, interpolation=cv2.INTER_AREA)
        return image

    def __getitem__(self, i):
        image: np.ndarray = cv2.imread(os.path.join(self.img_dir, self.img_names[i]), cv2.IMREAD_COLOR)
        image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert image.dtype == np.uint8

        image = self._resize_input(image)
        if self.transform:
            image = self.transform(image=image)["image"]
        image = self.normalize_transform(image=image)["image"]

        if self.mode in ["training", "validation"]:
            bird_class = self.classes[self.img_names[i]]
            return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1), torch.tensor(bird_class)
        else:
            return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)


def get_dataloaders(
        train_img_dir: str,
        train_gt: Dict[str, int],
        input_shape: Tuple[int, int] = (96, 96),
        batch_size: int = 64,
        split: float = 0.9,
        fast_train: bool = False) -> Tuple[DataLoader, DataLoader]:

    transforms = A.Compose([
        A.OneOf([
            A.HorizontalFlip(p=0.4),
            A.Rotate(p=0.6)
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(p=0.4),
            A.Sharpen(p=0.6),
        ], p=0.2),
        A.OneOf([
            A.ColorJitter(p=0.4),
            A.CLAHE(p=0.2),
            A.Solarize(p=0.2),
            A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=0.2)
        ], p=0.8),
    ], p=0.8)

    dataset_size = len(os.listdir(train_img_dir))
    indices = list(range(dataset_size))

    if fast_train:
        split = int(np.floor(0.1 * dataset_size))
    else:
        split = int(np.floor(split * dataset_size))

    np.random.shuffle(indices)
    if fast_train:
        train_indices, val_indices = indices[:split], indices[len(indices) - split:]
    else:
        train_indices, val_indices = indices[:split], indices[split:]

    train_dataset = BirdDataset(
        mode="training",
        input_shape=input_shape,
        img_dir=train_img_dir,
        train_gt=train_gt,
        indexes=train_indices,
        transform=transforms,
    )

    val_dataset = BirdDataset(
        mode="training",
        input_shape=input_shape,
        img_dir=train_img_dir,
        train_gt=train_gt,
        indexes=val_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False
    )

    return train_loader, val_loader


def get_model(num_classes: int, use_pretrained_weights: bool = True, unfreeze_last: int = 2) -> nn.Module:
    weights = None
    if use_pretrained_weights:
        weights = torchvision.models.EfficientNet_B3_Weights

    model = torchvision.models.efficientnet_b3(weights)

    for param in model.parameters():
        param.requires_grad = False

    child_ct = 0
    is_looping = True
    for child in list(model.children())[-2::-1]:
        for grand_child in list(child.children())[::-1]:
            for name, param in grand_child.named_parameters():
                param.requires_grad = True
            if child_ct == unfreeze_last:
                is_looping = False
                break
            child_ct += 1
        if not is_looping:
            break

    model.classifier = nn.Sequential(
        nn.Linear(1536, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )

    return model


def create_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss: float, epoch: int):
    if CHECKPOINT_FILE.is_file():
        os.remove(CHECKPOINT_FILE)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, CHECKPOINT_FILE)


def load_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int, float]:
    if not CHECKPOINT_FILE.is_file():
        print("No checkpoints detected! Starting training from scratch!")
        return model, optimizer, 0, torch.inf

    checkpoint = torch.load(CHECKPOINT_FILE, map_location=torch.device('cpu'))
    model.to("cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def train_classifier(train_gt: Dict[str, int],
                     train_img_dir: str,
                     fast_train: bool,
                     input_shape: Tuple[int, int] = (244, 244),
                     batch_size: int = 8,
                     lr: float = 3e-4,
                     num_epochs: int = 20,
                     enable_logging: bool = False,
                     enable_checkpointning: bool = False,
                     saving_steps: int = 50):
    train_loader, val_loader = get_dataloaders(train_img_dir, train_gt, input_shape, batch_size, fast_train=fast_train)

    use_pretrained_weights = False if fast_train else True
    model = get_model(50, use_pretrained_weights=use_pretrained_weights).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=5)

    if fast_train:
        enable_checkpointning = False
        enable_logging = False
        start_epoch = 0
        num_epochs = 1
    else:
        print("Trying to load model and optimizer from checkpoint...")
        model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer)

    if enable_logging:
        import wandb

    total_step = len(train_loader)
    print(f"Total training steps: {total_step * num_epochs}")
    print_trainable_parameters(model)

    print("Start training...")
    batch_ct = 0
    for epoch in range(start_epoch, num_epochs):
        # Training
        train_running_loss = []
        train_running_correct = 0
        with tqdm(train_loader, unit="batch", leave=True) as tqdm_epoch:
            for images, labels in tqdm_epoch:
                tqdm_epoch.set_description(f"Training epoch {epoch}")

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                # Forward
                outputs = model(images)
                loss = criterion(outputs, labels)
                train_running_loss.append(loss.item())
                _, preds = torch.max(outputs.data, 1)
                train_running_correct += (preds == labels).sum().item()
                batch_ct += 1

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                del images, outputs, labels

                if enable_checkpointning and batch_ct % saving_steps == 0:
                    model = model.to("cpu")
                    create_checkpoint(model, optimizer, loss.item(), epoch)
                    model = model.to(DEVICE)

        epoch_train_loss = sum(train_running_loss) / len(train_running_loss)
        epoch_train_acc = 100. * (train_running_correct / len(train_loader.dataset))
        print(F"AVERAGE TRAINING LOSS PER EPOCH {epoch}: {epoch_train_loss:.4f}")
        print(F"AVERAGE TRAINING ACCURACY PER EPOCH {epoch}: {epoch_train_acc:.4f}")
        if enable_logging:
            wandb.log({"Average training loss": epoch_train_loss, "Average training accuracy":epoch_train_acc}, step=epoch)

        # Validation
        model.eval()
        with torch.no_grad():
            val_running_loss = []
            val_running_correct = 0
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss.append(loss.item())
                _, preds = torch.max(outputs.data, 1)
                val_running_correct += (preds == labels).sum().item()

        epoch_val_loss = sum(val_running_loss) / len(val_running_loss)
        epoch_val_acc = 100. * (val_running_correct / len(val_loader.dataset))
        print(F"AVERAGE VALIDATION LOSS PER EPOCH {epoch}: {epoch_val_loss:.4f}")
        print(F"AVERAGE VALIDATION ACCURACY PER EPOCH {epoch}: {epoch_val_acc:.4f}")
        if enable_logging:
            wandb.log({"Average validation loss": epoch_val_loss, "Average validation accuracy": epoch_val_acc}, step=epoch)

        lr_scheduler.step(epoch_val_loss)
        if enable_logging:
            wandb.log({"lr": optimizer.param_groups[0]["lr"]})


def classify(model_filename: str, test_img_dir: str, input_shape: Tuple[int, int] = (244, 244), batch_size: int = 1):
    model = get_model(50, use_pretrained_weights=False)
    checkpoint = torch.load(model_filename, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)

    inference_dataset = BirdDataset(mode="inference", input_shape=input_shape, img_dir=test_img_dir)
    inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    model.eval()
    preds = {}
    with torch.no_grad():
        for i, img in enumerate(inference_dataloader):
            img = img.to(DEVICE)
            outputs = model(img).to("cpu")

            _, labels = torch.max(outputs.data, 1)
            for j in range(batch_size):
                preds[inference_dataset.img_names[i * batch_size + j]] = labels[j].numpy()
    return preds


if __name__ == "__main__":
#     import wandb
#
#     wandb.login(key=os.environ["WANDB_KEY"])
#     wandb.init(
#         project="BirdClassification",
#         resume="allow",
#         name="efficientNet_b3"
#     )
#
#     img2coords = get_labels_dict(r"./data/00_test_img_input/train/gt.csv")
#     train_classifier(train_img_dir=r"./data/00_test_img_input/train/images",
#                      train_gt=img2coords,
#                      input_shape=(244, 244),
#                      batch_size=64,
#                      lr=3e-4,
#                      num_epochs=16,
#                      fast_train=True,
#                      enable_logging=True,
#                      enable_checkpointning=True,
#                      )
#
    preds = classify("./birds_model.ckpt", r"./data/00_test_img_input/test/images")
    print(preds)
    print(type(preds))

