import os
import random
from typing import Tuple, Literal, Optional, Callable

import cv2
import numpy as np
import albumentations as A
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2 ** 32 - 1)
np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2 ** 32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2 ** 32 - 1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_NAME = "model.pt"


class FacePointDataset(Dataset):
    def __init__(self, img_dir: str, coords_file: str, input_shape: Tuple[int, ...],
                 mode: Literal["training", "validation", "inference"] = "training", transform: Optional = None,
                 train_val_split: float = 0.8, fast_train: bool = False, fast_train_val_split: float = 0.01):
        self.img_dir = img_dir
        self.img_paths = sorted(os.listdir(img_dir))
        self.transform = transform
        self.input_shape = input_shape
        self.coords = dict()
        self.mode = mode

        if mode != "inference":
            with open(coords_file) as f:
                next(f)  # to skip first row
                for line in f:
                    parts = line.rstrip('\n').split(',')
                    coords = [float(x) for x in parts[1:]]
                    coords = np.array(coords, dtype=np.float32)
                    self.coords.update({parts[0]: coords})

        if fast_train:
            self.img_names = self.img_paths[: int(fast_train_val_split * len(self.img_paths))]
        elif mode:
            self.img_names = self.img_paths[: int(train_val_split * len(self.img_paths))]
        else:
            self.img_names = self.img_paths[int(train_val_split * len(self.img_paths)):]

    def __len__(self):
        return len(self.img_names)

    def _resize_image_and_coords(self, image: np.ndarray, curr_coords: Optional[np.ndarray] = None) -> Tuple[
        torch.TensorType, Optional[np.ndarray]]:
        if self.mode != "inference":
            curr_coords[0::2] *= self.input_shape[0] / image.shape[0]
            curr_coords[1::2] *= self.input_shape[1] / image.shape[1]

        if image.shape[0] < self.input_shape[0] and image.shape[1] < self.input_shape[1]:
            image = cv2.resize(image, self.input_shape, interpolation=cv2.INTER_CUBIC)
        else:
            image = cv2.resize(image, self.input_shape, interpolation=cv2.INTER_AREA)

        return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1), curr_coords

    def __getitem__(self, i):
        image: np.ndarray = cv2.imread(os.path.join(self.img_dir, self.img_paths[i]), cv2.IMREAD_COLOR)
        image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # uint8 image

        if self.mode != "inference":
            curr_coords = self.coords[self.img_paths[i].split("\\")[-1].rstrip('\n')].copy()

            # Transforms can be applied only for training and validation
            if self.transform:
                transform_res = self.transform(image=image, keypoints=curr_coords.reshape(-1, 2))
                transformed_coords = np.array(transform_res["keypoints"], dtype=np.float32).flatten()
                if len(curr_coords) == len(
                        transformed_coords):  # Or we can process images with wrong coords separatly in collate_fn
                    curr_coords = transformed_coords
                    image = transform_res["image"]
            image, curr_coords = self._resize_image_and_coords(image, curr_coords)
            return image, curr_coords
        else:
            scale_coefs = (image.shape[0] / self.input_shape[0], image.shape[1] / self.input_shape[1])
            image, _ = self._resize_image_and_coords(image)
            return image, self.img_paths[i], scale_coefs


class BottleneckResidualBlock(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 skip_connection: Optional[nn.Module] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(BottleneckResidualBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.skip_connection = skip_connection
        self.stride = stride

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.GELU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            norm_layer(out_channels),
            nn.GELU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False),
            norm_layer(out_channels * self.expansion),
        )
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)

        if self.skip_connection is not None:
            residual = self.skip_connection(x)

        out += residual
        out = self.gelu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, layers, num_points: int = 28, norm_layer: Optional[Callable[..., nn.Module]] = None,
                 init_residual: bool = False) -> None:
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.in_planes = 64
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3),
            norm_layer(self.in_planes),
            nn.GELU())
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer_0 = self._make_layer(64, layers[0], stride=1)
        self.layer_1 = self._make_layer(128, layers[1], stride=2)
        self.layer_2 = self._make_layer(256, layers[2], stride=2)
        self.layer_3 = self._make_layer(512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512 * BottleneckResidualBlock.expansion, 512 * BottleneckResidualBlock.expansion),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(512 * BottleneckResidualBlock.expansion, num_points)
        )
        self.__init_layers()

    def _make_layer(self, planes, blocks, stride=1):
        skip_connection = None
        if stride != 1 or self.in_planes != planes * BottleneckResidualBlock.expansion:
            skip_connection = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * BottleneckResidualBlock.expansion, kernel_size=1, stride=stride),
                self.norm_layer(planes * BottleneckResidualBlock.expansion),
            )

        layers = [
            BottleneckResidualBlock(
                in_channels=self.in_planes,
                out_channels=planes,
                stride=stride,
                skip_connection=skip_connection,
                norm_layer=self.norm_layer
            )
        ]
        self.in_planes = planes * BottleneckResidualBlock.expansion

        for i in range(1, blocks):
            layers.append(BottleneckResidualBlock(self.in_planes, planes, norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def __init_layers(self, init_residual: bool = False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckResidualBlock) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.max_pool(self.conv_1(x))
        x = self.layer_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_dataloaders(train_img_dir: str, coords_file: str, input_shape: Tuple[int, int] = (96, 96),
                    batch_size: int = 64, fast_train: bool = True):
    transforms = A.Compose([
        A.Rotate(limit=30),
        A.OneOf([
            A.Blur(blur_limit=2, p=0.3),
            # A.OpticalDistortion(p=0.1, distort_limit=0.1, shift_limit=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5)
        ], p=0.3),
        A.HorizontalFlip(p=0.2),
        A.Equalize(p=0.2),
        A.Normalize(always_apply=True),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    dataset_train = FacePointDataset(
        mode="training", input_shape=input_shape,
        img_dir=train_img_dir, coords_file=coords_file,
        transform=transforms,
        fast_train=fast_train,
        fast_train_val_split=0.01 if fast_train else None
    )

    dataset_val = FacePointDataset(
        mode="validation", input_shape=input_shape,
        img_dir=train_img_dir, coords_file=coords_file)

    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count()
    )

    return train_loader, val_loader


def scale_predict(outputs: torch.Tensor, coords: torch.Tensor, input_shape: Tuple[int, int]) -> Tuple[
    torch.Tensor, ...]:
    scaled_outputs = outputs.clone().detach()
    scaled_coords = coords.clone().detach()
    scaled_outputs[:, 0::2] *= 100 / input_shape[0]
    scaled_outputs[:, 1::2] *= 100 / input_shape[1]
    scaled_coords[:, 0::2] *= 100 / input_shape[0]
    scaled_coords[:, 0::2] *= 100 / input_shape[1]
    return scaled_outputs, scaled_coords


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


def train_detector(train_img_dir: str, coords_file: str, checkpoint_dir,
                   input_shape: Tuple[int, int] = (96, 96),
                   batch_size: int = 64, lr: float = 3e-4,
                   num_epochs: int = 20, fast_train: bool = True, enable_logging: bool = False,
                   logging_steps: int = 10, enable_checkpointning: bool = False, saving_steps: int = 50):
    if fast_train and enable_checkpointning:
        enable_checkpointning = False
    if enable_logging:
        import wandb

    train_loader, val_loader = get_dataloaders(train_img_dir, coords_file, input_shape, batch_size, fast_train)

    model = ResNet([2, 2, 2, 2]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7)

    print_trainable_parameters(model)
    total_step = len(train_loader)
    print(f"Total training steps: {total_step * num_epochs}")

    batch_ct = 0
    for epoch in range(num_epochs):
        with tqdm(train_loader, unit="batch") as tqdm_epoch:
            for images, coords in tqdm_epoch:
                tqdm_epoch.set_description(f"Training epoch {epoch}")

                images = images.to(DEVICE)
                coords = coords.to(DEVICE)

                # Forward
                outputs = model(images)
                scaled_outputs, scaled_coords = scale_predict(outputs, coords, input_shape)
                loss = criterion(outputs, coords)
                error = criterion(scaled_outputs, scaled_coords)
                batch_ct += 1

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                del images, coords, outputs
                torch.cuda.empty_cache()

                if enable_logging and (batch_ct + 1) % logging_steps == 0:
                    wandb.log({
                        "Train loss": loss.item(),
                        "Train error": error.item(),
                        "epoch": epoch
                    }, step=batch_ct + 1)
                    print('Epoch [{}/{}], Loss: {:.4f}, Error: {:.4f}'
                          .format(epoch + 1, num_epochs, loss.item(), error.item()))

                if enable_checkpointning and batch_ct % saving_steps == 0:
                    model = model.to("cpu")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                    }, os.path.join(checkpoint_dir, CHECKPOINT_NAME))
                    model = model.to(DEVICE)

        # Validation
        with torch.no_grad(), tqdm(val_loader, unit="batch") as tqdm_val_epoch:
            for images, coords in tqdm_val_epoch:
                tqdm_val_epoch.set_description(f"Training epoch {epoch}")
                images = images.to(DEVICE)
                coords = coords.to(DEVICE)
                outputs = model(images)
                scaled_outputs, scaled_coords = scale_predict(outputs, coords, input_shape)
                loss = criterion(outputs, coords)
                error = criterion(scaled_outputs, scaled_coords)

                if enable_logging:
                    wandb.log({
                        "Val loss": loss.item(),
                        "Val error": error.item(),
                        "epoch": epoch
                    }, step=batch_ct + 1)
                    print('Epoch [{}/{}], Loss: {:.4f}, Error: {:.4f}'
                          .format(epoch + 1, num_epochs, loss.item(), error.item()))


if __name__ == "__main__":
    os.environ["WANDB_KEY"] = "your_key"
    os.environ["WANDB_PROJECT"] = "FacePoints_4"
    os.environ["NAME"] = "First launch"
    os.environ["WANDB_RUN_ID"] = os.getenv("NAME")
    os.environ["WANDB_RESUME"] = "allow"

    import wandb

    wandb.login(key=os.environ["WANDB_KEY"])

    wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        name=os.getenv("NAME"),
        resume=os.getenv("WANDB_RESUME"),
        id=os.getenv("WANDB_RUN_ID"),
    )

    train_detector(train_img_dir=r"./data/00_test_img_input/train/images",
                   coords_file=r"./data/00_test_img_input/train/gt.csv",
                   checkpoint_dir="./checkpoints",
                   input_shape=(128, 128),
                   batch_size=64,
                   lr=3e-4,
                   num_epochs=100,
                   fast_train=False,
                   enable_logging=True,
                   enable_checkpointning=False,
                   )
