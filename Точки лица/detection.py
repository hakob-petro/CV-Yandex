import os
import random
from pathlib import Path
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
CHECKPOINT_FILE = Path(os.path.join(os.path.abspath(os.path.dirname(__file__)), "facepoints_model.pt"))


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


class FacePointDataset(Dataset):
    def __init__(self,
                 img_dir: str,
                 input_shape: Tuple[int, ...],
                 coords_file: Optional[str] = None,
                 mode: Literal["training", "validation", "inference"] = "training",
                 transform: Optional = None,
                 split: float = 0.01,
                 fast_train: bool = False,
                 ):
        self.img_dir = img_dir
        self.img_names = sorted(os.listdir(img_dir))

        self.transform = transform
        self.normalize_transform = A.Compose([A.Normalize(always_apply=True)])

        self.input_shape = input_shape
        self.coords = dict()
        self.mode = mode

        if mode in ["training", "validation"]:
            with open(coords_file) as f:
                next(f)  # skip first row
                for line in f:
                    parts = line.rstrip('\n').split(',')
                    coords = [float(x) for x in parts[1:]]
                    coords = np.array(coords, dtype=np.float32)
                    self.coords.update({parts[0]: coords})

        if mode == "training":
            self.img_names = self.img_names[: int(split * len(self.img_names))]
        elif mode == "validation":
            if fast_train:
                self.img_names = self.img_names[int((1 - split) * len(self.img_names)):]
            self.img_names = self.img_names[int(split * len(self.img_names)):]

    def __len__(self):
        return len(self.img_names)

    def _resize_input(
            self,
            image: np.ndarray,
            curr_coords: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        if self.mode in ["training", "validation"]:
            curr_coords[0::2] *= self.input_shape[0] / image.shape[0]
            curr_coords[1::2] *= self.input_shape[1] / image.shape[1]

        if image.shape[0] < self.input_shape[0] and image.shape[1] < self.input_shape[1]:
            image = cv2.resize(image, self.input_shape, interpolation=cv2.INTER_CUBIC)
        else:
            image = cv2.resize(image, self.input_shape, interpolation=cv2.INTER_AREA)
        return image, curr_coords

    def __getitem__(self, i):
        image: np.ndarray = cv2.imread(os.path.join(self.img_dir, self.img_names[i]), cv2.IMREAD_COLOR)
        image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        assert image.dtype == np.uint8

        if self.mode in ["training", "validation"]:
            coords = self.coords[self.img_names[i]].copy()

            # First resize, then apply transformations
            image, coords = self._resize_input(image, coords)

            if self.transform:
                transform_res = self.transform(image=image, keypoints=coords.reshape(-1, 2))
                transform_coords = np.array(transform_res["keypoints"], dtype=np.float32)
                if len(coords) == transform_coords.size:
                    coords = transform_coords.flatten()
                    image = transform_res["image"]

            image = self.normalize_transform(image=image)["image"]
            return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1), torch.from_numpy(coords)
        else:
            scale_coeffs = (image.shape[0] / self.input_shape[0], image.shape[1] / self.input_shape[1])
            image, _ = self._resize_input(image)
            image = self.normalize_transform(image=image)["image"]
            return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1), scale_coeffs


class BottleneckResidualBlock(nn.Module):
    expansion: int = 4

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(BottleneckResidualBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.downsample = downsample
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.gelu(out)
        return out


class CustomResNet(nn.Module):
    def __init__(self,
                 layers, num_points: int = 28,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 init_residual: bool = False) -> None:
        super(CustomResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.norm_layer = norm_layer
        self.in_planes = 64

        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3),
            norm_layer(self.in_planes),
            nn.GELU()
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer_0 = self._make_layer(64, layers[0], stride=1)
        self.layer_1 = self._make_layer(128, layers[1], stride=2)
        self.layer_2 = self._make_layer(256, layers[2], stride=2)
        self.layer_3 = self._make_layer(512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(512 * BottleneckResidualBlock.expansion, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_points)
        )

        self.__init_layers(init_residual)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * BottleneckResidualBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes,
                    planes * BottleneckResidualBlock.expansion,
                    kernel_size=1,
                    stride=stride
                ),
                self.norm_layer(planes * BottleneckResidualBlock.expansion),
            )

        layers = [
            BottleneckResidualBlock(
                in_channels=self.in_planes,
                out_channels=planes,
                stride=stride,
                downsample=downsample,
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
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.normal_(m.bias)

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


def get_dataloaders(
        train_img_dir: str,
        coords_file: str,
        input_shape: Tuple[int, int] = (96, 96),
        batch_size: int = 64,
        fast_train: bool = False) -> Tuple[DataLoader, DataLoader]:

    transforms = A.Compose([
        A.OneOf([
            A.ZoomBlur(p=0.4),
            A.Sharpen(p=0.6),
        ], p=0.2),
        A.OneOf([
            A.ColorJitter(p=0.4),
            A.CLAHE(p=0.2),
            A.Solarize(p=0.2),
            A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=0.2)
        ], p=0.8),
    ], p=0.8, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    dataset_train = FacePointDataset(
        mode="training",
        input_shape=input_shape,
        img_dir=train_img_dir,
        coords_file=coords_file,
        transform=transforms,
        fast_train=fast_train,
        split=0.01 if fast_train else 0.9
    )

    dataset_val = FacePointDataset(
        mode="validation",
        input_shape=input_shape,
        img_dir=train_img_dir,
        coords_file=coords_file,
        fast_train=fast_train,
        split=0.01 if fast_train else 0.9
    )

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


def scale_predict(
        outputs: torch.Tensor, coords: torch.Tensor, input_shape: Tuple[int, int]) -> Tuple[torch.Tensor, ...]:
    scaled_outputs = outputs.clone().detach()
    scaled_coords = coords.clone().detach()
    scaled_outputs[:, 0::2] *= 100 / input_shape[0]
    scaled_outputs[:, 1::2] *= 100 / input_shape[1]
    scaled_coords[:, 0::2] *= 100 / input_shape[0]
    scaled_coords[:, 0::2] *= 100 / input_shape[1]
    return scaled_outputs, scaled_coords


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

    checkpoint = torch.load(CHECKPOINT_FILE)
    model.to("cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def train_detector(train_img_dir: str,
                   coords_file: str,
                   fast_train: bool,
                   input_shape: Tuple[int, int] = (100, 100),
                   batch_size: int = 64,
                   lr: float = 3e-4,
                   num_epochs: int = 20,
                   enable_logging: bool = False,
                   logging_steps: int = 20,
                   enable_checkpointning: bool = False,
                   saving_steps: int = 50):
    train_loader, val_loader = get_dataloaders(train_img_dir, coords_file, input_shape, batch_size, fast_train)

    model = CustomResNet([2, 2, 2, 2]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=30)

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
        with tqdm(train_loader, unit="batch", leave=True) as tqdm_epoch:
            for images, coords in tqdm_epoch:
                tqdm_epoch.set_description(f"Training epoch {epoch}")

                images = images.to(DEVICE)
                coords = coords.to(DEVICE)

                # Forward
                outputs = model(images)
                loss = criterion(outputs, coords)

                scaled_outputs, scaled_coords = scale_predict(outputs, coords, input_shape)
                error = criterion(scaled_outputs, scaled_coords)
                batch_ct += 1

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                del images, outputs, coords

                if enable_logging and (batch_ct + 1) % logging_steps == 0:
                    wandb.log({
                        "Train loss": loss.item(),
                        "Train error": error.item(),
                        "epoch": epoch
                    }, step=batch_ct + 1)
                    print(f''
                          f'Epoch: [{epoch}/{num_epochs - 1}],'
                          f' Training Loss: {loss.item():.4f},'
                          f' Training Error: :{error.item():.4f}')

                if enable_checkpointning and batch_ct % saving_steps == 0:
                    model = model.to("cpu")
                    create_checkpoint(model, optimizer, loss.item(), epoch)
                    model = model.to(DEVICE)

        # Validation
        with torch.no_grad():
            val_errors = []
            for i, (images, coords) in enumerate(val_loader):
                images = images.to(DEVICE)
                coords = coords.to(DEVICE)
                outputs = model(images)
                scaled_outputs, scaled_coords = scale_predict(outputs, coords, input_shape)
                loss = criterion(outputs, coords)
                error = criterion(scaled_outputs, scaled_coords)

                val_errors.append(error.item())

            if val_errors:
                if enable_logging:
                    wandb.log({"Average val error per epoch": sum(val_errors) / len(val_errors)}, step=batch_ct + 1)
                print("AVERAGE VALIADTION ERROR: {:.4f}".format(sum(val_errors) / len(val_errors)))

        lr_scheduler.step(sum(val_errors) / len(val_errors))
        if enable_logging:
            wandb.log({"lr": optimizer.param_groups[0]["lr"]})


def detect(test_img_dir: str, input_shape: Tuple[int, int] = (100, 100), batch_size: int = 1):
    model = CustomResNet([2, 2, 2, 2])
    checkpoint = torch.load(CHECKPOINT_FILE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)

    inference_dataset = FacePointDataset(mode="inference", input_shape=input_shape, img_dir=test_img_dir)
    inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

    model.eval()

    preds = {}
    with torch.no_grad():
        for i, (img, scale_coeffs) in enumerate(inference_dataloader):
            img = img.to(DEVICE)
            coords = model(img)
            coords = coords.to("cpu")
            coords[:, 0::2] *= scale_coeffs[:, 0]
            coords[:, 1::2] *= scale_coeffs[:, 1]
            for j in range(batch_size):
                preds[inference_dataset.img_names[i * batch_size + j]] = coords[j].numpy()
    return preds


if __name__ == "__main__":
    import wandb

    wandb.login(key=os.environ["WANDB_KEY"])
    wandb.init(
        project="FacePoints",
        resume="allow",
        name="lr_3e-2_b_32_steplr"
    )

    train_detector(train_img_dir=r"./data/00_test_img_input/train/images",
                   coords_file=r"./data/00_test_img_input/train/gt.csv",
                   input_shape=(100, 100),
                   batch_size=64,
                   lr=3e-2,
                   num_epochs=100,
                   fast_train=False,
                   enable_logging=True,
                   enable_checkpointning=True,
                   )
