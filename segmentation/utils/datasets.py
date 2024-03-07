import json
import pickle
import random
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from utils.sample import DSample

from .points_sampler import MultiPointSampler


class ISDataset(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        augmentator=None,
        points_sampler=MultiPointSampler(max_num_points=24),
        min_object_area=1000,
        epoch_len=-1,
    ):
        super().__init__()
        self.epoch_len = epoch_len
        self.augmentator = augmentator
        self.min_object_area = min_object_area
        self.points_sampler = points_sampler
        self.to_tensor = transforms.ToTensor()

        self.dataset_samples = None

    def __getitem__(self, index):
        if self.epoch_len > 0:
            index = random.randrange(0, len(self.dataset_samples))

        sample = self.get_sample(index)
        sample = self.augment_sample(sample)
        sample.remove_small_objects(self.min_object_area)

        self.points_sampler.sample_object(sample)
        points = np.array(self.points_sampler.sample_points())
        mask = self.points_sampler.selected_mask

        output = {
            "images": self.to_tensor(sample.image),
            "points": points.astype(np.float32),
            "instances": mask,
        }

        return output

    def augment_sample(self, sample) -> DSample:
        if self.augmentator is None:
            return sample
        sample.augment(self.augmentator)
        return sample

    def get_sample(self, index) -> DSample:
        raise NotImplementedError

    def __len__(self):
        if self.epoch_len > 0:
            return self.epoch_len
        return self.get_samples_number()

    def get_samples_number(self):
        return len(self.dataset_samples)


class TestDataset(ISDataset):
    def __init__(self, images=None, masks=None, **kwargs):
        super().__init__(**kwargs)

        self._images_path = Path(images)
        self._insts_path = Path(masks)

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob("*.*"))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob("*.*")}

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[image_name.split(".")[0]])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = np.max(cv2.imread(mask_path).astype(np.int32), axis=2)
        instances_mask[instances_mask > 0] = 1

        return DSample(
            image, instances_mask, objects_ids=[1], sample_id=index, imname=image_name
        )


class CocoLvisDataset(ISDataset):
    def __init__(
        self,
        dataset_path,
        split="train",
        stuff_prob=0.0,
        allow_list_name=None,
        anno_file="hannotation.pickle",
        **kwargs,
    ):
        super().__init__(**kwargs)
        dataset_path = Path(dataset_path)
        self._split_path = dataset_path / split
        self.split = split
        self._images_path = self._split_path / "images"
        self._masks_path = self._split_path / "masks"
        self.stuff_prob = stuff_prob

        with open(self._split_path / anno_file, "rb") as f:
            self.dataset_samples = sorted(pickle.load(f).items())

        if allow_list_name is not None:
            allow_list_path = self._split_path / allow_list_name
            with open(allow_list_path, "r") as f:
                allow_images_ids = json.load(f)
            allow_images_ids = set(allow_images_ids)

            self.dataset_samples = [
                sample
                for sample in self.dataset_samples
                if sample[0] in allow_images_ids
            ]

    def get_sample(self, index) -> DSample:
        image_id, sample = self.dataset_samples[index]
        image_path = self._images_path / f"{image_id}.jpg"

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        packed_masks_path = self._masks_path / f"{image_id}.pickle"
        with open(packed_masks_path, "rb") as f:
            encoded_layers, objs_mapping = pickle.load(f)
        layers = [cv2.imdecode(x, cv2.IMREAD_UNCHANGED) for x in encoded_layers]
        layers = np.stack(layers, axis=2)

        instances_info = deepcopy(sample["hierarchy"])
        for inst_id, inst_info in list(instances_info.items()):
            if inst_info is None:
                inst_info = {"children": [], "parent": None, "node_level": 0}
                instances_info[inst_id] = inst_info
            inst_info["mapping"] = objs_mapping[inst_id]

        if self.stuff_prob > 0 and random.random() < self.stuff_prob:
            for inst_id in range(sample["num_instance_masks"], len(objs_mapping)):
                instances_info[inst_id] = {
                    "mapping": objs_mapping[inst_id],
                    "parent": None,
                    "children": [],
                }
        else:
            for inst_id in range(sample["num_instance_masks"], len(objs_mapping)):
                layer_indx, mask_id = objs_mapping[inst_id]
                layers[:, :, layer_indx][layers[:, :, layer_indx] == mask_id] = 0

        return DSample(image, layers, objects=instances_info)
