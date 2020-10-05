from torch.utils.data.dataset import Dataset
from PIL import Image
import os
import json
from lib import *
import albumentations as a
import numpy as np
from torchvision import transforms
import cv2
from torch import Tensor


class ButtDataset(Dataset):
    """
    Dataset for cigarette butt segmentation problem
    """
    def __init__(self, image_dir: str, coco_path: str, mode="trainval", augs=(), size=512):
        """
        :param image_dir: Images directory
        :param coco_path: Path to COCO annotations
        :param mode: "trainval" - train and validation (requires COCO annotations), "inference" - unlabeled
        :param augs: Augmentation pipeline
        :param size: Input image size
        """
        self.image_dir = image_dir
        self.coco_path = coco_path
        self.size = size
        assert mode in ["trainval", "inference"]
        self.mode = mode

        # image names
        files = os.listdir(self.image_dir)
        # COCO annotations
        if self.mode == "trainval":
            annotations = json.load(open(self.coco_path, "r"))
            self.annotations = annotations

        image_ids = [int(file.split(".")[0]) for file in files]

        self.files = files
        self.image_ids = image_ids

        self.augs = a.Compose(augs)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_name = self.files[index]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        image.load()
        image = transforms.Resize(size=self.size)(image)
        image = np.array(image)

        if self.mode == "trainval":
            mask = get_mask(image_id, self.annotations)
            mask = cv2.resize(mask, (self.size, self.size))
            image, mask = self.augs(image=image, mask=mask).values()
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)

            return image, mask

        if self.mode == "inference":
            image = self.augs(image=image)["image"]
            image = transforms.ToTensor()(image)
            return image, None

    def __len__(self):
        """
        :return: Number of images in dataset
        """
        return len(self.image_ids)
