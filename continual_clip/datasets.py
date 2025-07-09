

import os
import torch.nn as nn

from continuum import ClassIncremental, InstanceIncremental
from continuum.datasets import (
    CIFAR100, ImageNet100, TinyImageNet200, ImageFolderDataset, Core50, CUB200, Food101,OxfordPet,Caltech101
)
from .utils import get_dataset_class_names, get_workdir
import pdb


class ImageNet_C(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)

    def get_data(self):
        self.data_path = self.data_path
        return super().get_data()


class ImageNet1000(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "val")
        return super().get_data()


class ImageNet_R(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)
    @property
    def transformations(self):
        """Default transformations if nothing is provided to the scenario."""
        return [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "test")
        return super().get_data()

class VTAB(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)
    @property
    def transformations(self):
        """Default transformations if nothing is provided to the scenario."""
        return [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "test")
        return super().get_data()


def get_dataset(cfg, is_train, transforms=None):
    if cfg.dataset == "cifar100":
        data_path = cfg.dataset_root
        dataset = CIFAR100(
            data_path=data_path, 
            download=True, 
            train=is_train, 
            # transforms=transforms
        )
        classes_names = dataset.dataset.classes
    elif cfg.dataset == "imagenet_R":
        data_path = cfg.dataset_root
        dataset = ImageNet_R(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
    elif cfg.dataset == "cub200":
        data_path = cfg.dataset_root
        dataset = CUB200(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
    elif cfg.dataset == "food101":
        data_path = cfg.dataset_root
        dataset = Food101(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
    elif cfg.dataset == "oxford_pet":
        data_path = cfg.dataset_root
        dataset = OxfordPet(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
    elif cfg.dataset == "caltech101":
        data_path = cfg.dataset_root
        dataset = Caltech101(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    elif cfg.dataset == "vtab":
        data_path = cfg.dataset_root
        dataset = VTAB(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
    elif cfg.dataset == "imagenet_c":
        data_path = cfg.dataset_root
        dataset = ImageNet_C(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    elif cfg.dataset == "tinyimagenet":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = TinyImageNet200(
            data_path, 
            train=is_train,
            download=True
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
        
    elif cfg.dataset == "imagenet100":
        data_path = cfg.dataset_root
        dataset = ImageNet100(
            data_path, 
            train=is_train,
            data_subset=os.path.join(get_workdir(os.getcwd()), "class_orders/train_100.txt" if is_train else "class_orders/val_100.txt")
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    elif cfg.dataset == "imagenet1000":
        data_path = cfg.dataset_root
        dataset = ImageNet1000(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    elif cfg.dataset == "core50":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = dataset = Core50(
            data_path, 
            scenario="domains", 
            classification="category", 
            train=is_train
        )
        classes_names = [
            "plug adapters", "mobile phones", "scissors", "light bulbs", "cans", 
            "glasses", "balls", "markers", "cups", "remote controls"
        ]
    
    else:
        ValueError(f"'{cfg.dataset}' is a invalid dataset.")

    return dataset, classes_names


def build_cl_scenarios(cfg, is_train, transforms) -> nn.Module:

    dataset, classes_names = get_dataset(cfg, is_train)

    if cfg.scenario == "class":
        scenario = ClassIncremental(
            dataset,
            initial_increment=cfg.initial_increment,
            increment=cfg.increment,
            transformations=transforms.transforms, # Convert Compose into list
            class_order=cfg.class_order,
        )

    elif cfg.scenario == "domain":
        scenario = InstanceIncremental(
            dataset,
            transformations=transforms.transforms,
        )

    elif cfg.scenario == "task-agnostic":
        NotImplementedError("Method has not been implemented. Soon be added.")

    else:
        ValueError(f"You have entered `{cfg.scenario}` which is not a defined scenario, " 
                    "please choose from {{'class', 'domain', 'task-agnostic'}}.")

    return scenario, classes_names