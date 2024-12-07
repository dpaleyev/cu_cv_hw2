import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple
from torchvision.transforms import TrivialAugmentWide, RandAugment, AutoAugment, AutoAugmentPolicy, AugMix

def load_prepared_data(data_dir: str) -> tuple:
    """
    Загружает подготовленные данные из указанной директории

    Параметры:
        data_dir (str): путь к директории с данными.

    Возвращает:
    tuple: кортеж из четырех элементов:
        - (train_images, train_labels): тренировочные изображения и метки
        - (val_images, val_labels): валидационные изображения и метки
        - test_images: тестовые изображения без меток
        - classes: названия классов
    """
    train_images = np.load(os.path.join(data_dir, "train_images.npy"))
    train_labels = np.load(os.path.join(data_dir, "train_labels.npy"))
    val_images = np.load(os.path.join(data_dir, "val_images.npy"))
    val_labels = np.load(os.path.join(data_dir, "val_labels.npy"))
    test_images = np.load(os.path.join(data_dir, "test_images.npy"))
    classes = np.load(os.path.join(data_dir, "classes.npy"))

    return (train_images, train_labels), (val_images, val_labels), test_images, classes

class QuickDrawDataset(Dataset):
    """
    Кастомный Dataset для работы с нашим набором данных Quick, Draw!

    Параметры:
        images (np.ndarray): массив изображений
        labels (np.ndarray | None): массив меток классов или None, если метки отсутствуют (например тестовые)
        transform (callable | None): преобразования (аугментации), которые применяются к изображениям (по умолчанию None)
    """

    def __init__(self, images: np.ndarray, labels: Optional[np.ndarray] = None, transform: Optional[callable] = None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        image = self.images[idx].reshape(28, 28)
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).unsqueeze(0)
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image

def get_transforms(hflip_prob=0.5, ta_wide=False, random_erase_prob=0.1):
    transforms = [T.ToTensor()]

    if hflip_prob > 0:
        transforms.append(T.RandomHorizontalFlip(hflip_prob))
    if ta_wide:
        transforms.append(TrivialAugmentWide(interpolation=T.InterpolationMode.BILINEAR))

    transforms.extend(
        [
            T.Normalize([0.5], [0.5]),
        ]
    )

    if random_erase_prob > 0:
        transforms.append(T.RandomErasing(p=random_erase_prob))

    return T.Compose(transforms)
