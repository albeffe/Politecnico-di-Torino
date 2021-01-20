# Politecnico di Torino
# 01TXFSM - Machine learning and Deep learning
# Homework 2
# Alberto Maria Falletta - s277971

from torchvision.datasets import VisionDataset
from PIL import Image
import os
import os.path
import sys


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset(in_dir, in_split, class_to_idx):
    """
    This function generates a list of PIL images and
    associated label in numeric format encoded by
    class_to_idx dictionary.
    The list is generated according to a file txt
    Class BACKGROUND_Google is filtered out.
    """

    images = []
    dir2 = os.path.expanduser(in_dir)  # Expands relative paths

    if in_split == "test":
        split_file = open("/content/Caltech101/test.txt", 'r')
        img_paths_list = split_file.readlines()
        split_file.close()
    else:
        split_file = open("/content/Caltech101/train.txt", 'r')
        img_paths_list = split_file.readlines()
        split_file.close()

    for img_path in img_paths_list:
        target = img_path.split('/')[0]

        if target != "BACKGROUND_Google":
            path = os.path.join('/content/Caltech101/101_ObjectCategories', img_path[:-1])  # removing '\n' character
            item = (pil_loader(path), class_to_idx[target])
            images.append(item)

    return images


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, split, class_to_idx)

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files"))

        self.split = split
        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in samples]

    def _find_classes(self, in_dir):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(in_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):

        image, label = self.samples[index][0], self.samples[index][1]
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        length = len(self.samples)
        return length
