# Politecnico di Torino
# 01TXFSM - Machine learning and Deep learning
# Homework 3
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


def make_dataset(in_dir, class_to_idx):
    """
    This function generates a list of PIL images and
    associated label in numeric format encoded by
    class_to_idx dictionary.
    The list is generated according to a file txt
    Class BACKGROUND_Google is filtered out.
    """

    instances = []
    in_dir = os.path.expanduser(in_dir)  # Expands relative paths

    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(in_dir, target_class)

        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = pil_loader(path), class_index
                instances.append(item)

    return instances


class Pacs(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(Pacs, self).__init__(root, transform=transform, target_transform=target_transform)

        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx)

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files"))

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
