import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import Augmentation

class Map(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.root_dir = root_dir
        self.transform = transform
        self.files = self.get_files(root_dir, split)
        if not self.files: raise Exception(f"No images found in {root_dir}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self):
        return len(self.files)

    def get_files(self, root_dir, split):
        root = Path(root_dir)
        with open(f"{root_dir}/{split}.txt") as f:
            lines = f.read().splitlines()

        file_names = [line.split(",")[-1].strip() for line in lines if line != ""]
        files = os.listdir(f"{root_dir}/images")
        files = list(filter(lambda x: x in file_names, files))
        files = [f"{root_dir}/images/{name}" for name in files]
        return files

    def __getitem__(self, index):
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('images', 'labels')
        image = np.array(Image.open(img_path))
        label = np.array(Image.open(lbl_path))

        if self.transform:
          image_transform = self.transform(image=image)
          label_transform = self.transform(image=label)
          image = image_transform["image"]
          label = label_transform["image"]
        return image, label

if __name__ == "__main__":
    dataset = Map(root_dir="DATASETS/maps", split="val", transform=test)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=5)
    image, label = next(iter(dataloader)) # batch size x channel x width x height
    images = torch.vstack([image, label])
    plt.imshow(make_grid(images, nrow=5).numpy().transpose((1, 2, 0)))
    plt.show()        