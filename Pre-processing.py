import numpy as np
import os
from PIL import Image
import cv2


root_dir = "DATASETS/maps/train/"
folder = "DATASETS/maps"
list_files = os.listdir(root_dir)

for img_file in list_files:
  img_path = f"{root_dir}{img_file}"
  both_image = np.array(Image.open(img_path))
  image = both_image[:, :600, :]
  label = both_image[:, 600:, :]
  cv2.imwrite(f"{folder}/images/{img_file[:-4]}.png",image)
  cv2.imwrite(f"{folder}/labels/{img_file[:-4]}.png",label)