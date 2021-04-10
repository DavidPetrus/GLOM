import numpy as np
import torch

from utils import resize_image, normalize_image, mask_random_crop

from absl import flags

FLAGS = flags.FLAGS


class Dataset(torch.utils.data.Dataset):

  def __init__(self, image_files, labels=None):

        #self.labels = labels
        self.image_files = image_files

  def __len__(self):
        return len(self.image_files)

  def __getitem__(self, index):
        # Select sample
        img_file = self.image_files[index]

        img = cv2.imread(img_file)
        img = resize_image(img)
        img = normalize_image(img)
        img = torch.from_numpy(img)
        masked_img, mask = mask_random_crop(img)
        net_input = torch.cat([masked_img, mask], dim=2)

        return net_input, img
