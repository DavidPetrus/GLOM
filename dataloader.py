import numpy as np
import torch
import cv2
import time

from utils import resize_image, normalize_image, mask_random_crop, random_crop_resize

from absl import flags

FLAGS = flags.FLAGS


class JHMDB_Dataset(torch.utils.data.Dataset):

  def __init__(self, video_files, granularity, labels=None):

        #self.labels = labels
        self.video_files = video_files
        self.granularity = granularity

  def __len__(self):
        return len(self.video_files)

  def __getitem__(self, index):
        f_ix = np.random.randint(0,10)
        count = 0
        # Select sample
        video_file = self.video_files[index]
        video_cap = cv2.VideoCapture(video_file)
        frames = []
        idx_offset = 0
        fails = 0
        while True:
            ret,frame = video_cap.read()
            if not ret:
                time.sleep(0.1)
                fails += 1
                if fails > 10:
                    idx_offset += 1
                    video_cap = cv2.VideoCapture(self.video_files[min(len(self.video_files)-1,index+idx_offset)])
                elif fails > 100:
                    break
                continue

            if count == f_ix:
                cropped,dims = random_crop_resize(frame)

                frame = normalize_image(frame)
                frame = torch.from_numpy(np.ascontiguousarray(frame))
                #frames.append(torch.movedim(frame,2,0))
                frame = frame.movedim(2,0)

                cropped = normalize_image(cropped)
                cropped = torch.from_numpy(np.ascontiguousarray(cropped))
                cropped = cropped.movedim(2,0)
                break

            count += 1

        return frame,cropped,dims


class ADE20k_Dataset(torch.utils.data.Dataset):

  def __init__(self, image_files, granularity, labels=None):

        #self.labels = labels
        self.image_files = image_files
        self.granularity = granularity

  def __len__(self):
        return len(self.image_files)

  def __getitem__(self, index):
        # Select sample
        img_file = self.image_files[index]

        img = cv2.imread(img_file)[:,:,::-1]
        img = resize_image(img, self.granularity[-1])
        img = normalize_image(img)
        img = torch.from_numpy(np.ascontiguousarray(img))
        masked_img, mask = mask_random_crop(img.detach().clone())
        net_input = torch.cat([masked_img, mask], dim=2)

        return torch.movedim(net_input,2,0), torch.movedim(img,2,0)
