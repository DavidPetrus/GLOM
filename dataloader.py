import numpy as np
import torch
import torch.nn.functional as F
import cv2
import time

from utils import resize_image, normalize_image, mask_random_crop, random_crop_resize, color_distortion

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
        self.batch_size = FLAGS.batch_size
        self.color_aug = color_distortion(FLAGS.aug_strength*FLAGS.brightness, FLAGS.aug_strength*FLAGS.contrast, \
                                          FLAGS.aug_strength*FLAGS.saturation, FLAGS.aug_strength*FLAGS.hue)

    def __len__(self):
        return len(self.image_files)//self.batch_size

    def __getitem__(self, index):
        '''crop_props = []
        for c in range(FLAGS.num_crops):
            crop_h = np.random.randint(FLAGS.min_crop//8,FLAGS.max_crop//8)
            crop_w = np.random.randint(FLAGS.min_crop//8,FLAGS.max_crop//8)
            resize_frac = np.random.uniform(FLAGS.min_resize,FLAGS.max_resize,size=(1,))[0]
            t_size = (max(int(crop_h*resize_frac),FLAGS.min_crop//8),max(int(crop_w*resize_frac),FLAGS.min_crop//8))
            crop_props.append(((crop_h,crop_w),t_size))'''

        img_batch = []
        augs_batch = [[] for c in range(FLAGS.num_crops)]
        all_dims = [[] for c in range(FLAGS.num_crops)]
        for i in range(self.batch_size):
            # Select sample
            img_file = self.image_files[self.batch_size*index + i]

            img = cv2.imread(img_file)[:,:,::-1]
            h,w,_ = img.shape
            aspect_ratio = h/w
            if max(h,w) > 512:
                if h > w:
                    img = cv2.resize(img, (int(512//aspect_ratio),512))
                else:
                    img = cv2.resize(img, (512,int(aspect_ratio*512)))

            img = img[:img.shape[0]-img.shape[0]%8,:img.shape[1]-img.shape[1]%8]
            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            img = img.movedim(2,0)

            for c in range(FLAGS.num_crops):
                #aug, crop_dims = random_crop_resize(img, crop_props[c][0], crop_props[c][1])
                aug, crop_dims = random_crop_resize(img)
                aug = normalize_image(aug)
                augs_batch[c].append(aug)
                all_dims[c].append(crop_dims)
            
            img = normalize_image(img)
            img_batch.append(img.unsqueeze(0))

        return img_batch, augs_batch, all_dims
